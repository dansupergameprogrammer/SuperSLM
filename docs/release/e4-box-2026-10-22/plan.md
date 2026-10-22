# SuperSLM — Precached Prefixes: Plan of Record

2026-10-22 · planner · **rev 1.18.12** · status: **rev 1.18.3 APPROVED by the principal 2026-10-20 22:52Z ("Go ahead with recommend": approval-summary-rev1.18.md, PC-14 (a) squash early, PC-14 (c) GPU twins labelled unreleased, PC-15 1.13.0 row dropped, all as recommended).** Earlier: **rev 1.4 APPROVED by the principal 2026-09-30 22:08 (PC-1 to PC-12
confirmed as recommended; D-SLM7972..D-SLM7983).** **Rev 1.18 folded the principal's own decisions, nothing else:
the nine build recommendations accepted 2026-10-17 14:51Z (PC-15), the release numbering and the squash to `main` of
2026-10-20 (PC-1 as renumbered, PC-14), and the two plugin API additions folded 2026-10-20 20:52Z (PC-16). Rev 1.18.1
folds its scoped re-check (plan review TEMPER, F1–F16; coverage audit, G1–G20 and N1–N6) and the conductor's rulings on
the open points (§15.19). Rev 1.18.2 folds the strike on rev 1.18.1 (FRACTURED: a deferred drop mark against a later
request whose hash is pending, and the drop's disk part against an earlier request it did not name; lesser planes
1–8) on the conductor's rulings (§15.20). The adversary's re-check of rev 1.18.2 HOLDS; rev 1.18.3 fixes its three
minor findings, and the conductor keeps the pass bound (§15.21, §13 item 21). Rev 1.18.4 is a text-only erratum on
the approved rev 1.18.3, from the suite audit of the E4, E5 and U2 red suite and the red suite's readings (§15.22,
§13 item 22). Rev 1.18.5 clarifies four approved rules from the U2 code review, with no design change (§15.23, §13
item 23). Rev 1.18.6 is an erratum from the box-fixes code review: 8.14's constructions are stated in their reachable
form (one page free, each request one page larger), and U6 carries the GPU host's held-release defect (§15.24, §13
item 24). Rev 1.18.7 records the principal's ruling of 2026-10-21 22:20Z: plugin 1.3.0 carries the CPU prefix cache
and the two schema skip-ahead options, and re-vendors engine 1.15.0 (§15.25). Rev 1.18.8 rules on U4's one-job
build reading. It trips §5.3 item 4's trigger, so a CPU cache build is now sliced into the prefill chunk jobs
`CreatePrefix` already uses, in plugin 1.3.0's scope (§15.27). Rev 1.18.9 corrects two receipts from the debunker's
post-M1 record. `main`'s "parentless 2026-10-12 roots" were a shallow clone's boundary, so M1's and M2's roots check
is replaced by an ancestry check run in a full clone. The fixture must refuse a shallow history, as §8 already says
(§15.28). Rev 1.18.10 restates cell 5.9 (a) as the chunk planner's own rule: a chunk above the share must be the
planner's floor (§15.29). Rev 1.18.11 redefines 5.9 (d) as the build's own chunk overruns, with a median-price grade
in place of "expected 0", and records the cache-chunk price as a calibration follow-up that does not block plugin
1.3.0 (§15.31). Rev 1.18.12: the `prefix_prefill` arm is timed under the build's own conditions, and a margin
factor is rejected (§15.32).** Gate loop on rev 1.4 closed (§13). Folded through rev 1.4: plan
review and coverage audit rounds 1–2; the strike on rev 1.1 (FRACTURED, colliding schema-name hashes); the re-strike
on rev 1.3 (held (i) and (ii); FRACTURED the admission claim for private nested restores, folded as text). **Rev
1.5** folded E0's re-anchor on paged KV's built code (§15.5). **Rev 1.6** folded the re-gate of rev 1.5 (§15.6).
**Rev 1.7** folds the scoped re-check of rev 1.6 (§15.7: plan review T-3026, temper; coverage audit T-3027, gaps
named). Every finding held at source and landed. The changes:
- the independent `SCM1` writer and the certified-artifact audit's commissioning move to the debunker, inside E0
  (§8, §9 E0 and E1; the conductor's ruling D-SLM7990);
- 7.17 checks each status name's stated ordinal, not "the last ordinal" (D-SLM7991, superseding D-SLM7989);
- 5.11 grades the pool-destroy status, so a leaked holder fails the root map case (D-SLM7992);
- 5.6's armed half arms only the allocation position (D-SLM7993);
- the fault slot gains an after-map must-reject and a fired-or-not record;
- the merge-base refusal is specified to be buildable, with its limit stated (D-SLM7994, refining D-SLM7988);
- 8.10's admission bound subtracts the pages 1.0.0's prefixes hold (D-SLM7995);
- the lane-park seam parks between jobs, behind a barrier that yields to teardown (D-SLM7996, superseding
  D-SLM7987's "between Layer-1 calls").

**Rev 1.8** folds the scoped rev 1.7 review T-3032, coverage re-audit T-3033, and reference-population audit
T-3031 (§15.8). T-3022's reference streams remain reproducibility evidence only (D-SLM7997); the test author
regenerates and commissions the discriminating population before E2 grades it. E0 re-runs the certified-artifact
audit after independent commissioning. The approved promise and PC decisions remain as filed.

**Rev 1.9** folds plan review T-3041 and coverage re-audit T-3042 (§15.9). The certified-artifact audit is
COMMISSIONED and its eight readings stand (T-3030, D-SLM8001); its zero-hash branch is commissioned with a
printable natural name (D-SLM8002). The independent `SCM1` writer is COMMISSIONED with the test author's
must-reject (T-3043, D-SLM8010). E0 records their actual commissioning order (§8, §9). The remaining corrections
pin the one-page overdraw construction, every reference-anchor movement (D-SLM8013), the E2 macro-vitality
cell, and the current-token oracle.

**Rev 1.10** resolves T-3060 F16 against the adopter's complete first-step path at `5d5ca58`: adoption copies
`hidden_scale`, but the first final norm ignores its incoming scale and the next embed replaces it. In-domain
`m`/`e` edits therefore leave adopter tokens unchanged (S20, §3.4, §7, §14). E1 authors the 6.4 isolated
`current_token` mutant; E2 runs it after unmutated 6.4 passes. E2 also runs the T-3060 F2–F15 suite
discrimination audit and its mutants against built verbs before E2 is green (§9, §15.10).

**Rev 1.11** folds plan review T-3062 and coverage re-audit T-3063 (§15.11). E1 closes only after the test author
has authored every [E2]–[E4] cell's complete deciding assertions, including assertions that stubbed verbs keep
from running, and the coverage auditor has rechecked that design against T-3060. E2 product implementation
starts after that gate. E2 then reaches those assertions on built verbs and records the independent `SPX1`
format-oracle and named mutant proofs before its green verdict; E3/E4 run their carried proofs at their own
gates. D-SLM8017 closes T-3046 W-1's E0 audit-order finding (§8, §9). Rev 1.11's delta re-enters scoped plan
review and coverage audit (§13 item 11).

**Rev 1.12** resolves T-3070 A1/A2 (§15.12). Cell 7.12 compares B=4/B=16 canonical blobs outside
their independently checked model-hash and digest fields, because B is part of the model artifact.
Cell 5.7 uses a validated constructed GPU-shaped operand at CPU-only E2; F2 captures and checks a real
GPU save. E1 authors both fixtures before its coverage recheck. The delta re-enters scoped plan review
and coverage audit (§13 item 12).

Through rev 1.12, no PC changes, and neither §3.4 *What is promised* nor §13's quoted promise changes. Rev 1.13 is
logged in §15.13, rev 1.14 in §15.14, rev 1.15 in §15.15, rev 1.16 in §15.16, rev 1.17 in §15.17, rev 1.18 in §15.18,
rev 1.18.1 in §15.19 (rev 1.18 is kept beside this file as `plan-rev1.18.md`), rev 1.18.2 in §15.20 (rev 1.18.1 is
kept beside this file as `plan-rev1.18.1.md`), rev 1.18.3 in §15.21 (rev 1.18.2 is kept beside this file as
`plan-rev1.18.2.md`), rev 1.18.4 in §15.22 (rev 1.18.3 is kept beside this file as `plan-rev1.18.3.md`), rev 1.18.5 in §15.23 (rev 1.18.4
is kept beside this file as `plan-rev1.18.4.md`), rev 1.18.6 in §15.24 (rev 1.18.5 is kept beside this file as
`plan-rev1.18.5.md`), rev 1.18.7 in §15.25 (rev 1.18.6 is kept beside this file as `plan-rev1.18.6.md`), rev 1.18.8 in §15.27 (rev 1.18.7
is kept beside this file as `plan-rev1.18.7.md`), rev 1.18.9 in §15.28 (rev 1.18.8 is kept beside this file as
`plan-rev1.18.8.md`), rev 1.18.10 in §15.29 (rev 1.18.9 is kept beside this file as `plan-rev1.18.9.md`), rev 1.18.11
in §15.31 (rev 1.18.10 is kept beside this file as `plan-rev1.18.10.md`), rev 1.18.12 in §15.32 (rev 1.18.11 is kept
beside this file as `plan-rev1.18.11.md`).
*Rev 1.18 changes promise (i)'s sentence*, on the principal's approval of 2026-10-17 (Q2): the page-size qualifier
rev 1.13 stated beside it is now in it (§0, §3.4, §13 item 2). Rev 1.17 is kept beside this file as `plan-rev1.17.md`. Earlier revisions are in this file's git history; every finding and where it landed is in §15. Builds on the
approved paged-KV plan of record, rev 16.1 (`/mnt/project-files/plans/superslm-paged-kv/plan.md`, cited below as
*PKV*), and changes nothing in it. Input: the research spike of 2026-09-30
(`/mnt/project-files/plans/superslm-prefix-precache-spike/spike-findings-2026-09-30.md`, bench beside it).
Source read for this plan: SuperSLM **`5d5ca58`**, the tip of paged KV's build-ahead branch
(`claude/shared-prefix-kv-build-wj8q9c`; `origin/main`@`5bd23c8` is its ancestor; `CMakeLists.txt` reads 1.12.0,
untagged). It replaces the "1.12.0 tagged" anchor, on the principal's 2026-10-01 instruction to build ahead of the
tag. Rev 1.4 read `origin/main`@`f4a1b6a`. SuperSLM-Unreal `main`@`7c58a92` (plugin 1.0.0, vendoring SuperSLM
v1.9.0); the project's own vendored copy has since moved to v1.11.0 (S14, U0). SuperSLM line numbers below are
`5d5ca58`'s, except one citation inside the promise text, which is left untouched (§15.5), and rev 1.13's, which are
`3b6631f`'s and say so. Plugin line numbers are `7c58a92`'s. E0's evidence: `e0-reanchor-pkv-5d5ca58.md`, beside this file.
*Rev 1.18 read:* SuperSLM `origin/main`@`2e12a6e`, `v1.13.1` (tag object `034a76a`, peeled `d697927`), `v1.13.0`
(tag object `9636d7b`, peeled `9e235965`), `claude/project-thread-wt2dpx`@`83a82c2`; Wizard
`claude/project-thread-wt2dpx`@`4c44fe9d9` for the plugin's `SuperSLMDetokenizer.h`; the build records under
`build-2026-10-17/` and `merge-path-2026-10-20.md`, beside this file.

## 0. Summary

**What it builds.** A frozen prefix (its K/V rows plus the state an adopter inherits) becomes bytes and comes back
as a frozen prefix handle without prefill. Every adopter of a restored prefix behaves bit-identically to an
adopter of the same prefix built by prefill, and a budget-mode adopter shares its pages exactly as it would share
a prefill-built prefix's (PKV §3.5). On top of that, the Unreal plugin keeps a prefix cache: resident frozen
handles under a memory budget with least-recently-used eviction, an asynchronous disk tier, and prefixes cooked at
build time and shipped with the game.

**The promise, exactly** (§3.4, *What is promised*; *rev 1.3*). For every blob this library's own save wrote,
under the same release version (or, between releases, the same build), adopters of the prefix it restores, in pools
of the restore pool's page size, emit exactly the tokens adopters of the prefill-built prefix emit in a pool of that
page size (*rev 1.18*, Q2: an adopter in a pool of another page size is refused `SSLM_INVALID_ARGUMENT`, for a
prefill-built prefix as for a restored one; save then restore is the way to move a prefix to another pool kind). This
rests on one model invariant that rev 1.3 adds
(PC-12): a model maps only if its schema names have distinct, non-zero name hashes. Any accidental
change to a blob is refused, by its size and magic checks or by its digest, and the plugin then recomputes. This is
integrity, not authenticity: a blob deliberately edited with its digest recomputed can be accepted, and in such a
blob every field not fixed by steps 2–6 is attacker-chosen within the limits steps 7–8 enforce (§3.4 names the
fixed fields and the chosen ones).

**Why.** The spike measured recompute at ~19 ms per token (0.5B, 24 layers, cloud Xeon, one thread) against
~0.5 ms per 1,000 tokens to copy K/V in from RAM: a 1,000-token prefix is ~20 s to rebuild and ~6 MiB to load.
Paged KV (PKV) already makes switching between *resident* prefixes cheap; nothing lets a prefix survive a level
load, a restart, or ship with the game. This plan adds that.

**Releases** (§9; decision PC-1, *renumbered by the principal 2026-10-20*, rev 1.18):
- **SuperSLM 1.14.0** — CPU verbs `sslm_prefix_save` / `sslm_prefix_restore` (built after 1.12.0; tagged after
  1.13.0). "After 1.12.0" means after 1.12.0's code: the build starts ahead of the 1.12.0 tag, on paged KV's
  build-ahead tip `5d5ca58`, on the principal's 2026-10-01 instruction (§8; *rev 1.6*). *Rev 1.18 (PC-14):* this
  plan's line lands on `main` as **one squash commit** (`main` was re-rooted on 2026-10-12 into parentless snapshot
  commits, so the line's history is not merged; *rev 1.18.9, §15.28:* that premise is false. The 2026-10-12
  "roots" were a shallow clone's boundary; on the remote each has parents, and `a5f3ad1` is an ancestor of `main`.
  The squash is still how PC-14 lands the line), on top of `main`'s 1.13.1. The squash carries the GPU twins' code
  too; in 1.14.0's tree they are declared and exported but labelled unreleased (§4 preamble; PC-14 (c), *rev
  1.18.1*: settled that way by the conductor's ruling, with the relabel covering every file and an instrument, 7.22).
- **SuperSLM 1.15.0** — schema fast-decode, a separate plan's release. *Rev 1.18.1 (plan review F4; audit N6,
  reading (a)):* rev 1.18 said "nothing of this plan ships in it", which is false once M1 lands: 1.15.0 is tagged from
  `main` after 1.14.0, so its tree carries 1.14.0's CPU verbs (released in 1.14.0, unchanged) and the GPU twins,
  still exported and labelled unreleased. Nothing of this plan is *released* in 1.15.0; its publish gate runs 7.22's
  label check too (a cross-plan note to that plan's owner is in `reviews/fold-rev1p18p1.md`).
- **SuperSLM 1.16.0** — GPU twins `sslm_gpu_prefix_save` / `sslm_gpu_prefix_restore` (after 1.14.0 and 1.15.0;
  rev 1.17 numbered them 1.15.0).
- **SuperSLM-Unreal 1.3.0** — the prefix cache on the CPU backend **and the two schema skip-ahead options**,
  canonical and exact, from the fast-decode plan's P1 and P1-X. It re-vendors **1.15.0** with
  `SUPERSLM_WITH_SCHEMA_JUMP_FORWARD` set to 1. *Rev 1.18.7, the principal 2026-10-21 22:20Z: "it should have both,
  the work is done".* Rev 1.18.6 said it vendored 1.14.0 and carried the cache alone. U0–U4 still build on 1.14.0.
  (Rev 1.17: 1.1.0; the paging release took plugin 1.2.0.)
- **SuperSLM-Unreal 1.4.0** — the GPU cache and nested cache entries, vendoring 1.16.0 (rev 1.17: 1.2.0; the next
  plugin number after 1.3.0). **Gated on the plugin's own move onto paged KV (budget-mode CPU sequences, pooled GPU
  sequences, GPU prefix verbs)** (decision PC-6, §5.1). *Rev 1.13:* that plan exists and ships first; its release is
  plugin 1.2.0.

**Steps** (§9): E0 re-anchor on paged KV's built code (`5d5ca58`, ahead of the 1.12.0 tag) → E1 CPU red suite → E2 save and private restore → E3 nested re-share →
E4 box timings and real-artifact oracle → E5 release 1.14.0 → F0 re-anchor on 1.13.0 → F1 GPU red suite → F2 GPU
twins → F3 GPU certification → F4 release 1.16.0 → U0 plugin re-pin to 1.14.0 → U1 plugin red suite → U2 cache
core and CPU tiers → U3 cook → U4 box runs → (engine 1.15.0 released, *rev 1.18.7*) → U5 plugin 1.3.0, re-vendoring
1.15.0 with the skip-ahead options → (after the plugin paging plan and 1.16.0) U6–U8 GPU
cache and nested entries → U9 plugin 1.4.0. *Rev 1.18:* a step **M1, the squash to `main`** (PC-14). *Rev 1.18.1
(the conductor's ruling on PC-14 (a)):* M1 runs **early, inside E2**, after E2's code review and its MSVC builds and
before E2's reference grading. This plan's work after M1 (E2's remaining receipts, E3–E5) runs on a **post-M1 branch
cut from `main` at the squash**, graded with `<line> = origin/main` (§8), and lands on `main` as a second squash,
**M2**, before E5's tag. *Rev 1.18.2:* `v1.14.0` is tagged on M2's squash commit, and M2 re-runs the release tree's
cells at the box.

**Where it stands** (*rev 1.18*; checked against git 2026-10-20, recorded in `merge-path-2026-10-20.md` beside this
file). Built on SuperSLM `claude/project-thread-wt2dpx` (tip `83a82c2`): E0–E3, and F0–F2 in the cloud on a software
GPU (lavapipe), not on either real GPU. The 2026-10-17 box queue has not run, so no E2, E3 or F2 box receipt exists.
Four changes have had no independent review: the 1.13.0 merge and re-anchor (`6426a72`, `3b6631f`), the shared
`SPX1` codec (`3a00abb`, which touches the CPU restore), and the GPU snapshot fixes `fdbb97b` and `83a82c2`. E4's
cells still need their rule-B re-derivation (10.1), and E5's cells 7.17, 9.9, 9.10 and the 9.5 pins are not written.
The plugin's cache (U1–U3, U6) is drafted, uncompiled and switched off on Wizard `claude/project-thread-wt2dpx`.

**Principal's time** (§11, underived): about 7.5–9 hours at the box through plugin 1.3.0 (*rev 1.18*: 7–8.5 before
the API additions), and 9.5–11.5 hours with the GPU cache, in runs of 20–120 minutes. *Rev 1.18:* the long legs (the 4.1 grid, 4.2/E3 on MSVC, the unpinned 1.5B
legs) run on the work machine, never on the home box (§1).

**Decisions for the principal** (§12 has each with its cost; recommendation after each):
- **PC-1..PC-4** are the defaults the principal already chose (sequencing, engine verbs, plugin cache, scope).
  *Confirm.* *Rev 1.18:* PC-1's release numbers are the principal's of 2026-10-20 (above).
- **PC-5** *(Rev 1.1: re-decided on the evidence, §15.)* One blob format for both backends. A blob restores only on
  the backend that wrote it, and a GPU blob only on the adapter vendor and device that wrote it, until F3's
  byte-identity cells show the rows equal across CPU and both GPUs. F4 then offers a one-line relaxation. *Confirm.*
- **PC-6** The plugin's move onto paged KV is its own plan; the CPU cache ships without it (adopters copy, no page
  sharing), the GPU cache waits for it. *Confirm, and commission that plan next.*
- **PC-7** The blob records its producer: the engine version and, on GPU, the adapter vendor and device. A different
  producer is refused with a new status, `SSLM_RESTORE_PRODUCER_MISMATCH`. This separates releases, not development
  builds on one version line; the promise is scoped to match (§3.4) (*rev 1.3*). *Confirm.*
- **PC-8** The blob carries a SHA-256 of itself, checked on every restore. *Confirm.*
- **PC-9** Only a frozen prefix can be saved. *Confirm.*
- **PC-10** The disk tier is off by default; cooked prefixes and the memory tier are on. *Confirm.*
- **PC-11** *(Rev 1.1; rev 1.2 adds the charge rule.)* The cache builds a missing prefix with
  `sslm_prefix_begin_budgeted(L)`, not legacy begin. Cached entries do not count against `PrefixBlockCount`, and
  the cache's byte budget is added to the pool at `Configure()`. Every cache build or restore is charged at its peak
  draw before it draws, and a request that cannot be charged resolves `BudgetExhausted` by name. *Confirm.*
- **PC-12** *(Rev 1.3; from the strike.)* In 1.14.0, a model whose schema names include two with equal `Fnv1a64`
  hashes, or one whose hash is 0 (the "unbound" sentinel), is refused at model map, and the converter refuses to
  emit one. This closes the planned `SPX1` paths and the shipped `SSB5` path with no blob-layout change. Cost: such
  a model, which maps today, is refused. *Confirm.*
- **PC-13** *(Rev 1.13; from the plugin paging plan's cross-plan note. **ACCEPTED by the principal 2026-10-15 14:55Z**, as decision 2 of the paging approval page.)* The CPU-cache release re-pins the plugin
  from 1.13.0 to 1.14.0. 1.14.0 keeps 1.13.0's sequence save formats (`SSB5`, `SSB6`, `SLM5`, `SLM6`) byte for byte,
  proved by cells, and the plugin's save wrapper accepts 1.13.0-tagged saves of those formats. Players' saves then
  cross the paging release → CPU-cache release boundary instead of being refused `Layer1Mismatch` (§5.11).
  *Accept.* *Rev 1.18:* the paging release (plugin 1.2.0) vendors **1.13.1** (`d697927`), so the predecessor these
  cells and the wrapper name is 1.13.1, and 1.13.1 keeps 1.13.0's formats (§3.8). *Rev 1.18.1 (conductor's ruling
  on PC-15 Q8):* the 1.13.0 compatibility row is **dropped**; no released plugin carried 1.13.0.
- **PC-14** *(Rev 1.18; **decided by the principal 2026-10-20**.)* The line lands on `main` as one squash commit (step
  M1). *Rev 1.18.1:* the conductor ruled the open points: (a) M1 runs early, inside E2, with post-M1 work on a branch
  cut from `main` and landed by M2; (b) is not a question: approved PC-7 already fixes 1.13.1 at the squash, E5 bumps;
  (c) the GPU twins are exported and labelled unreleased, every file, with 7.22's instrument at E5 and at 1.15.0.
- **PC-15** *(Rev 1.18; **accepted by the principal 2026-10-17 14:51Z**.)* The nine build recommendations Q1–Q9
  (`build-2026-10-17/open-questions.md`), each folded as recommended; §12 lists where each lands.
- **PC-16** *(Rev 1.18; **folded on the principal's word 2026-10-20 20:52Z**.)* Two additive plugin API additions:
  explicit residency control (warm, drop) and Prefix Set entries defined by text (§5.6, §5.8). Neither changes an
  existing promise; §5.6 names the one place the drop touches §5.11 (a) and why it holds. *Rev 1.18.2:* the strike on
  rev 1.18.1 fractured two of the residency rules. Both are folded on the conductor's rulings: a mark now waits behind
  earlier pending hashes in a stated pass order, and the disk part covers "in flight or waiting". A set registered
  under another model is refused (§5.6, §5.8, §15.20).

---

## Index

0. [Summary](#0-summary)
1. [Goal, scope and constraints](#1-goal-scope-and-constraints)
2. [Ground truth (verified at source)](#2-ground-truth-verified-at-source)
3. [Design — engine, CPU (1.14.0)](#3-design--engine-cpu-1140)
   - 3.1 What a frozen prefix is, and what the blob must carry
   - 3.2 The `SPX1` blob
   - 3.3 `sslm_prefix_save`
   - 3.4 `sslm_prefix_restore`: validation, draw, scatter
   - 3.5 Nested prefixes: re-sharing a live parent
   - 3.6 The restore matrix
   - 3.7 Concurrency
   - 3.8 ABI surface, statuses and mirrors
4. [Design — engine, GPU (1.16.0)](#4-design--engine-gpu-1160)
5. [Design — the plugin's prefix cache](#5-design--the-plugins-prefix-cache)
   - 5.1 Prerequisite: the plugin on paged KV
   - 5.2 The cache key
   - 5.3 Tiers
   - 5.4 Budget, eviction and pins
   - 5.5 Threading
   - 5.6 API
   - 5.7 The disk wrapper
   - 5.8 Cooking prefixes
   - 5.9 Every fallback
   - 5.10 Handle dispositions against the existing prefix verbs (*rev 1.1*)
   - 5.11 On the paging release: budget-mode adopters, region C, the save boundary (*rev 1.13*)
6. [Left out, and dated deferrals](#6-left-out-and-dated-deferrals)
7. [Coverage Model](#7-coverage-model)
8. [Instruments and references](#8-instruments-and-references)
9. [Build steps](#9-build-steps)
10. [Risks](#10-risks)
11. [Estimate](#11-estimate)
12. [Decisions](#12-decisions)
13. [Gate path](#13-gate-path)
14. [Not verified at source](#14-not-verified-at-source)
15. [Fold map](#15-fold-map)
   - 15.8 Round 8 (rev 1.8): the scoped re-check of rev 1.7 and E0 reference-population audit
   - 15.9 Round 9 (rev 1.9): the scoped re-check of rev 1.8 and executed E0 commissioning
   - 15.10 Round 10 (rev 1.10): F16 source resolution and E1/E2 gate sequencing
   - 15.11 Round 11 (rev 1.11): complete E1 red design before the E2 build
   - 15.12 Round 12 (rev 1.12): T-3070 A1/A2 fixture rulings
   - 15.13 Round 13 (rev 1.13): the E2 re-anchor on 1.13.0 and the paging plan's cross-plan notes
   - 15.14 Round 14 (rev 1.14): the scoped re-gate of rev 1.13
   - 15.15 Round 15 (rev 1.15): the re-check of rev 1.14
   - 15.16 Round 16 (rev 1.16): the strike on rev 1.15
   - 15.17 Round 17 (rev 1.17): the re-strike on rev 1.16
   - 15.18 Round 18 (rev 1.18): the principal's build answers, release numbering and API additions
   - 15.19 Round 19 (rev 1.18.1): the scoped re-check of rev 1.18 and the conductor's rulings
   - 15.20 Round 20 (rev 1.18.2): the strike on rev 1.18.1
   - 15.21 Round 21 (rev 1.18.3): the re-check of rev 1.18.2
   - 15.22 Round 22 (rev 1.18.4): erratum from the E4, E5 and U2 suite audit
   - 15.23 Round 23 (rev 1.18.5): clarifications from the U2 code review
   - 15.24 Round 24 (rev 1.18.6): 8.14's reachable construction; U6's held releases
   - 15.25 Round 25 (rev 1.18.7): plugin 1.3.0 carries the cache and the skip-ahead options
   - 15.26 U4 build erratum (2026-10-22): the packaged prefill count and the restore's per-MiB term
   - 15.27 Round 27 (rev 1.18.8): U4's one-job build reading trips the slicing trigger
   - 15.28 Round 28 (rev 1.18.9): the roots receipt and the shallow-history refusal
   - 15.29 Round 29 (rev 1.18.10): cell 5.9 (a) read as the chunk planner's rule
   - 15.31 Round 31 (rev 1.18.11): cell 5.9 (d), chunk overruns and the cache-chunk price
   - 15.32 Round 32 (rev 1.18.12): the `prefix_prefill` arm timed under the build's conditions
   - 15.30 Coverage re-audit erratum (2026-10-22): 11.1's `length < 0` exemption, 8.13's A22 legs, 5.10's mutant line

---

## 1. Goal, scope and constraints

**Goal.** A game switches between regularly used prefixes, and reloads them after a level load or a restart,
without paying prefill for them again, with output bit-identical to recomputing them.

**What it must deliver.**
- (a) An engine verb that writes a frozen prefix to a self-contained blob, and one that rebuilds a frozen prefix
  handle from that blob with no prefill, on CPU and on GPU.
- (b) Adopters of a restored prefix emit exactly the tokens adopters of the prefill-built prefix emit, and a
  budget-mode adopter shares the restored prefix's full pages (PKV §3.5).
- (c) A nested prefix (made by `begin_from`) may re-share a live parent's pages on restore, under the byte-compare
  identity rule PKV §3.7 uses for `sslm_seq_restore_shared`.
- (d) In the plugin: a memory tier with a byte budget and least-recently-used eviction, an asynchronous disk tier,
  and cooked prefixes shipped with the game; every mismatch or corruption falls back to recompute.
- (e) Nothing the cache does runs on the game thread except bookkeeping of the kind the plugin's `Tick()` already
  does (§5.5).

**Scope.** Engine verbs (CPU, then GPU) and the plugin cache. Not in scope (§6): automatic prefix matching (PKV §6
deferral stays), a pre-paging stopgap, the plugin's move onto paged KV (PC-6).

**Constraints.**

| Constraint | Source |
|---|---|
| Bit-exact integer engine; a restored prefix's rows and state are byte-equal to the saved prefix's | project convention; PKV §1 |
| C ABI additive only: new verbs, appended status ordinals, no changed signature | `sslm_abi.h:1-12`; PKV §1 |
| PKV rev 16.1 is approved and is not changed by this plan; where this plan needs a PKV structure it cites the PKV section and re-anchors on the built code (E0, F0) | the ask |
| CI runs only when the principal chooses; every commit this plan specifies carries `[skip ci]`, release tags excepted | project convention |
| Build, GPU and real-artifact runs on the principal's Windows box (3950X, RTX 2080 SUPER; RX 7900 XTX is the second certified GPU), as short resumable steps; CPU Linux builds may run in the cloud. *Rev 1.18:* legs longer than about 90 minutes (the 4.1 grid, ~7.5 h; 4.2 and E3 on MSVC, over 90 min; the unpinned 1.5B legs) run on the principal's work machine, never on the home box | project convention; the principal, 2026-10-20 |
| Each Unreal plugin depends only on its own parent library; plugin work stays off the game thread | project convention |
| Prefixes are base-only: no adapter enters a prefix or its blob | `sslm_abi.cpp:2290-2293` (prefix prefill passes a null adapter); PKV G10 |

---

## 2. Ground truth (verified at source)

Rev 1.5: SuperSLM rows read at `5d5ca58` (E0; rev 1.4 read `f4a1b6a`); plugin rows S15–S18 and S23 read at
SuperSLM-Unreal `7c58a92` and not re-read (SuperSLM-Unreal `main` is still `7c58a92`, tag `v1.0.0`, per E0's
`git ls-remote` of 2026-10-01).

| # | Fact | Where |
|---|---|---|
| S1 | *(Rev 1.5: paged.)* A prefix handle holds: model, pool, a page holder `kv` with `kv_mode`, `kv_budget`, `kv_origin` and `block_size`, `hidden_codes_storage` (int8, `hidden_size` bytes), a `SequenceLayerState` (hidden scale `{m, e}`, `layer_index`, `context_length`, the aggregate and four per-site saturation counts), `current_token` (−1 until the first prefill), `frozen`, `bound_schema`, `dfa_walk_state`. It holds no anti-LM state, no forced-token count and no `ready_for_logits` | `sslm_abi.cpp:282-322`; `forward_sites.h:477-533` |
| S2 | Prefix prefill runs whole tokens only (`PrefillWholeTokens`), so a prefix's `layer_index` is 0 between calls; PKV §3.5 states `layer_index` is always 0 for a frozen prefix | `PrefillWholeTokensImpl` `sslm_abi.cpp:1823` (`layer_index = 0` at `:1987`), wrapper `:2001`; `sslm_prefix_prefill` `:2274-2294`; PKV §3.5 |
| S3 | Adopt copies into the sequence: K/V, hidden codes and scale, `layer_index`, all five saturation counts, `context_length`, `current_token`; it derives `ready_for_logits = (current_token ≥ 0)`, clears the anti-LM state and sets `forced_token_count = 0`; it moves `dfa_walk_state` by the schema rule. So those fields are exactly what a prefix blob must carry for an adopter to be unchanged. *Rev 1.5:* the adopter's mode decides how K/V arrives: a budget adopter shares the prefix's pages and must use the same pool (`SSLM_INVALID_ARGUMENT` otherwise); a whole_reserve adopter copies. Adopt sets the adopter's `kv_origin` to the prefix's length | `sslm_abi.cpp:2528-2684`; mode `:2550-2551`; `kv_origin` `:2621`, `:2638`; `ready_for_logits` `:2673` |
| S4 | *(Rev 1.5: a PKV change.)* The first `sslm_prefix_freeze` sets `frozen` and returns the unmapped reserve to the pool (`HolderReturnReserve`), so a frozen prefix holds `ceil(L/B)` pages; later freezes are no-ops returning `SSLM_OK`. `sslm_prefix_set_schema` refuses once frozen or once `current_token != -1`; an empty frozen prefix (never prefilled) is legal to adopt | `sslm_abi.cpp:2301-2306`, `:2252-2265`; `sslm_abi_functions.inc:134-136` |
| S5 | Model identity in every CPU save is the artifact's 32-byte `RawIntegrityHash()` (SHA-256 of the artifact) plus `kv_precision`; mismatches return `SSLM_RESTORE_MODEL_MISMATCH` and `SSLM_RESTORE_KV_MISMATCH` | save `sslm_abi.cpp:3575-3580`; restore `:3760-3768`; `artifact.h:222` |
| S6 | Schema identity in a CPU save is `Fnv1a64(name)` (0 = none) plus the walk state; restore resolves it by name hash and refuses with `SSLM_RESTORE_SCHEMA_MISMATCH` an unknown hash, a walk state ≥ the schema's `state_count`, or a non-sentinel walk on an unbound blob — all before drawing any page | `sslm_abi.cpp:3778-3811` |
| S7 | *(Rev 1.5: built.)* Two CPU save formats are written. `SSB5` (156-byte fixed header, residual, anti-LM history, one whole block) for every holder not in budget mode. `SSB6` for budget-mode sequences only: `SSB5`'s header with every offset kept, then `kv_mode`@156, `budget`@160, `origin`@164 (172 bytes), then the residual, the anti-LM history, `kv_positions` (u64) = `L'`, and canonical rows `[run][pos < L'][d]`. No verb saves a prefix | format comment `sslm_abi.cpp:3469-3475`; `sslm_seq_save` `:3525-3671` (`write_ssb6` at `:3548`); reader `SeqRestoreImpl` `:3714-4146` |
| S8 | *(Rev 1.5: built.)* The status list ends at `SSLM_KV_BUDGET_EXCEEDED` = 29, then the `SSLM_STATUS_NEXT_FREE` sentinel. The next free ordinal is **30** | `include/superslm/sslm_abi.h:185-196` (the `SSLM_STATUS_ENUM_LIST` X-macro, `:104-196`) |
| S9 | `SslmGpuStatus` has no invalid-argument or schema status: GPU restore returns `SSLM_SEQUENCE_KV_BUFFER_MISMATCH` for a malformed blob and for an unresolvable schema or walk state, `SSLM_RESTORE_MODEL_MISMATCH` for another model, `SSLM_OUTPUT_BUFFER_TOO_SMALL` for a short save buffer. `gpu_1p0.cpp` is unchanged since `f4a1b6a`; `gpu_1p0.h` gained one comment line | `gpu_1p0.h:96-270`, `:395-421`; `gpu_1p0.cpp:2707-2733` |
| S10 | GPU `SLM5` carries the model hash, the five saturation counts, the schema index and walk, and `ready_for_logits`; GPU schemas are indexed by the same model's schema table (same artifact hash ⇒ same index order) | `superslm_gpu.cpp:4566-4606`; `gpu_1p0.cpp:2713-2735` |
| S11 | The engine has a SHA-256 with runtime SHA-NI dispatch, `superslm::Sha256Hash(data, len, out)` | `include/superslm/sha256.h:41`, `:77-94`; `src/sha256.cpp` |
| S12 | The engine has no version constant in code: the version exists only as `project(superslm VERSION 1.12.0)` in CMake (*rev 1.5*; untagged at `5d5ca58`; the release step bumps it). The plugin builds the vendored sources with Unreal's toolchain, so a CMake-only definition would not reach it | `CMakeLists.txt:2`; grep of `include/` and `src/` for a version symbol (none, no `sslm_version.h`); `VENDORED_VERSION.txt` |
| S13 | *(Rev 1.5: superseded, as expected.)* PKV's CPU slice is built: `src/kv_pages.{h,cpp}`, the budget verbs, `begin_from`, `SSB6`, `sslm_seq_restore_shared`, the stats verbs and the api-slot census entry. PKV's GPU prefix verbs and `SLM6` are not built | `sslm_abi_functions.inc:85-163`; `tests/t2807-api-slot/c_abi_verbs.txt:45`; grep of `src/`, `include/` for `sslm_gpu_prefix` (none) |
| S14 | Plugin 1.0.0 vendors SuperSLM v1.9.0. Its save wrapper (`'SLUW'` v2) tags backend, the Layer-1 tag and commit, and the artifact hash; a blob saved under another Layer-1 pin is refused (`Layer1Mismatch`) before anything is queued. *Rev 1.5:* the project's own vendored copy (Wizard `Plugins/SuperSLMUnreal`, `develop`@`4dffa209`) is at SuperSLM **v1.11.0** (`8d85b7b`), re-pinned 2026-09-30; U0 names which tree it re-pins (§9) | `SuperSLMSaveBlob.h`; `SuperSLMSaveRestoreTypes.h:12`, `:76`; `VENDORED_VERSION.txt` (both trees) |
| S15 | Plugin CPU prefixes: `CreatePrefix(tokens)` (legacy `sslm_prefix_begin`, prefill on worker lanes, freeze), `AdoptPrefix` (queued, a whole-block copy today), `ReleasePrefix` (refused while an adopt of it is queued); live prefixes are capped by `PrefixBlockCount`, and the pool is `BlockCount + PrefixBlockCount` blocks | `SuperSLMSubsystem.h:267-291`; `SuperSLMRuntimeConfig.h:57-58`; `SuperSLMSubsystem.cpp:1909`, `:3292`, `:3897` |
| S16 | Plugin GPU backend has no prefix creation, and `RequestAdoptPrefix` resolves `UnsupportedOnGpu` | `SuperSLMGpuSubsystem.h:431-441`; `SuperSLMSaveRestoreTypes.h:60` |
| S17 | Every plugin Layer-1 call runs on plugin-owned worker threads; `Tick()` does plan and apply bookkeeping only; plugin already uses `Async(EAsyncExecution::ThreadPool, …)` for off-thread work; `SuperSLM::ComputeArtifactDigest` is Layer 1's SHA-256 | `SuperSLMSubsystem.h:190-203`; `SuperSLMSubsystem.cpp:2052`; `SuperSLMIntegrity.h:8-20` |
| S18 | The plugin has an Editor module (`SuperSLMUnrealEditor`) beside the Runtime module | `SuperSLMUnreal.uplugin` |
| S19 | GPU restore today validates the schema after creating the fresh handle and releases it on refusal (it does allocate first) | `gpu_1p0.cpp:2707-2733` |
| S20 | *(Rev 1.10, T-3060 F16: source-settled.)* `SSB5`/`SSB6` restore reads the hidden scale `m` and `e` with no bound (and `current_token` with none beyond the pending-embed rule). The forward funnel refuses an incoming scale whose `m` does not fit int32 (`CarriedScaleMantissaOutOfDomain`); `e` saturates when scales combine. Adoption copies the prefix's `hidden_scale` into the sequence. For a nonempty frozen prefix, `ready_for_logits` sends the first decode directly to `RmsNormSite`: its `incoming_scale` parameter is unused, its funnel receives an empty incoming span, and logits receive only the norm's output codes. The next decode embeds the produced token and replaces `hidden_scale`. An empty prefix has no ready-for-logits state and embeds before its first norm; a `begin_from` child also embeds on prefill. Thus no adopter token depends on a frozen prefix's `m` or `e`, even though those fields are copied. Every scale the forward stores, the embed's included, has passed the funnel's int32 check, so save never writes an `m` outside int32. The funnel's int32 refusal is pinned by the engine's own cells (`TestRequantChainCheckedRejectsIncomingFactorMantissaOutOfInt32Domain`, `TestRequantChainCheckedRunningProductOutOfInt32DomainIsRejectedNotTruncated`, `TestResidualReconcileSiteStepZeroGuardIsLoadBearing`), not by a cell of this plan | `sslm_abi.cpp:3817-3818`; `checked_chain_funnel.h:74`, `:430-432`; `checked_chain_funnel.cpp:199`, `:327-331`; adopt `sslm_abi.cpp:2655-2673`; decode `:3012-3019`, `:3052-3063`, `:3093-3110`, `:3227`, `:3291`; `forward_sites.cpp:837-896`; prefill `sslm_abi.cpp:1943-1987`, `:2228-2238`; funnel cells `tests/test_main.cpp:9125`, `:9065`, `:13848` |
| S21 | *(Rev 1.1; unchanged at `5d5ca58`.)* CPU/GPU K/V byte identity: `docs/platform-support.md:311-315` says every layer's K/V state is bit-identical to the CPU reference on both certified GPUs. The evidence at source is the cross-vendor battery's `c5` cell (`tools/t2039_c5_harness.cpp:201-215`, pinned `RESULT=BIT-IDENTICAL` in `tools/t2116_crossvendor/pins/pins.txt`). It compares every layer and KV head at **position 0 only**, for one token on the real 1.5B artifact, in a run-only package outside CI. `test_main.cpp`'s GPU comparisons (`:24697-24920`) are synthetic and single-position. No cell compares a multi-position span of K/V bytes across CPU and GPU, or across the two GPUs. *Rev 1.2:* the same doc's "the full 45,845-token argmax sequence" (`:315`) is the single-token run's argmax **token id** (`pins.txt:9`, `ARGMAX_CPU=45845`), not a sequence length, so the public claim is weaker than it reads | as cited |
| S22 | *(Rev 1.1; rev 1.2 corrected; unchanged at `5d5ca58`.)* GPU batched prompt prefill, real 1.5B, public entry point: 61.96 tok/s (128 tokens) and 61.93 tok/s (256) on the RTX 2080 SUPER; 73.44 tok/s (256) on the RX 7900 XTX. The 7.19× beside them is batched prefill over the **pre-batching per-token prefill path** (8.62 → 61.96 tok/s), and batched prefill runs at about the decode rate (62.79–63.11 tok/s at 1.5B). No 0.5B GPU prefill figure is published; the spike cites ~8.4 ms per token for GPU decode at 0.5B | `docs/platform-support.md:295`, `:324-336`, `:334`; spike §2 |
| S23 | *(Rev 1.1.)* Plugin: `CreatePrefix` refuses when `S.Prefixes.Num() >= PrefixBlockCount`; `ReleasePrefix` releases any live entry of `State->Prefixes` whose adopts have drained; no plugin code calls `sslm_prefix_set_schema`; schemas resolve by name through `sslm_schema_lookup` | `SuperSLMSubsystem.cpp:2406`, `:2499-2542`; grep of `Source/SuperSLMUnreal`; `SuperSLMRuntimeRegistry.cpp:543` |
| S25 | *(Rev 1.3; from the strike.)* Schema identity is lossy. The schema-section parser refuses duplicate exact names (`DuplicateSchemaName`) but not duplicate name hashes. `ByNameHash` returns the **first** entry with a matching hash, and the source says a 64-bit collision "is not specially handled". The shipped `SSB5` restore resolves through it. Two distinct printable names with equal `Fnv1a64` exist: `P1z16@?^_s(Q` and `abcdefgh`, both `0x25da8c1836a8d66d` (the strike's construction; reproduced). The save sentinel for "unbound" is hash 0, and no rule stops a non-empty name hashing to 0. *Rev 1.5:* the GPU model maps schemas through the same parser | `schema_masks.h:176-186`, `:300-307`; restore `sslm_abi.cpp:3783`; GPU `gpu_1p0.cpp:1247` |
| S24 | *(Rev 1.1.)* Both SHA-256 dispatch arms (portable and SHA-NI) are pinned to the FIPS vectors; the D3D12 harness reads `DXGI_ADAPTER_DESC1` (which carries `VendorId` and `DeviceId`) | `tests/sha256_portable_forced_tests.cpp`; `d3d12_harness.h:429-485` |
| S26 | *(Rev 1.5, E0.)* `kv_pages` has no single draw-and-share primitive. `HolderCreate` draws a reserve under one pool-mutex section; `HolderShareLeading` shares under a second. The share can refuse after the draw: `CheckShareableLocked` refuses a source page at refcount `UINT32_MAX` (`kInvalidArgument` → `SSLM_INVALID_ARGUMENT`) or at 0 (`kRefcountUnderflow` → `SSLM_ARTIFACT_REJECTED`, reachable only through caller UB). `HolderShareLeading` requires `dst->mapped == 0`, so the share precedes any private page being mapped. PKV's own restore and `begin_from` take this shape: draw, share, release the holder on a share refusal, then `MapPagesFor`. No ABI seam reaches the `UINT32_MAX` refusal. *Rev 1.6:* `MapPagesFor` has two refusals of its own, checked before it maps anything: `SSLM_INVALID_ARGUMENT` when the pages needed exceed the table's entries, and `SSLM_KV_POOL_EXHAUSTED` when the reserve is short. PKV's restore releases the holder on either (`:4032-4037`). PKV pins the short-reserve branch through a test-only seam, `sslm_pkv_test_only_drain_reserve` (its cell 11.2), and calls the past-the-table branch unreachable. It is: `B` divides `context_cap` by construction, so a table of `cap_pages = context_cap / B` entries holds `ceil(L/B)` for every `L ≤ context_cap`. *Rev 1.13 (re-anchor F4):* at 1.13.0, `B` need not divide the cap; the table has `cap_pages = ceil(context_cap/B)` entries, and `ceil(L/B) ≤ ceil(context_cap/B)` for every `L ≤ context_cap`, so the branch stays unreachable (S30). The page module's test-only code compiles under `SUPERSLM_ENABLE_KV_PAGES_TEST_SEAMS` | `kv_pages.cpp:148-181`, `:198-217`, `:93-100`; `sslm_abi.cpp:1577-1594` (`MapPageStatus`), `:1611-1618` (`CreateHolder`), `:1641-1653` (`MapPagesFor`), `:3997-4038` (restore), `:2210-2219` (`begin_from`); geometry `:669-697`; drain seam `:4626-4632`, `tests/paged-kv/pkv_abi.h:114-116`; `kv_pages.cpp:441-464` |
| S29 | *(Rev 1.6.)* The fill probe has three usable shapes. `ProbeSplit` refuses `k < 2` with a failed check, never a loop. The two-sided `ProbeExactlyFree` reads exactly `k ≥ 2` free pages on a state it can rebuild. The one-state `ProbeExactlyFreeOneState` reads exactly `k ≥ 3` on a live state it cannot rebuild, and leaves the pool as found. `AdmitPages(k)` alone admits budgeted creates totalling `k`. Every create is admitted only if at least `k` pages are free. A create of budget `(r − 1)·B` reserves exactly `r` pages for `2 ≤ r ≤ cap_pages` (`R(budget)`, executed for every `r` at `(B, cap)` = (16, 4,096), (4, 4,096) and (16, 32,768)). `HolderCreate` refuses before drawing and otherwise draws exactly the reserve. A fresh pool is exactly `block_count · cap_pages` pages, all free. So admitting creates totalling a fresh pool's whole page count `P` leaves **exactly 0** free, shown by admission alone. One limit, executed: at `cap_pages = 2`, `ProbeSplit` splits an odd `k` with a 1-page part (`k = 3` gives `[1, 2]`), which is a budget-0 create refused `SSLM_INVALID_ARGUMENT`. The probe then fails loudly. It does not pass silently. For `cap_pages ≥ 3` every split of `k ∈ [2, 399]` sums to `k` with every part in `[2, cap_pages]` (executed) | `tests/paged-kv/pkv_common.h:587-602` (`ProbeSplit`), `:617-633` (`AdmitPages`), `:660-688` (two-sided), `:695-718` (one-state); `ReservationPages` `sslm_abi.cpp:705-712`; `HolderCreate` `kv_pages.cpp:148-181`; pool size `sslm_abi.cpp:1167-1174`; `sslm_seq_create_budgeted` `:2388-2398` |
| S27 | *(Rev 1.5, E0.)* The `sslm_status` list is generated from one X-macro, and three files mirror it by hand with explicit ordinals: `tests/t2130-g5-red-suite/sslm_g5.h`, `tests/t2138-abi-red-suite/sslm_abi.h` and `tools/t2139_gate_c_type_identity_check.cpp`. `docs/api.md` states `SSLM_KV_BUDGET_EXCEEDED`'s own ordinal (29) in prose, twice; it states no "last ordinal" (*rev 1.7*). Generated checks re-expand the X-macro and compare sentinels against each hand-written mirror: `tools/t2139_gate_c_real_suite_side_check.cpp` (`sslm_g5.h`), `tools/t2141_gate_c_t2138_suite_side_check.cpp` (t2138's header), and the type-identity check's own assertion; `t2139_gate_c_sentinel_negative.cpp` is their must-reject construction. None is edited when a status is appended. The two ordinal cells PKV G35 names pin `SslmGpuStatus`, not `sslm_status` | `sslm_g5.h:240`; t2138 `sslm_abi.h:177`; `t2139_gate_c_type_identity_check.cpp:100`, sentinel assertion `:192`; `t2141_gate_c_t2138_suite_side_check.cpp:77-80`; `docs/api.md:553`, `:787`; `tests/t2956-token-finish-red-suite/cell_status_ordinals.cpp:16-46`; `tests/t2791-gpu-prefill-read-red-suite/cell_status_ordinals.cpp:20-50` |
| S28 | *(Rev 1.5, E0.)* No converter writes the schema section: `tools/convert_model.py` writes no `SCM1`, and `tools/sslm_convert_schema.py` compiles mask pages only. The only live `SCM1` serializer is `_serialize_scm1` in `tools/t2132_build_g5_fixture.py`, imported by five files. A second definition is a frozen, hash-pinned test input (do not edit). One more copy lives outside the repository, in the project's example-asset rebuild script | `t2132_build_g5_fixture.py:109-169` (header pack `:168`); importers `tools/_t2199_s8_synthetic_full_model_fixture.py:33`, `tools/gen_decode_threading_fixture.py:47`, `tests/t2130-g5-red-suite/make_t2922_dual_schema_fixture.py:13`, `tests/t2130-g5-red-suite/t2912_value_level_red.py:369`, `tests/t2899-schema-deadend-red-suite/make_t2909_string_schema_fixture.py:69`; frozen copy `tests/t2130-g5-red-suite/reference/historical/0062c99_t2132_build_g5_fixture.py:101` (pinned by `t2921_frozen_fixture_integrity.py`, loaded by `t2913_common.py:76`); `sslm_convert_schema.py:732`; Wizard `Claude/Brunel/t2804-aex/t2818_rebuild_aex_schemas.py:134` |
| S30 | *(Rev 1.13; the E2 re-anchor on 1.13.0, read at `3b6631f`.)* **Page geometry is the pool's.** (a) B is a rule, not CFG1: the run target `S` is 4 KiB when `D ≤ 64` and cap ≤ 4,096, else 8 KiB, and B is the smallest power of two ≥ 16 with `B·D·elem ≥ S`, clamped to the cap. CFG1's `kv_block_size` no longer selects B. So: real 0.5B (`D` 64, cap 32,768) B = 128; real 1.5B B = 64; the synthetic 0.5B and 1.5B shapes B = 64; pkv_def 128, pkv_odd 256, pkv_32k 256, pkv_qk 64. (b) `cap_pages = ceil(cap/B)`, so a whole-cap holder's last page can be partial (pkv_odd, cap 4,100: 17 pages, the last holding 4 positions). (c) The legacy `sslm_kv_pool_create` and `sslm_kv_page_pool_create` both take the model's rule B; the **contiguous option** `sslm_kv_pool_create_contiguous` has one page per block, B = cap, `cap_pages = 1`. (d) `ReservationPagesIn(pool, budget) = min(ceil(budget/B_pool) + 1, cap_pages_pool)`, so R is per pool (1 in the contiguous option). (e) **Adopt is refused `SSLM_INVALID_ARGUMENT` between pools whose page sizes differ, for either adopter mode**, before the bind permission is revoked or any state moves (PKV A22, cell 4.15; D-SLM8434). A whole_reserve adopter still copies across pools of one page size. (f) A test-only pin, `KvPagesTestPinPagePositions`, replaces the rule's B on a model with no pool yet; it requires B to divide the cap. The E1/E2 cells written at B = 16 and the B = 4 twin are pinned through it | `PageRuleB` `sslm_abi.cpp:706`, `ComputePageGeometry` `:739`, `:764`; `ReservationPagesIn` `:860`; `sslm_kv_pool_create` `:1288`; `sslm_kv_page_pool_create` `:1396`; `sslm_kv_pool_create_contiguous` `:1479`; adopt `:2854`; pin `:5188`; `tests/precache/precache_common.h:26`, `precache.cmake:32`; `tools/gen_paged_kv_fixture.py:86-89`; `e2-reanchor-v1.13.0.md` F2, F4–F6 |

---

## 3. Design — engine, CPU (1.14.0)

Files: `src/sslm_abi.cpp` (verbs, blob writer and reader), `src/kv_pages.*` (PKV §3.3: draw, share, scatter),
`include/superslm/sslm_abi_functions.inc` (declarations, outside the Gate A comparable subset, as PKV §3.6 puts
its verbs), `include/superslm/sslm_abi.h` (one status), new `include/superslm/sslm_version.h`.

### 3.1 What a frozen prefix is, and what the blob must carry

A frozen prefix is, for every purpose an adopter can observe (S3): the K/V rows `[0, L)` where `L =
context_length`; the residual (hidden codes and scale); the five saturation counts; `current_token`; the schema
binding and walk state; and, for a prefix made by `begin_from`, its `origin` (the parent's length, PKV §3.4).
*Rev 1.3:* the blob carries the schema binding as `Fnv1a64(name)`, which is lossy (S25). It names the binding
uniquely only because PC-12 refuses at map any model with two equal non-zero name hashes, or a name hashing to
the "unbound" sentinel 0. Under that invariant, and with the model hash tying the blob to one artifact, the hash
resolves to exactly the schema the prefix was bound to.
`layer_index` is 0 (S2) and `frozen` is true; neither needs storing. The blob carries exactly these, plus identity
(backend; producer, which is the engine version and on GPU the adapter's vendor and device; model hash;
`kv_precision`) and a digest. It carries no page size, page identities,
reuse history, mode or budget, so it is a function of the prefix's valid state alone. *Rev 1.13:* at 1.13.0 one
model has pools of two page sizes, its rule B and the contiguous option's cap (S30), wherever the rule B is below the
cap (*rev 1.14*: where the rule clamps B to the cap, both kinds share one page size and adopt across them is
admitted). Carrying no page size, one blob
restores into either, and every `B` in §3.4–3.6 is the **restore pool's** page size. *Rev 1.5:* `SSB6` does carry
a sequence's `kv_mode` and `budget` (S7), because a sequence's mode and budget decide its later prefill and decode.
A frozen prefix's do not, so leaving them out of `SPX1` loses nothing an adopter can observe: prefill is refused once
a prefix is frozen (`SSLM_PREFIX_FROZEN_REJECTED`, `sslm_abi.cpp:2284`), adopt keys on the **adopter's** mode
(S3), a `begin_from` child's origin is the parent's length (`:2226`), and no verb reports a prefix's mode or budget.

### 3.2 The `SPX1` blob

One format for both backends (PC-5), little-endian, **184-byte fixed header** (*rev 1.1*: 176 in rev 1; the producer
adapter field is new, PC-5 and PC-7):

| Offset | Bytes | Field |
|---|---|---|
| 0 | 4 | magic `'S','P','X','1'` |
| 4 | 4 | `backend` (u32: 0 CPU, 1 GPU) |
| 8 | 12 | engine version, `major`, `minor`, `patch` (u32 each; PC-7) |
| 20 | 8 | producer adapter `vendor_id`, `device_id` (u32 each; 0, 0 on CPU; PC-5, PC-7) |
| 28 | 32 | `digest`: SHA-256 of the whole blob with this field zeroed (PC-8) |
| 60 | 32 | model hash (`RawIntegrityHash()`, S5) |
| 92 | 4 | `kv_precision` |
| 96 | 8 | `schema_name_hash` (0 = none, S6) |
| 104 | 4 | `dfa_walk_state` |
| 108 | 4 | `current_token` (−1 iff `L = 0`) |
| 112 | 8 | `context_length` `L` (i64) |
| 120 | 8 | `origin` (i64; 0 for a root prefix) |
| 128 | 16 | hidden scale `m`, `e` (i64 each) |
| 144 | 40 | aggregate saturation count, then `kv_landing`, `k_channel_landing`, `rope_q`, `rope_k` (u64 each) |

Then the residual (`hidden_size` bytes, when `hidden_size > 0`, as `SSB5` writes it), then the K/V rows of
positions `[0, L)` in **canonical layout**: today's layout with `context_cap` replaced by `L` (PKV §3.1's one-page
layout at `B = L`, the layout `SSB6` uses). **Size** `= 184 + hidden_size + L · bytes_per_token`, exact; a blob of
any other size is refused. *Executed* (python over the geometry constants below; §8):

| `L` | 0.5B blob | 0.5B resident pages, real artifact (cap 32,768, rule `B = 128`) | 0.5B resident pages at cap 4,096 (rule `B = 64`) | 1.5B blob |
|---|---|---|---|---|
| 0 | 1,080 B | 0 | 0 | 1,720 B |
| 128 | 0.751 MiB | 1 (0.750 MiB) | 2 (0.750 MiB) | 1.752 MiB |
| 1,000 | 5.860 MiB | 8 (6.000 MiB) | 16 (6.000 MiB) | 13.674 MiB |
| 2,000 | 11.720 MiB | 16 (12.000 MiB) | 32 (12.000 MiB) | 27.345 MiB |
| 4,096 | 24.001 MiB | 32 (24.000 MiB) | 64 (24.000 MiB) | 56.002 MiB |
| 32,768 | 192.001 MiB | 256 (192.000 MiB) | — (above the cap) | 448.002 MiB |

(`bytes_per_token` 6,144 at 0.5B and 14,336 at 1.5B, PKV §3.1; `hidden_size` 896 and 1,536. Rev 1.1's 8 extra
header bytes move no MiB figure at three decimals. *Rev 1.13 (re-anchor F2):* the resident-page columns are at
1.13.0's rule B (S30): a 0.5B page is 768 KiB at `B = 128` and 384 KiB at `B = 64`; a 1.5B page at its rule `B = 64`
is 896 KiB. Rev 1.12's column was at `B = 16` (8 / 63 / 125 / 256 / 2,048 pages), which only the test-only pin now
produces. In the contiguous option every non-empty prefix is one page of the whole cap, 192.000 MiB at the real
0.5B artifact. The blob columns do not depend on B.)

A nested prefix's blob is self-contained: it carries all rows `[0, L)`, parent span included (PC-2). Cost at the
PKV cohort shape (world 1,000 + persona 200): 7.03 MiB per persona blob against 1.17 MiB for its own rows only.

*Rev 1.5:* E0 checked this field order against `SSB6`'s built header conventions at `5d5ca58` and kept it unchanged
(evidence: `e0-reanchor-pkv-5d5ca58.md`, *`SPX1` field order*). The fields are not conditional.

### 3.3 `sslm_prefix_save`

`sslm_status sslm_prefix_save(sslm_prefix prefix, void* buf, size_t* n)`

- `*n` two-call convention of `sslm_seq_save` (S7): too small or null `buf` → `SSLM_BUFFER_TOO_SMALL` with `*n` set
  to the exact size; success writes exactly `*n` bytes.
- Refusals, nothing written: null `n` or `prefix` → `SSLM_INVALID_ARGUMENT`; an **unfrozen** prefix →
  `SSLM_INVALID_ARGUMENT`, as adopt refuses one (PC-9); a prefix whose `layer_index != 0` → `SSLM_INVALID_ARGUMENT`
  (unreachable by S2; a real check, guard-vitality cell 11.2).
- Gathers rows `[0, L)` from the prefix's pages (shared and private alike) into canonical layout, writes the
  header (producer adapter 0, 0 on CPU), computes the digest last.
- Takes no lock: a frozen prefix is immutable (PKV §3.8). Releasing it during a save is caller UB, as for adopt.

### 3.4 `sslm_prefix_restore`: validation, draw, scatter

`sslm_status sslm_prefix_restore(sslm_model model, sslm_kv_pool* pool, const void* buf, size_t n,
sslm_prefix parent_or_null, sslm_prefix* out, uint32_t* out_shared_pages)`

**Validation, in this order, all before any page is drawn or any handle allocated** (the order is the status
precedence; each refusal changes nothing and leaves `*out` null):
1. null `out` → `SSLM_INVALID_ARGUMENT`, nothing written. From here on, every refusal sets `*out = nullptr`
   first. Null `model`, `pool`, `*pool` or `buf` → `SSLM_INVALID_ARGUMENT`; `(*pool)->model != model` →
   `SSLM_INVALID_ARGUMENT` (the check every draw site makes, e.g. `sslm_abi.cpp:2162`, `:2181`, `:3972`).
2. `n < 184`, or magic not `SPX1` → `SSLM_INVALID_ARGUMENT` (so an `SSB*` or `SLM*` blob is refused here, and an
   `SPX1` blob given to `sslm_seq_restore` is refused by that verb's magic check).
3. digest mismatch → `SSLM_INVALID_ARGUMENT`. The digest is checked before any other field is trusted, so every
   single-byte corruption past the magic lands here.
4. `backend != 0` → `SSLM_INVALID_ARGUMENT` (PC-5).
5. producer ≠ this library's (engine version, and adapter vendor and device, which are 0, 0 on CPU) →
   `SSLM_RESTORE_PRODUCER_MISMATCH`, new (PC-7). Major, minor and patch each count.
6. model hash → `SSLM_RESTORE_MODEL_MISMATCH`; then `kv_precision` → `SSLM_RESTORE_KV_MISMATCH` (S5).
7. schema → `SSLM_RESTORE_SCHEMA_MISMATCH`, with exactly S6's three rules. The hash resolves to one schema only
   because PC-12 holds for every mapped model (*rev 1.3*).
8. domains → `SSLM_INVALID_ARGUMENT`, in this order (*rev 1.1*, tightened):
   - `0 ≤ L ≤ context_cap`, checked **before** any arithmetic on `L`;
   - `n == 184 + hidden_size + L · bytes_per_token`, computed with checked arithmetic (as the `SSB*` reader does,
     `sslm_abi.cpp:3918-3925`, and `SSB6`'s exact-size rule, `:3938-3944`). Once `L ≤ context_cap` holds, no `L` can wrap the formula, so the checked arithmetic
     is **defence in depth**: no cell can make it fire, and 11.1 exempts it (*rev 1.2*);
   - `0 ≤ origin ≤ L`;
   - `L == 0` ⇒ `current_token == −1`; `L > 0` ⇒ `0 ≤ current_token < vocab`;
   - the hidden scale's `m` fits int32, the bound the forward funnel enforces on every scale it takes in or
     produces (`CarriedScaleMantissaOutOfDomain`, S20). *Rev 1.5 (E0):* an `m` outside int32 is out of the funnel's
     domain and is never produced by save, because every scale the forward stores has passed that check (S20; that
     check is pinned by the engine's funnel cells, S20, *rev 1.6*). So refusing such a blob refuses no blob this
     library wrote. The bound is kept as a domain rule, not as a decode
     guard: the first adopter norm ignores the copied scale, and the next embed replaces it (S20), so in an edited
     blob a chosen `m` or `e` changes no adopter token.

**What is promised, exactly** (*rev 1.1*; this is what §13 hands the adversary).
- **(i) Fidelity.** For every blob this library's own `sslm_prefix_save` wrote, restored by a build of the same
  release version (or, between releases, by the same build), a restore that returns `SSLM_OK` yields a prefix whose
  adopters, in pools of the restore pool's page size, emit exactly the tokens adopters of the prefill-built prefix
  emit in a pool of that page size. *Rev 1.18 (PC-15 Q2):* the qualifier moves from the third scope statement into the
  sentence; the promise is unchanged in substance. *Rev 1.3, two scope statements:*
  - It holds because a model maps only under PC-12. Rev 1.1's unscoped form was struck: a model with two colliding
    schema names restored an untouched blob bound to the wrong schema (S25).
  - Between releases, two development builds on one version line share a producer (PC-7). A blob from the other
    build can be accepted, and it is outside (i). Released builds of one version are covered, including across
    compilers (cell 4.3 pins byte-identical blobs across MSVC, gcc and clang). The plugin's tiers separate
    development builds anyway, because its key and wrapper carry the Layer-1 commit (§5.2, §5.7).
  - *Rev 1.13, a third scope statement (re-anchor F5):* "adopters" are adopters in pools of the **restore pool's
    page size**, and the prefill-built prefix they are compared with is one built in a pool of that page size. At
    1.13.0 a model has pools of two page sizes wherever its rule B is below the cap (S30), a blob restores into either, and adopt is refused
    `SSLM_INVALID_ARGUMENT` between pools whose page sizes differ, for a prefill-built prefix exactly as for a
    restored one (X13). So an adopter in a pool of another page size is refused by name, never given other tokens.
    *Rev 1.18:* the sentence of (i) above, and §13 item 2's quoted promise, now carry this qualifier themselves (Q2);
    save then restore is the way to move a prefix to another pool kind.
- **(ii) Integrity.** Any accidental change to such a blob (truncation, extension, or any changed byte) is refused,
  barring a SHA-256 collision. Step 2 (size, magic) or step 3 (digest) refuses it before any other field is read.
- **Not promised: authenticity.** The digest is not a security boundary. A blob edited on purpose, with its digest
  recomputed, passes step 3. *Rev 1.2, stated as a partition of §3.2's table so it is complete:* in such a blob
  **every field not fixed by steps 2–6 is attacker-chosen within the limits steps 7–8 enforce**, as `SSB5`
  accepts its own (S20).
  - *Fixed exactly:* magic, `backend`, producer, model hash, `kv_precision`; the size follows from `L`; the digest
    is derived.
  - *Chosen within limits:*
    - the schema binding: any schema the model resolves, or none with the sentinel walk;
    - the walk state: anywhere below that schema's `state_count`;
    - `L`: anywhere in `[0, context_cap]`, with a row count to match;
    - `origin`: anywhere in `[0, L]`;
    - `current_token`: anywhere in `[0, vocab)` when `L > 0`;
    - `m`: anywhere within int32;
    - `e`: any i64, with no stated bound (the funnel saturates `e` when combining scales, `checked_chain_funnel.cpp:197`);
    - the five saturation counts: any u64;
    - the residual codes and the K/V rows: any bytes.

  An edited blob can therefore be accepted and give other tokens.
- **(iii) Memory safety, a separate claim.** A hostile blob is either refused with a named status, or accepted and
  then decodes to tokens or to a named forward status, never undefined behaviour. Cell 2.3b, under ASan and UBSan.
  This claim is not part of the strike promise.

The existing `SSB5` restore also leaves `m` and `e` unchecked (S20). This plan does not change `SSB5`.

**Draw and build** (*rev 1.5*: draw and share are two critical sections, in PKV's built order, S26). Evaluate the
parent hint (§3.5), which yields `s` shared pages (`s = 0` without a match). Then, in this order:
1. **Draw** `ceil(L/B) − s` pages pool → a new holder's reserve, under the pool mutex (`CreateHolder`, PKV §3.3). If
   they are unavailable, `SSLM_KV_POOL_EXHAUSTED` with nothing drawn and no refcount changed. Because the draw is
   first, this refusal still draws nothing.
2. **Share** the `s` parent pages into the holder's leading entries (`HolderShareLeading`, refcount +1 each), under a
   **second** pool-mutex section, when `s > 0`. It must precede any private mapping (`HolderShareLeading` requires
   the holder to map nothing yet). The share checks every source page before changing any count; if it refuses,
   release the holder (returning every drawn page) and return the mapped status: `SSLM_INVALID_ARGUMENT` for a
   parent page at the maximum refcount, `SSLM_ARTIFACT_REJECTED` for one at refcount 0 (reachable only through
   caller UB, §3.7). This is the shape PKV's own `sslm_seq_restore_shared` and `begin_from` have
   (`sslm_abi.cpp:3997-4008`, `:2210-2219`). The refusal needs about 2³² references to one page, and no ABI seam
   reaches it. *Rev 1.6 (D-SLM7984):* it is pinned all the same, because the release it guards is new restore code
   and leaving it out would leak `ceil(L/B) − s` pages on every refused share. The restore-site fault slot's
   **share** position (§8) makes this step return either refusal before any count changes, and cell 5.11 grades the
   release and the status. Rev 1.5 exempted it under 11.1, and that exemption is withdrawn.
3. **Map** the private pages with `MapPagesFor(L)` (`HolderMapNext` underneath, not `HolderMapMaterialized`). This
   maps exactly the `ceil(L/B) − s` drawn pages and empties the reserve. *Rev 1.6:* `MapPagesFor` has two refusals
   of its own, checked before it maps anything (S26). On either, release the holder, returning every drawn page and
   every shared refcount, and return the status it gave. The short-reserve branch (`SSLM_KV_POOL_EXHAUSTED`) is
   unreachable, because the reserve is drawn at exactly `ceil(L/B) − s`. It is pinned through the fault slot's
   **map** position, which empties the reserve before this step, as PKV pins its own (cell 5.11). The
   past-the-table branch (`SSLM_INVALID_ARGUMENT`) is unreachable: step 8 bounds `L ≤ context_cap`, the table has
   `ceil(context_cap/B)` entries, and `ceil(L/B) ≤ ceil(context_cap/B)` for every `L ≤ context_cap` (S26, S30;
   *rev 1.13*, re-anchor F4: rev 1.12 said "`B` divides the cap, and the table has `context_cap / B` entries", which
   1.13.0's partial last page makes false; the conclusion stands). No seam reaches it, so 11.1 exempts it, as it
   exempts the checked size arithmetic. Cell 4.7 restores at `L = cap` on a cap `B` does not divide, so the exemption's
   premise is executed, not only argued.
4. **Allocate** the handle and its residual storage; on `std::bad_alloc`, release the holder, returning every drawn
   page and every shared refcount, and return `SSLM_ALLOCATION_FAILED` (the shape `sslm_prefix_begin` has through
   `FinishPrefix`, `sslm_abi.cpp:2088-2125`, `:2155-2170`). Its test seam is the fault slot's **allocation**
   position, which fires after the draw, the share and the map (§8, *rev 1.2*; positions named in *rev 1.6*).

So **three refusals can follow a draw** (*rev 1.6*; rev 1.5 named two): the share refusal of step 2, the map
refusal of step 3, and the allocation failure of step 4. Each returns every drawn page and every shared refcount
before the verb returns (§3.6's *refused after a draw*), and each is pinned through the fault slot (5.2, 5.11).
Every validation refusal precedes the draw, and cell 5.6 tells them apart.
Scatter rows `[s·B, L)` from the blob into the private pages; set the state fields from the header, `layer_index =
0`, `frozen = true`, the schema from step 7, `origin` from the header; `*out_shared_pages = s` when non-null.

**What the restored handle is.** A frozen prefix whose table maps exactly `ceil(L/B)` pages: the leading `s` shared
(by PKV §3.3's provenance definition), the rest private and dirty. It holds no reserve. That is the page state a
frozen prefix built by prefill holds after its first freeze (PKV §3.4 "prefix freeze"), so pool accounting cannot
tell the two apart. Its mode is recorded as `budget`; a prefix's mode never decides an adopt (PKV §3.5) and no verb
reports a prefix's mode, so the value is unobservable. **Peak draw** (*rev 1.4*: narrowed after the re-strike):
restore draws `ceil(L/B) − s` pages. *Rev 1.13 (re-anchor F6):* every comparison below is in the restore pool:
`B` is that pool's page size and `R` is that pool's `ReservationPagesIn(pool, budget)` (S30), which is what the
build it replaces would reserve there. In the contiguous option `B` is the cap, so `ceil(L/B) = R(budget) = 1` for
every `L ≥ 1` and the comparisons hold trivially.
- **Root prefix:** building it by prefill draws at least `ceil(L/B)` before its freeze (legacy begin `ceil(cap/B)`;
  budgeted begin `R(budget) ≥ ceil(L/B)`). So a root restore is never refused where the build it replaces would
  have been admitted, and legacy admission counts (PKV §3.5) are unchanged.
- **Nested prefix, matching parent:** `begin_from` shares the parent's `floor(origin/B)` full pages and draws
  `R(budget)`; a restore whose hint meets (a)–(f) shares the same `s = floor(origin/B)` and draws
  `ceil(L/B) − s ≤ R(budget)`. Build-equivalent admission holds.
- **Nested prefix, private restore** (null hint, or a hint failing any of (b)–(f); `s = 0`): the restore needs
  **`ceil(L/B)` free pages**, which can be far more than the `begin_from` build it replaces. At the strike's cell
  (parent 1,000 resident, child to 1,200, `B = 16`) the build draws `R(200) = 14` pages and is admitted with 14
  free, while a private restore needs 75 and is refused `SSLM_KV_POOL_EXHAUSTED`, nothing drawn. *Rev 1.13 (re-anchor
  F2):* `B = 16` is now the test-only pin's view, which cell 5.10 keeps. At the rule B the same cell reads **3 against
  10** at the real 0.5B artifact (`B = 128`) and **5 against 19** at the real 1.5B artifact and the synthetic shapes
  (`B = 64`) (executed arithmetic, §8; *rev 1.14*: a python transcription of `PageRuleB` and `ReservationPagesIn`, not an engine run; the `B = 16` figures were executed on the engine). **Build-equivalent admission is promised only with a matching parent.** A caller that restores a nested blob without one sizes for
  `ceil(L/B)`, or accepts the named refusal.

Cells 7.4 and 5.10 pin these. (Rev 1.3 said "`begin_from` shares the same `s`", which holds only for a matching
hint: the build shares the real parent whatever the later restore is given.)

**What adopters see.** Adopt reads only the fields S3 lists and the rows. All are byte-equal to the saved prefix's,
so an adopter of either mode, `begin_from` from the restored prefix, and a sequence restored with
`sslm_seq_restore_shared` against it all behave as they would against the original. *Rev 1.13 (re-anchor F5):*
that holds for adopters in pools of the **restore pool's page size**, against an original in a pool of that page
size. 1.13.0's adopt also reads the two pools' page sizes, and refuses `SSLM_INVALID_ARGUMENT` when they differ, for
a whole_reserve (copy) adopter as for a budget one (S30 (e)). A restored prefix is bound by that rule exactly as a
prefill-built one is, so an adopter in a pool of another page size gets the refusal the original would give it in
the same pairing of pools (X13, cell 8.13). Moving a prefix between pool kinds is what save and restore are for:
1.13.0's own contract names them as the way (`docs/api.md:733-736` at `a5f3ad1`, the `v1.13.0` tag: "…`SSLM_INVALID_ARGUMENT`
wherever their page sizes differ, so move state between pool kinds by save and restore"; *rev 1.14*, correcting rev
1.13's citation of `sslm_abi_functions.inc:172-174` and `docs/api.md:1036-1039` at `3b6631f`, which are this plan's
own header text, written at E2 (§3.8)), and cell 9.8 restores one blob into both kinds. `begin_from` draws from the parent's own pool, and a
restore hint shares only within one pool (§3.5 (d)), so neither crosses page sizes.

### 3.5 Nested prefixes: re-sharing a live parent

`parent_or_null` is a **sharing hint, never a requirement for correctness** (the PKV §3.7 rule for
`restore_shared`): without it, or with one that fails, the restore still rebuilds equal bytes when `ceil(L/B)` pages
are free. It can be required for **admission** at a pool size that admits the `begin_from` build (§3.4 *Peak draw*;
*rev 1.4*). The restore
shares the parent's full pages `[0, floor(origin/B))`, `s = floor(origin/B)`, **only if** all hold:
- (a) `origin > 0`;
- (b) the parent is frozen;
- (c) the parent's model is `model` (step 6 has tied the blob to `model`);
- (d) the parent's pool is `*pool`;
- (e) the parent's `context_length == origin`;
- (f) the parent's rows at positions `[0, s·B)`, every layer, K and V, **compare byte-equal** to the blob's rows for
  those positions.

Otherwise `s = 0`, the restore is private, and `*out_shared_pages = 0` with `SSLM_OK`. The rows
`[s·B, L)`, which include the parent's partial tail page, always come from the blob, so the restored child is
byte-equal to the saved child whatever the parent's tail holds. Nothing new is stored for this: the blob's rows are
the fact compared, at a cost of `O(origin)` bytes, the order of the scatter it saves. A stored hash or stored parent
tokens are rejected for PKV §3.7's reasons (identity assumed, not checked).

*Executed* at the PKV cohort shape (origin 1,000, `L` 1,200, `B` 16): `s` = 62 shared pages, 13 private pages,
rows scattered from position 992. *Rev 1.13 (re-anchor F2):* `B = 16` is now the test-only pin's view, which the
E2/E3 cells built on these figures keep (5.1, 5.2, 5.10, 5.11, 7.4). At the rule B (S30): the real 0.5B artifact
(`B = 128`) gives `s` = 7, 3 private, scatter from 896; the real 1.5B artifact and the synthetic shapes (`B = 64`)
give `s` = 15, 4 private, scatter from 960 (executed arithmetic, §8; *rev 1.14*). Cell 4.7's nested leg runs at the rule B.
*Rev 1.14 (re-gate audit B1):* in the contiguous option `B` is the cap, so a nested blob restored there with a
matching parent in that contiguous pool has `s = floor(origin/cap)`: 0 for every `origin < cap` (a private restore of
one page), and 1 only at `origin = L = cap` (the whole page shared, nothing drawn). A parent in a page pool fails
(d). Every operand of the hint (`s`, (f)'s span `[0, s·B)`, the scatter start `s·B`) uses the restore pool's `B`,
never the model's rule B; cell 9.8's nested leg [E3] pins this.

A root prefix (origin 0) ignores the hint (condition a). A parent whose own table shares pages from a grandparent
is a legal parent: sharing is by page, and PKV §3.3's refcount law already covers a page held by three tables.

### 3.6 The restore matrix

Every verb that rebuilds a prefix from bytes, crossed with every blob kind and hint it can receive. Dispositions
as PKV §3.9, plus one PKV's table does not need (*rev 1.6*):
- **keeps:** the handle is frozen, its `L`, `origin`, state fields and rows equal the saved prefix's, and its pages
  are `ceil(L/B)` with `s` shared, `B` being the restore pool's page size (*rev 1.13*);
- **refused:** named status, nothing drawn, no refcount changed, `*out` null;
- **refused after a draw:** named status, `*out` null, and every page drawn and every shared refcount taken has
  been returned before the verb returns. The pool's free count and every refcount are then as before the call. This
  is the share, map and allocation refusals of §3.4 steps 2–4. Rev 1.5 said "no third disposition" while its own
  step 2 drew and then returned. X12 names the inputs that reach a post-draw refusal, X3's matching hint with a
  saturated parent page among them, so X1–X4's "keeps" covers every other input they name. In the field X12 needs
  an allocation failure, about 2³² references to one page, or caller UB. In a test it is reached through the fault
  slot (§8).

No fourth disposition.

| # | Verb | Given | Disposition | Cells |
|---|---|---|---|---|
| X1 | `sslm_prefix_restore` | `SPX1` root blob, null hint | keeps; `s = 0`. *Rev 1.3:* rows X1–X4 "keep" the schema binding because PC-12 holds for every mapped model, including a blob bound to a non-first schema of a many-schema model (6.7). *Rev 1.13:* at the rule B, a cap `B` does not divide included (4.7) | 9.1, 10.1, 6.7, 4.7 |
| X2 | `sslm_prefix_restore` | `SPX1` root blob, any hint | keeps; hint ignored, `s = 0` | 9.2 |
| X3 | `sslm_prefix_restore` | `SPX1` nested blob, hint meeting (a)–(f) | keeps; `s = floor(origin/B)` | 9.3, 10.2 |
| X4 | `sslm_prefix_restore` | `SPX1` nested blob, null hint, or a hint failing any one of (b)–(f) (each its own case, including a parent of another pool or model) | keeps; private, `s = 0` | 9.3, 2.6 |
| X5 | `sslm_prefix_restore` | `SPX1` from another producer (engine major, minor or patch; adapter vendor or device) | refused `SSLM_RESTORE_PRODUCER_MISMATCH` | 2.3 |
| X6 | `sslm_prefix_restore` | `SPX1` from another model / `kv_precision` / unresolvable schema | refused `SSLM_RESTORE_MODEL_MISMATCH` / `_KV_MISMATCH` / `_SCHEMA_MISMATCH` | 2.3 |
| X7 | `sslm_prefix_restore` | `SPX1` with `backend = 1`; any `SSB*`/`SLM*` blob; bad digest; bad size or domain | refused `SSLM_INVALID_ARGUMENT` | 2.1, 2.2, 2.3 |
| X8 | `sslm_prefix_restore` | a valid blob, pool without `ceil(L/B) − s` free pages | refused `SSLM_KV_POOL_EXHAUSTED` | 5.1 |
| X9 | `sslm_seq_restore`, `sslm_seq_restore_shared` | an `SPX1` blob | refused `SSLM_INVALID_ARGUMENT` (magic) | 2.2 |
| X11 | *(Rev 1.3.)* `sslm_prefix_restore` and the GPU twin | any `SPX1`, against a model with two equal non-zero schema-name hashes or a name hashing to 0 | unreachable: that model is refused at map (PC-12), so no model, pool or blob for it exists | 2.13 |
| X10 | `sslm_prefix_save` | frozen prefix (root, nested, empty, restored) / unfrozen | writes `SPX1` / refused `SSLM_INVALID_ARGUMENT` | 9.1, 2.4 |
| X12 | *(Rev 1.6.)* `sslm_prefix_restore` | a blob that passes steps 1–8, then a refused share (a parent page at the maximum refcount, or at 0 through caller UB), a short reserve at the map, or `std::bad_alloc` at allocation | refused after a draw: `SSLM_INVALID_ARGUMENT` / `SSLM_ARTIFACT_REJECTED`, `SSLM_KV_POOL_EXHAUSTED`, `SSLM_ALLOCATION_FAILED`; every drawn page and shared refcount returned | 5.2, 5.11 |
| X13 | *(Rev 1.13; re-anchor F5.)* `sslm_prefix_restore` | `SPX1` (root or nested) into a pool whose page size differs from that of the pool the saving prefix lived in (a rule-B page pool ↔ the contiguous option) | keeps, with `ceil(L/B)` pages at the restore pool's B; re-saved, the blob is byte-identical. An adopter of the restored prefix in a pool of another page size, of either mode, is refused `SSLM_INVALID_ARGUMENT`, as an adopter of the prefill-built prefix is in the same pairing of pools (S30 (e)); nothing drawn, adopter unchanged. *Rev 1.14:* a nested blob's hint is evaluated at the restore pool's `B` (§3.5) | 8.13, 9.8 (root [E2]; nested [E3], *rev 1.14*), 7.12 |

### 3.7 Concurrency

- A frozen prefix is immutable: any number of threads may save it, adopt it, `begin_from` it, and pass it as a
  restore hint at once. *Rev 1.5:* restore reads the hint's rows with no lock, which is correct because a frozen
  prefix's table and rows do not change (PKV's built comparison, `SharingHandleMatches`, `sslm_abi.cpp:3685-3707`,
  does the same). Only refcounts change under the pool mutex.
- Concurrent restores into one pool contend only on the pool mutex (PKV §3.3). A restore that succeeds takes it
  once when `s = 0` (the draw section) and twice when `s > 0` (the draw section, then the share section; §3.4,
  *rev 1.5*; "once when `s = 0`" *rev 1.6*). A refusal after a draw also takes it to return the pages.
- Releasing a prefix while it is being saved, adopted, or read as a hint is caller UB (PKV §3.8), stated in the
  header. Releasing a parent after a child restored against it is legal: the child holds refcounts on the shared
  pages, so they stay until its release.

### 3.8 ABI surface, statuses and mirrors

- **Verbs:** `sslm_prefix_save`, `sslm_prefix_restore` in `sslm_abi_functions.inc`, regenerated into the api-slot
  census `c_abi_verbs.txt` (PKV §3.6).
- **Status:** `SSLM_RESTORE_PRODUCER_MISMATCH` (*rev 1.1*: named `_ENGINE_MISMATCH` in rev 1), appended after PKV's `SSLM_KV_BUDGET_EXCEEDED` as ordinal **30**
  (S8). It is added to the `SSLM_STATUS_ENUM_LIST` X-macro and, in the same commit, to every hand-written mirror of
  the list (*rev 1.5*, S27): `tests/t2130-g5-red-suite/sslm_g5.h`, `tests/t2138-abi-red-suite/sslm_abi.h` and
  `tools/t2139_gate_c_type_identity_check.cpp`, and to `docs/api.md`, which states the new status with its
  ordinal, 30, as it states `SSLM_KV_BUDGET_EXCEEDED`'s (*rev 1.7*). Each mirror takes
  the form its own file uses (*rev 1.6*): an explicit `= 30` in `sslm_g5.h` and in the type-identity check; in list
  order, with no `=`, in t2138's header, whose enumerators are positional. Each hand-written mirror has a
  compile-time check that re-expands the X-macro and compares `SSLM_STATUS_NEXT_FREE` sentinels:
  `t2139_gate_c_real_suite_side_check.cpp` (`sslm_g5.h`), `t2141_gate_c_t2138_suite_side_check.cpp` (t2138's
  header), and the type-identity check's own sentinel assertion. None of them needs an edit. Those targets are
  `EXCLUDE_FROM_ALL`, so a plain `cmake --build` does not reach them (*rev 1.6*; rev 1.5 said "fails the build"). A
  missed mirror fails ctest's Gate C entries: `t2139_gate_c_must_accept_compiles`,
  `t2139_gate_c_real_suite_side_check_compiles` and `t2141_gate_c_t2138_suite_side_must_accept_compiles`, each with
  its commissioned must-reject (`CMakeLists.txt:794-858`). They are cell 7.16 and are named in E2's exit.
  `docs/api.md`'s ordinals are checked per status name by a source check (cell 7.17; *rev 1.7*, D-SLM7991,
  superseding D-SLM7989's "last ordinal" form). E1's
  red cells need the status before E2 adds it. They compare against a test-side constant
  `PC_RESTORE_PRODUCER_MISMATCH ((sslm_status)30)`, defined as PKV defines its own (`tests/paged-kv/pkv_abi.h:37-39`).
  *Rev 1.7:* the `static_assert` that it equals the enumerator sits under `#ifdef SSLM_HAS_PREFIX_SAVE_ABI`, a
  macro E2 defines in `sslm_abi.h` beside the new status and the two verbs, as PKV's assert sits under
  `SSLM_HAS_PAGED_KV_ABI` (`tests/paged-kv/pkv_common.h:41-42`; the macro is defined at `sslm_abi.h:282`). Before
  E2 the assert is compiled out; from E2 it is compiled in. The two ordinal cells PKV G35 names pin
  `SslmGpuStatus`, so they change at F2 with the GPU status (§4 item 3), not here. The step re-runs the grep for the
  current last enumerator so a mirror added since is not missed.
- **Model map (PC-12, *rev 1.3*):** the schema-section parser adds two refusals beside `DuplicateSchemaName`
  (`schema_masks.h:304-306`): `DuplicateSchemaNameHash` (two names with equal `Fnv1a64`) and `ZeroSchemaNameHash`
  (a name whose `Fnv1a64` is 0). Both surface through the parser's existing artifact-rejection path, so there is no
  new status ordinal and the ABI change is additive. The GPU model maps schemas through the same parser
  (`gpu_1p0.cpp:1247`, S25; *rev 1.5*), so both backends refuse at map. `docs/api.md` and the 1.14.0 release note
  state the new refusal.
- **The converter refusal (PC-12, *rev 1.5*: located).** No converter writes the schema section; the repository's one
  live `SCM1` serializer is `_serialize_scm1` in `tools/t2132_build_g5_fixture.py` (S28). It refuses, naming both
  schemas, a schema list with two equal `Fnv1a64` name hashes or a name hashing to 0. That one change covers all
  five in-repository callers. Two copies are not changed, and the map-time refusal above catches what either
  emits:
  - the frozen historical definition under `tests/t2130-g5-red-suite/reference/historical/`, a hash-pinned test
    input that must stay byte-identical;
  - the copy outside this repository, in the project's example-asset rebuild script, which 1.14.0 does not ship.
    E0's certified-artifact audit at the box (§8) covers the example assets it produced.

  *Rev 1.6 (D-SLM7985).* `_serialize_scm1` has **no opt-out**: no argument, environment variable or flag makes it
  emit a colliding or zero-hash list. PC-12's own must-reject artifacts are therefore never built through it: 2.13's
  two-name and zero-hash models, which 2.13, 2.15, 11.6 and the certified-artifact audit use. An **independent
  test-only `SCM1` writer** builds them (§8; the zero-hash name is recorded in D-SLM8002), authored by the
  debunker inside E0 (*rev 1.7*, D-SLM7990). It packs design §13.2's format with python `struct` (a 24-byte header,
  56-byte descriptors, the name blob, then the four per-schema blocks) and never imports `_serialize_scm1`. It is
  commissioned by byte equality with `_serialize_scm1` on non-colliding lists (§8). So the converter keeps doing
  exactly what PC-12 says, the must-reject bytes stay regenerable at test time, and they are produced by the same
  format the real path emits.
- **Engine version:** new `include/superslm/sslm_version.h` with `SUPERSLM_VERSION_MAJOR/MINOR/PATCH`, and a CI
  source check `tests/ci/check_version_header.py` that fails unless they equal `CMakeLists.txt`'s `project(…
  VERSION …)`. That makes the constant correct in every build system, the plugin's included (S12), and the release
  step's version bump fails the check until both move. *Rev 1.18 (PC-14, PC-7):* the branch's header reads 1.13.0
  and `main`'s CMake reads 1.13.1 (checked 2026-10-20). At the squash (M1) both read **1.13.1**, the version line the
  squash lands on, and E5's release commit moves both to 1.14.0. *Rev 1.18.1 (plan review F11):* this is not a
  choice: approved PC-7 already says that between releases CMake carries the last released version, so only this
  reading survives (rev 1.18 listed it as PC-14 (b)). Setting 1.14.0 at the squash would give development builds the
  release's producer, which PC-7 cannot tell apart from the release. M1's receipts read the two values, not only
  their equality (§9 M1; audit G2).
- **Sequence save formats are 1.13.0's** (*rev 1.13*, PC-13). 1.14.0 changes no byte that `sslm_seq_save` writes or
  `SeqRestoreImpl` accepts (`SSB5`, `SSB6`), and no GPU source (`SLM5`, `SLM6`). At `3b6631f` that holds: the diff
  from `v1.13.0` (`a5f3ad1`) to the branch tip touches `src/` and `include/` in five files, all additive (no line
  removed in `src/sslm_abi.cpp`; the two verbs, their helpers, the test seams, the status, the macro, the PC-12
  parser checks and `sslm_version.h`), and no GPU file. Sequence blobs carry no engine version (S7, PC-7's
  alternative), so the engine already accepts a 1.13.0 sequence save; the boundary players meet is the plugin
  wrapper's Layer-1 tag (§5.11). The executable proof is cells 9.9 and 9.10 at E5, not this diff reading. PC-12's
  parser refusal is the one way a 1.13.0 save can stop restoring: its model no longer maps. *Rev 1.16 (strike L2):*
  that refusal lives in `schema_masks.h`, which the GPU model map compiles too (`src/gpu/gpu_1p0.cpp:70` includes
  it), so "no GPU source" changes no GPU file but does change which models the GPU maps, by the same single
  exception. E0's audit found no
  certified artifact or example asset it refuses (§8).
  *Rev 1.18 (PC-13 retargeted, PC-14).* The predecessor 1.14.0 is compared with is now **1.13.1** (tag object
  `034a76a`, peeled and release commit both `d697927`), the version the paging release (plugin 1.2.0) vendors.
  `v1.13.0` → `v1.13.1` changes 21 files, of which four are library sources under `include/` and `src/`, all GPU
  (`gpu_1p0.h`, `gpu_1p0.cpp`, `d3d12_harness.h`, `gpu_kv_pool_geometry.h`: the pool cap and
  `sslm_gpu_kv_pool_max_page_count`; *rev 1.18.1*, plan review F13: rev 1.18 said "four files"), and no sequence
  writer or reader, so
  1.13.1 keeps 1.13.0's formats byte for byte by diff reading (executed by 9.9 and 9.10 at E5). This section's "no GPU
  source" **no longer holds** for 1.14.0: the squash carries §4's GPU code (§4 preamble), so 1.14.0 changes GPU
  source. The claim that matters, that no byte `SLM5` or `SLM6` writes or accepts changes, is what 9.10 executes; the
  diff reading at E5 names every GPU file the squash touches.
- **Header text:** the two verbs' contracts as §3.3–3.5 state them, including that the hint is never required for
  correctness but a private nested restore needs `ceil(L/B)` free pages (*rev 1.4*), that `B` is the restore pool's
  page size and the restored prefix is adoptable only from pools of that page size (*rev 1.13*; written at
  `3b6631f`, `sslm_abi_functions.inc:172-174`),
  the UB sentence of §3.7, that a blob restores only on the producer and backend that wrote it, and §3.4's
  *What is promised* with its partition of fixed and attacker-chosen fields.

---

## 4. Design — engine, GPU (1.16.0)

Files: `src/gpu/gpu_1p0.cpp`, `include/superslm/gpu_1p0.h`, the blob code shared with §3 (one writer and one
reader of `SPX1`, parameterised by backend; the project's one-real-implementation rule). Built on the 1.13.0 tree
with PKV's GPU prefixes (PKV §4 items 3, 7, 8, 10); re-anchored at F0 because none of it exists at `5d5ca58` (S13).
*Rev 1.18 (PC-1 renumbered, PC-14).* The twins release as **1.16.0**, after 1.14.0 and the separate fast-decode
release 1.15.0. F0–F2 are built on `claude/project-thread-wt2dpx` (tip `83a82c2`) in the cloud on a software GPU
(lavapipe), not yet on either certified GPU, and one `SPX1` codec serves both backends (`3a00abb`). Because the line
lands on `main` as one squash (M1), 1.14.0's tree carries this code. *Rev 1.18.1 (the conductor's ruling on PC-14
(c)):* the two GPU verbs are declared and exported in 1.14.0, and labelled **unreleased until 1.16.0**, with no
promise of §3.4 or of this section attached to them in 1.14.0 or 1.15.0; their `SslmGpuStatus` ordinal is fixed at
the squash and does not move at 1.16.0 (no GPU status is appended after `SSLM_GPU_PREFIX_SCHEMA_MISMATCH` on `main`
or on the fast-decode line). The relabel is **every file**, not two (plan review F5, audit G3): at `83a82c2` this
plan's GPU code is labelled "(1.15.0)" in `include/superslm/gpu_1p0.h` (`:302`, `:1029`),
`include/superslm/gpu_1p0_bench_bridge.h:44`, `include/superslm/gpu_port.h:1127`, `src/gpu/gpu_1p0.cpp` (`:5874`,
`:6515`), `docs/api.md` (`:368`, `:375`, and `:518`'s status paragraph), and three test mirrors
(`tests/t2112-gpu-1p0-red-suite/sslm_gpu_1p0.h:140`, `tests/t2791-gpu-prefill-read-red-suite/cell_status_ordinals.cpp:64`,
`tests/t2956-token-finish-red-suite/cell_status_ordinals.cpp:18`). M1 rewrites each to name 1.16.0 and, in the public
header and `docs/api.md`, "unreleased"; cell 7.22's source check pins it, and runs at E5 and at the 1.15.0 release.
Items 1–3 and 7 below record F2's built rules, which the principal accepted on 2026-10-17 (PC-15 Q1, Q3).

1. **`sslm_gpu_prefix_save(ctx, prefix, out_blob, out_blob_size)`**. Gathers rows `[0, L)` from the pool's
   **host mirror**, which is complete for every written row (PKV G24, §4 item 10), and the prefix's internal
   sequence state. `backend = 1`; producer adapter = the context's adapter `VendorId` and `DeviceId` (the D3D12
   adapter description carries both, S24; F0 names the engine's accessor, §14). Unfrozen prefix → `SSLM_GPU_PREFIX_REJECTED` (PKV §4 item 7); a handle of another
   context → `SSLM_SEQUENCE_KV_BUFFER_MISMATCH`; a short buffer → `SSLM_OUTPUT_BUFFER_TOO_SMALL` with the size
   written (S9). **Does not submit** (a row for PKV §4.4's derived table), so it is not guarded and needs no Busy
   check. *Rev 1.18 (Q3, F2 readings R1, R2, R4):* the GPU prefix handle carries `current_token` (−1 while empty, set
   by prefill, copied by `begin_from`), and the GPU model stores the artifact's `kv_precision` at map. A GPU prefix
   at rest is not the CPU's `layer_index == 0`: save accepts exactly **(a)** `L > 0`, at full depth (`layer_index ==
   num_hidden_layers`), ready for logits, with `current_token` in `[0, vocab)`, or **(b)** `L = 0`, at layer 0, not
   ready, with `current_token == −1`. Any other frozen state is `SSLM_GPU_PREFIX_REJECTED`, beside "unfrozen".
   *Rev 1.18.1 (audit G5):* cell 11.8 [F2] grades it.
2. **`sslm_gpu_prefix_restore(ctx, model, pool, blob, blob_size, parent_or_null, out, out_shared_pages)`**. §3.4's
   rule on the GPU pool: validate everything, draw `ceil(L/B) − s`, then share `s` parent pages (*rev 1.5*: §3.4's
   order, draw then share; F0 re-anchors whether the GPU pool's primitives are also two sections and where a share
   refusal lands in item 3's order; *rev 1.6:* F0 records whether a GPU share can refuse after the draw, where, and
   with which `SslmGpuStatus`, and cell 5.11's GPU half [F2] receives it. A change this makes to a §4 rule is a fold
   before F1, R2) under condition (a)–(f) with (f) comparing the parent's **host-mirror** rows (as PKV §4 item 10 does for sequences), write the
   private rows into the host mirror, and upload **only those pages** through `UploadPoolRegionsSync` on the process
   singleton with the pool latch rule (PKV §4 item 8). It submits GPU work, so it is **guarded**: a row for PKV
   §4.4's derived table (sink `UploadPoolRegionsSync`, singleton, scope "any sequence"), `EntrySubmitGuard` placed
   by PKV §4.4's criterion, a `catch` row in the site census (PKV §4 item 9), and PKV's debug-layer requirement for
   every cell that runs it. *Rev 1.18 (Q3, R3, R5):* `SPX1` carries no ready flag, prefill snapshot or bind
   permission, so restore derives a frozen prefix's values from `L`: ready, and the prefill snapshot valid (set to the
   residual), iff `L > 0`; bind permission iff `L = 0`. *Rev 1.18.4 (code review round 1, the FIX):* **freeze sets
   the same values from `L`**. A frozen GPU prefix's prefill-snapshot state and its bind permission
   (`bind_eligible`) are set at freeze from its length, by the rule above, whatever its prefill history was. A
   prefix and its save-then-restore copy therefore agree for every history. That includes a budget stop followed
   by a retry that lands nothing, a refusal after earlier calls landed tokens, and an empty prefix given a
   zero-count prefill (6.8's legs A, B and D). Rev 1.18 set them only on restore, and on the original only at
   item 7's two stopping exits. The GPU share **can** refuse after the draw (rev 1.6's
   question, settled): every drawn page is returned and the status is `SSLM_SEQUENCE_KV_BUFFER_MISMATCH` for both
   page-module refusals (`kInvalidArgument`, `kRefcountUnderflow`), one status where the CPU gives two, as GPU
   `begin_from`'s share refusal is. Cell 5.11's GPU half grades exactly that; it is no longer pending on F0.
   *Rev 1.18.1 (audit G5):* cell 8.16 [F2] grades the derived ready and bind-permission state against prefill twins.
3. **Status precedence, a total order** (*rev 1.1*; the first failing check decides; each refusal changes
   nothing; validation precedes the draw, unlike today's GPU sequence restore, S19). It is §3.4's order with
   `SSLM_BUSY` inserted after the arguments:
   1. null or cross-context arguments (`model`, `pool`) → `SSLM_SEQUENCE_KV_BUFFER_MISMATCH` (PKV §4 item 7's rule;
      the hint is not a checked argument — a hint of another context or model gives a private restore, as PKV
      §3.9 A13);
   2. `SSLM_BUSY` while any sequence in the process is Submitted (PKV §4 item 4);
   3. size below 184 or magic not `SPX1` → `SSLM_SEQUENCE_KV_BUFFER_MISMATCH` (S9);
   4. digest → `SSLM_SEQUENCE_KV_BUFFER_MISMATCH`;
   5. `backend != 1` → `SSLM_SEQUENCE_KV_BUFFER_MISMATCH`;
   6. producer (engine version; adapter vendor and device) → new `SSLM_RESTORE_PRODUCER_MISMATCH`, appended to
      `SslmGpuStatus` after PKV's six and mirrored in both `SslmGpuStatus` mirrors (PKV G35's two ordinal cells,
      S27; *rev 1.5*: rev 1.4 put those cells in the CPU step);
   7. model hash → `SSLM_RESTORE_MODEL_MISMATCH`;
   8. `kv_precision` → `SSLM_SEQUENCE_KV_BUFFER_MISMATCH`;
   9. schema (S6's rules) → `SSLM_SEQUENCE_KV_BUFFER_MISMATCH`;
   10. domains (§3.4 step 8, same order) → `SSLM_SEQUENCE_KV_BUFFER_MISMATCH`;
   11. `SSLM_GPU_KV_POOL_EXHAUSTED`;
   11a. *(rev 1.18, Q3)* a share refusal after the draw → `SSLM_SEQUENCE_KV_BUFFER_MISMATCH`, every drawn page
       returned (item 2);
   12. the upload's device statuses by PKV §4.4's classifier (`SSLM_GPU_ALLOCATION_FAILED`, `SSLM_DEVICE_LOST`),
       with every drawn page and shared refcount returned.
4. **Schema identity on GPU** is written as the name hash (S6), resolved through the model's schema table at
   restore, and unique under PC-12 (*rev 1.3*); a GPU handle's index is converted at save. With one artifact the two agree (S10). *Rev 1.5:* the GPU
   model holds the same `SchemaMasksTable` as the CPU, and its entries carry `name_hash` (`gpu_1p0.cpp:351`,
   `:1247`; `schema_masks.h:129`, `:403`), so the hash is available on GPU with no fallback.
5. **Vendor scope (PC-5, *rev 1.1*).** Until F3's byte-identity cells (4.6) pass, a GPU blob restores only on
   the adapter vendor and device that wrote it (step 6). The granularity is the **device**: a cooked GPU entry hits
   only on the exact device it was built on (a 2080 SUPER or a 7900 XTX), and every other card, even of the same
   vendor, recomputes (*rev 1.2*). On a GPU that is not one of the two certified ones, the fidelity promise
   (§3.4 (i)) holds relative to that device's own prefill, which is all the plugin claims for GPU
   determinism today ("per device and not claimed in 1.0").
6. **Consumer effect.** The GPU verbs exist only for pooled GPU prefixes (PKV §4 item 12). A host on dedicated GPU
   sequences has no GPU prefix to save. That is why the plugin's GPU cache waits on PC-6.
7. **The prefill snapshot of a prefix stopped after landing tokens** (*rev 1.18*, PC-15 Q1; F2's V5 and its sibling).
   A GPU prefix's prefill that lands tokens and then stops, either on its budget (`SSLM_GPU_KV_BUDGET_EXCEEDED`,
   `fdbb97b`) or because the bound schema rejected a token (`SSLM_SEQUENCE_REJECTED` from the DFA exit, `83a82c2`),
   leaves the prefix ready at full depth. Its prefill snapshot is then **refreshed** from the residual and marked
   valid, exactly as item 2's restore derives it, so adopters of the original and of a save-then-restore copy read
   the same `sslm_gpu_seq_read_prefill_final_hidden` result. Refresh, not refuse: refusing the save would make a
   valid prefix a cache miss on GPU while the CPU saves it (the CPU has no snapshot). The rule is the prefix layer's
   only (kind 1); a plain sequence's snapshot contract (successful prefill only) is unchanged. Cell 6.8 [F2].
   *Rev 1.18.4:* item 2's freeze-time setting now gives this outcome for every history, and the exit refresh is no
   longer what carries it. Whether the exit refresh stays in the code is the builder's call under the review's
   remedy. The outcome promised here is unchanged.

---

## 5. Design — the plugin's prefix cache

Files (Runtime module): new `SuperSLMPrefixCache.h/.cpp` (backend-neutral policy: key, tiers, budget, LRU, pins,
coalescing, disk IO), `SuperSLMPrefixCacheBlob.h/.cpp` (the disk wrapper), additions to `USuperSLMSubsystem`
(CPU) and, at U6, `USuperSLMGpuSubsystem`; the Blueprint library. Editor module: the prefix-set build action and a
commandlet. One cache implementation serves both backends; each subsystem owns one instance bound to its own
pool.

### 5.1 Prerequisite: the plugin on paged KV

The plugin runs legacy mode today (S15): warm sequences from `sslm_seq_create`, prefixes from `sslm_prefix_begin`,
GPU sequences dedicated (S16). After the re-pin (U0) that keeps working, and PKV makes a legacy adopter **copy**
the prefix's occupied pages, `O(L)` rather than the whole block (PKV §3.5). So on the CPU:
- **without** the plugin's move onto budget mode: the cache works; each adopt copies ≤ 5.9 MiB for a 1,000-token
  0.5B prefix; no pages are shared; eviction frees all of a prefix's pages;
- **with** it: adopters share full pages; eviction releases the handle and adopters keep shared pages by refcount
  (PKV §3.5), as the principal's default states; nested entries (`begin_from`) become possible.

On the GPU nothing works without that move: GPU prefixes exist only for pooled GPU sequences (PKV §4 item 12).
PKV assigns that move to the plugin to plan, and no plan covers it (S16; the project's plan folder has none).
**PC-6:** this plan does not absorb it. U0–U5 ship the CPU cache on the plugin's current mode, written so the
switch to budget mode changes no cache code (the cache holds handles and calls the subsystem's own adopt). U6–U9
wait on that plan's release and on 1.16.0 (*rev 1.18*: rev 1.17 said 1.15.0).

*Rev 1.13.* That plan exists and ships first: the plugin paging plan (rev 1.7,
`/mnt/project-files/plans/superslm-unreal-paged-kv/plan.md`, cited as *paging*) releases on a SuperSLM 1.13.0 pin
before the CPU-cache release (its UP-1, decided 2026-10-15 under the principal's delegation). So U0–U5 land on a
plugin with both modes: legacy (`SequenceBudget = 0`, the "without" case above) and budget (the "with" case, from
U2, not U6). §5.11 folds what that changes here; nested cache entries and the GPU cache still wait for U6.

### 5.2 The cache key

`Key = SHA-256` (Layer 1's, S17) over, length-prefixed: the model artifact hash; the compiled Layer-1 tag and
commit (the plugin's engine identity, S14); the backend and, on GPU, the adapter's vendor and device ids
(PC-5, *rev 1.1*); the blob magic `SPX1`; adapter = none; the bound schema's
name (empty = none); the token ids; and the parent's key (absent for a root). The parent's key, not only the full
token list, because a nested prefix's identity includes where it was cut; content identity across different cuts
is not a fact this plan relies on.

Any change of model, engine pin, backend, format, schema, tokens or parent is a different key: a miss, never a
wrong hit. *Rev 1.3:* the key hashes the full schema name, but the `SPX1` payload carries only its hash. So a hit
from a byte-storing tier returns the right binding only under PC-12. A model that breaks it is refused at map, so
`Configure()` fails by name, and the editor's prefix-set build refuses it too (cell 2.15). A hash collision is not trusted either: every tier stores the full spec beside the entry and compares
it on a hit — the disk and cooked tiers on load (§5.7), and (*rev 1.1*) the memory tier on the game thread at
the hit, an O(tokens) compare of at most 128 KiB at cap 32,768.

### 5.3 Tiers

1. **Memory:** resident frozen handles, owned by the cache, in the subsystem's pool. A hit costs one lookup; the
   caller then adopts as today.
2. **Cooked:** blobs shipped inside a `USuperSLMPrefixSet` asset (§5.8), registered at runtime. A hit restores
   from bytes already in memory.
3. **Disk:** one file per key under a configured directory (default `Saved/SuperSLM/PrefixCache/`), read
   asynchronously. Off by default (PC-10). When on, a recomputed prefix is saved and written back.
4. **Recompute** (*rev 1.1*, PC-11): `sslm_prefix_begin_budgeted(L)`, not legacy begin; then, for a spec with a
   `SchemaName`, resolve it by name (`sslm_schema_lookup`, as the plugin's registry does, S23) and bind it with
   `sslm_prefix_set_schema` before the first prefill (which S4 requires); then prefill on worker lanes and freeze.
   An unknown name resolves the request `UnknownSchema` by name, with no entry and no file (§5.9). When the disk
   tier is on, save and write back. A budgeted prefix in the plugin's legacy pool is legal, and a legacy adopter
   copies it (PKV §3.5, A20). *Rev 1.8 (superseded by rev 1.18):* begin was its own worker job, with prefill and
   freeze in later jobs, so the drawn-before-freeze state was observable at the between-jobs barrier (8.10(c)).
   *Rev 1.18 (PC-15 Q7; the U2 draft's R13):* **on the CPU a cache build is one lane job**: begin, bind, the whole
   prefill and freeze, priced at `PrefixBeginCostMs`. It is not sliced into per-tick prefill jobs as `CreatePrefix`
   is; slicing is a box refinement if U4's tick-budget cells (5.9) show hitches. 8.10(c) keeps its drawn-and-unfrozen
   reading through a test-only **build hold** inside the job, after begin returns and before the first prefill (§8,
   *Lane park*), not through a job boundary. *Rev 1.18.8 (§15.27): U4's packaged run tripped that trigger. A
   2,000-token build ran as one job of 49,889 ms of worker time against its 2.261 ms price, and was delivered 15,417
   ticks late. So **on the CPU a cache build is now sliced into prefill chunk jobs**, planned and priced exactly as
   `CreatePrefix`'s are (the same chunk planner and the same per-chunk price, with the depth term at each chunk's
   start).*
   - **The first job** begins and binds. The build hold stays inside it, after begin and before the first prefill,
     so 8.10 (c)'s reading is unchanged.
   - **Each later job** prefills one planned chunk. The job of the last chunk also freezes.
   - **The save and write-back**, with the disk tier on, are unchanged.
   - **Order and lane.** The jobs of one build run in order on the lane the build began on.
   - **Interleaving.** Between chunk jobs, other jobs on that lane may run, and that is the point of the change.
   - **One request throughout.** The build's charge, pins, coalescing, waiters and in-flight status are the one
     request's from begin to freeze, as for the GPU's chunks below. Eviction never takes a building key, and a drop
     of one is deferred, as before.
   - **At a re-`Configure()`**, the chunk running drains with the teardown. No further chunk is issued, the request
     resolves `Cancelled` (§5.6, cell 1.7), and the unfrozen prefix is released exactly once before the pool is
     destroyed.

   The prefill counters (§15.26) count one call per chunk, as they already do. Slicing changes when the build's
   chunks run, never what it computes: the frozen prefix and its save are byte-equal to the one-job build's.
   **On the GPU** (U6) a cache build is split into **token-chunk
   submission-thread jobs**, each sized so its work stays under the submission thread's lock-holder bound (the
   frame-path bound, 10 s, past which the thread is marked unresponsive and the device-loss path runs). The chunk
   size is derived per token, as the plugin's load-time calls derive their bounds; the chunks of one build run in
   order and the build's charge, pins and coalescing are the one request's, as for a single job. Cell 5.13.

Lookup order: memory, cooked, disk, recompute. A tier that fails for any reason (§5.9) falls through to the next.

### 5.4 Budget, eviction and pins

- **Budget:** `PrefixCacheBudgetBytes` (config). An entry's cost is its private pages × page size
  (`sslm_kv_page_size`, PKV §3.6), known from `L` and `s` before the restore.
- **The charge rule** (*rev 1.2*; one rule for builds and restores). The ledger counts every page the cache holds
  **or may draw**: resident entries, ghost charges, and every build or restore in flight.
  - *(Rev 1.4.)* A nested restore is charged with `s = 0`, i.e. `ceil(L/B)` pages, because whether the engine
    shares is decided only inside the restore (condition (f)). When the result is applied, the charge falls to the
    private pages actually drawn (`ceil(L/B) − s`, with `s` from `*out_shared_pages`, which the ledger reads as
    bookkeeping, not as a verdict). So a hint that fails can never draw past the ledger. A nested restore whose
    `s = 0` charge does not fit resolves as §5.9 says: evict and retry, then `BudgetExhausted` (or `PoolExhausted`
    from the pool) by name. It is never retried as a `begin_from` build on the promise of build-equivalent
    admission. The request's caller may ask again once the parent is resident and the budget allows.
  - Each cache build or restore is charged at its **peak draw**, **before it draws**: a restore `ceil(L/B) − s`
    pages, a build `R(L)` pages (PKV §3.4's formula at budget `L`, at most `ceil(L/B) + 1`).
  - The charge is taken on the game thread when the request is admitted. The cache evicts least-recently-used
    **unpinned** entries until the charge fits. *Rev 1.14:* an evicted entry's charge is refunded only when its
    release has applied (§5.11 (a)), so "fits" counts the request's claims on pending refunds (*rev 1.15:* unclaimed
    refunds first, then its own evictions'), and a request that fits only by them waits for those applies and is
    admitted at the last one (§5.11 (a)'s deferred-refund admission).
  - If nothing more can be evicted and the charge still does not fit, *counting every pending refund it can claim*
    (*rev 1.14*; *rev 1.15*: unclaimed refunds included), the request resolves **`BudgetExhausted`**
    by name and draws nothing. It never overdraws, and pinned or leased entries do not change this.
  - At freeze a build's charge drops to its `ceil(L/B)` pages. A refused or failed build or restore returns its
    whole charge when its result is applied.
  - So the ledger never exceeds the budget, and every page it counts is one the cache holds or may still draw.
- **Ghost charges** (*rev 1.1*; nested entries, U6). When a parent is evicted while cached descendants share its
  full pages, those pages stay in the pool, held by the descendants. The parent's charge therefore stays on the
  ledger as a ghost charge until its last cached **descendant** leaves the cache (*rev 1.2*: rev 1.1 said "child",
  and a grandchild holds the grandparent's pages too). The ghost keeps the parent's whole charge, including a partial
  tail page that eviction frees, so it overcounts by at most one page per ghost. That is safe. Cell 1.9 measures
  it. (U2 has no nested entries, so no ghost charge arises there.)
- **Counting against the plugin's caps** (*rev 1.1*, PC-11). Cached entries are not in `State->Prefixes` and do not
  count against `PrefixBlockCount` (S23). The cache's pages come from blocks added to the pool at `Configure()`:
  `ceil(PrefixCacheBudgetBytes / block bytes)` blocks beside `BlockCount + PrefixBlockCount`, through the existing
  `sslm_kv_pool_create`. *What this buys, exactly (rev 1.2):* by the charge rule, the pages the cache holds or has
  in flight never exceed the budget, and so never exceed the extra blocks' pages. The pool's other
  `BlockCount + PrefixBlockCount` blocks therefore stay available to 1.0.0's warm sequences and live prefixes, with
  every entry leased and with builds in flight (cell 8.10). This holds while adopters copy (U2–U5). From U6,
  budget-mode adopters can keep pages of an evicted entry alive. Those pages belong to the sequences, and the
  plugin paging plan's lifecycle budget covers them, not this ledger. *Rev 1.13:* with paging first, "while
  adopters copy" is **legacy mode**, where this paragraph stands as written. In budget mode, adopters share from
  U2, and the extra blocks are region C of paging's ledger; §5.11 gives that mode's argument.
- **Legacy admission** (*rev 1.2*). PKV §3.5 limits its legacy-admission promise to pools in which every handle came
  from a no-budget verb. Under PC-11 the plugin's pool always holds budget-mode handles: budgeted builds, and every
  restore records `budget`. The promise still holds here, because budget-mode accounting differs only through
  budget **sharers**, and the CPU plugin has none until PC-6's plan. *Rev 1.13:* that remains true in legacy mode,
  where paging refuses nesting and `SSB6` (paging §4). In budget mode PKV's legacy-admission promise does not apply,
  and paging's ledger replaces it (§5.11).
- **Pins:** an entry is pinned while any operation that needs it alive is queued or running: an adopt of it, a
  save of it, a child restore or build using it as parent. *Rev 1.16 (strike F1):* a request **waiting** under §5.11
  (a) to restore or build a child pins its spec's parent from its decision, exactly as its queued job would. *Rev 1.17 (re-strike
  R2), a nested request's own eviction loop and its parent:* rev 1.16 said the loop "never evicts its parent", which
  refuses a nested **restore** that §5.4 as approved admits: the approved text pins a parent only while a child job
  is queued or running, none is queued during the request's own loop, and a nested restore is charged with `s = 0`,
  so it can restore privately without its parent. This plan adopts that reading for restores: **a nested restore's
  loop may evict its own parent** (the parent was touched by the request, so it is the most recent entry and goes
  last), and the restore then runs with a null hint, private, at the charge it already carries. A nested **build**
  (`begin_from`) cannot run without its parent, so **its loop never evicts its parent**. Against §5.4 read
  literally, that is an added refusal: a build that fits only by evicting its own parent is refused
  `BudgetExhausted`, where the literal loop would evict the parent, admit the build, and start a `begin_from`
  against a released handle (caller UB, §3.7; in the model it also overdraws the pool, `fold_runs.out` D). The
  literal reading is therefore not sound for builds, and the only sound reading of the approved text refuses the
  build too. It does not change PC-11: a request whose charge can be met only by releasing what it needs "cannot be
  charged", the named refusal is PC-11's stated cost, and nothing is overdrawn. The conductor may still put it to the
  principal as a disclosure (§15.17). Releasing a prefix under those is caller UB in the
  engine (§3.7), so the pin is what makes eviction safe. An entry leased to a caller (§5.6) is pinned until the
  lease is returned.
- **Parents:** a parent is evictable while its children stay cached; children keep its shared pages by refcount.
  Touching a child also touches its parent, so parents age more slowly than their children.
- **The pool is the hard limit.** The budget is a soft limit on what the cache holds; the pool can still be short
  when evicted prefixes' pages are held by budget-mode adopters (PKV §3.5, from U6). A restore or build refused
  `SSLM_KV_POOL_EXHAUSTED` evicts the next LRU unpinned entry and retries, until it succeeds or nothing is
  evictable; then the request resolves `PoolExhausted` by name (§5.9). No request waits on memory it cannot get.
  Before U6 this path is defence in depth: the charge rule keeps the cache within its own blocks. *Rev 1.13:* in
  budget mode from U2 too, because paging's region A holds the pages adopters keep after an eviction (§5.11).
- **Builds** (*rev 1.1*, PC-11): each recompute draws `R(L) ≤ ceil(L/B) + 1` pages and holds `ceil(L/B)` after
  freeze. No build reserves a whole context, so rev 1's build-concurrency rule (`PrefixBlockCount` recomputes in
  flight) is removed. A build refused `SSLM_KV_POOL_EXHAUSTED` follows the evict-and-retry rule above.
- **Coalescing:** a second request for a key whose load or build is in flight attaches to it; one build per key.
- **Re-`Configure()` and teardown** release every cached handle with the pool, as live prefixes are today (S15),
  after in-flight disk IO for this subsystem is cancelled or drained (*rev 1.18.2:* a drop's delete is always
  drained, never cancelled, §5.6 *Generations and teardown*), and after every restore or build the cache has
  queued on a worker lane has drained (cell 1.7). *Rev 1.18.5 (U2 code review, findings 1 and 2):* the drain is a
  property of the path the product takes. That path is the host's abandon-then-pool-destroy order at
  re-`Configure()` and at `Deinitialize()`, not only the core's `BeginTeardown()`, which the CPU host never calls.
  - **A drop's delete that waits on a write-back's save is drained under the old configuration,** with that
    configuration's disk-tier setting and directory, before the pool is destroyed. The save runs or fails, its
    completion is applied, and then the delete runs and its op resolves.
  - **The delete is never carried into the next `Configure()`.** It would run there under the new configuration's
    settings, and at `Deinitialize()` there is no next `Configure()` at all.

  This clarifies the approved rule and does not change it. Cell 3.10 (iii) runs on that path.
- **What a release's status means** (*rev 1.18.1*, plan review F1, audit G13 (a); the conductor's ruling). Rev 1.18
  recorded, as Q9's premise, the U2 draft's rule (its V12, `e7811590e`): a release the engine fails "has not
  applied", refunds nothing, stays pending and is issued again. **That is false against the engine and is
  withdrawn.** `sslm_prefix_release` deletes the handle on every non-null call, whatever `ReleaseHolder` returns
  (`src/sslm_abi.cpp:2610-2619` at `83a82c2`), and `sslm_gpu_prefix_release` consumes it on every status except
  `SSLM_BUSY` (`src/gpu/gpu_1p0.cpp:5597-5612`: "Only SSLM_BUSY leaves the internal sequence alive"). Issuing such a
  release again releases a freed handle (caller UB, §3.7), and holding its refund pending forever would also leave a
  waiter with a claim on it waiting for good. The rule is:
  - a release that **returned**, with any status but GPU `SSLM_BUSY`, **has applied**: its family's pending refund
    advances as for `SSLM_OK` (§5.11 (a)), and a non-`OK` status is logged once as `ReleaseFailed`, a diagnostic and
    a fallback-reason count, never a retry;
  - only a GPU release that returned `SSLM_BUSY` is issued again (the handle is still live), at a later tick, and its
    refund stays pending until a later issue returns anything else;
  - a CPU release on a null handle, and a GPU release given another context's handle, are plugin bugs: they are not
    retried, they are logged, and debug builds assert. *Known limit (rev 1.18.2; the strike on rev 1.18.1, lesser
    plane 4):* the second case returns `SSLM_SEQUENCE_KV_BUFFER_MISMATCH` **without consuming** the handle
    (`gpu_1p0.cpp:5599` at `83a82c2`, the `prefix->ctx != ctx` check). That is the same status a consumed refcount
    underflow returns, so the plugin cannot tell the two apart. Under the rule above it counts the release as
    applied and refunds the pages of a handle that is still live. The precondition is a plugin bug (the cache
    releases only handles it made on its own context), and the plugin does not try to separate them;
  - *GPU `SSLM_BUSY` is unreachable from the cache* (*rev 1.18.2*, lesser plane 3, read at source):
    `sslm_gpu_prefix_release` returns `SSLM_BUSY` only when the prefix's own internal sequence is `Submitted`
    (`gpu_1p0.cpp:2272-2275`, `CallProceedsOrBusy_SeqRelease`). That sequence is `Submitted` only inside the
    synchronous prefill's scope guard (`:4497-4510`), and the cache never releases an entry whose build is running.
    The re-issue branch is defensive, so cell 5.15's [U6] leg forces it with a plugin seam (§8) rather than an engine
    state;
  - at teardown and re-`Configure()` the same rule holds; a `SSLM_BUSY` release at teardown waits for the device to
    drain as the subsystem's teardown already does;
  - *a skipped release is not a release that returned* (*rev 1.18.5*; U2 code review, finding 3). The GPU
    submission thread can skip a job instead of running it, after a terminal device loss or once it has marked
    itself unresponsive. Its release never reached the engine, so its handle is still live. It **stays pending**,
    refunds nothing, logs no `ReleaseFailed`, is not re-issued (*After a terminal GPU device loss*, below), and
    **stays in the live-handle set** until teardown's sweep releases it. The host must deliver a skipped job as
    skipped, never as a returned `Failed`. Read as a returned failure, it would be counted as applied, its pages
    refunded while still mapped, and its handle dropped from the set, so that no sweep ever releases it. Cell 5.14.

  The U2 draft's V12 code (Wizard `e7811590e`, `SuperSLMPrefixCacheHost.cpp` around `:938`) implements the withdrawn
  rule and needs the builder's fix at U2. Cell 5.15 [U2] grades the rule.
- **After a terminal GPU device loss** (*rev 1.18*, PC-15 Q9; the U6 draft's W3). After a terminal device loss
  every job is **skipped**: it never ran, so its handle is still live, and a re-issue would spin every tick. So, once
  the GPU subsystem has taken the device-loss path, a skipped release is **not re-issued**: it stays pending and
  refunds nothing, the cache serves nothing more (every request falls back as the subsystem's own fallback says),
  and the ledger leaves with the pool at teardown or re-`Configure()`, which completes without waiting on it. Cell
  5.14.

### 5.5 Threading

| Work | Where |
|---|---|
| `RequestPrefix` / `ReturnPrefix`: copy the spec, enqueue, return a handle | game thread, O(tokens) copy, as `CreatePrefix` does today |
| `WarmPrefix` / `DropPrefix` (*rev 1.18*): copy the spec, enqueue, return an op id; a drop's decision and its mark | game thread: the call; the decision and the mark at `Tick()`'s apply pass, beside eviction |
| A drop's disk delete (*rev 1.18*) | a pool thread, serialized per key with write-back and (*rev 1.18.2*) the key's disk reads and (*rev 1.18.5*) the touch of a hit's file time, and exclusive with the trim |
| Key hashing, disk file read and write, wrapper parse and spec compare, directory scan and trim | a pool thread (`Async(EAsyncExecution::ThreadPool, …)`, S17) |
| Memory-tier lookup by key, and the memory hit's spec compare (*rev 1.1*) | `Tick()`'s apply pass, game thread: a map lookup and an O(tokens) compare, bookkeeping of the kind `CreatePrefix`'s copy is |
| `sslm_prefix_restore`, `sslm_prefix_save`, prefill, freeze, release, adopt | plugin worker lanes, like every Layer-1 call (S17) |
| Publishing results (phase, source, error) | `Tick()`'s apply pass, as existing results are |

**Ownership** (*rev 1.1*). The game thread owns all of the cache's mutable state: the entry map, the LRU order,
the ledger, leases, pins and coalescing lists. It changes them only inside API calls and in `Tick()`'s apply pass,
so no lock guards them. A pool-thread or worker task gets an immutable copy of what it needs (spec, key inputs,
blob bytes, a handle), never a reference into the cache. It posts its result to a completion queue that `Tick()`
drains. Eviction is decided on the game thread, and the entry leaves the map there. The `sslm_prefix_release` itself
runs on a worker lane. Cell 3.9 pins this.

A memory hit is therefore delivered at a tick, not inside the call; the cost is one tick of latency, the same
delivery rule every queued plugin operation has. *Rev 1.14:* a request that needs evictions is admitted only at the
apply pass of the last release it waits on (§5.11 (a)), so it is delivered at least one tick after the releases'
lane jobs run, by the same rule. The worker lane prices a restore by bytes (a per-MiB cost from the
calibration command's existing per-unit model, measured at U4), so a large restore is planned across ticks like a
long prefill and a late one is counted as a hitch, never hidden.

### 5.6 API

Additive to `USuperSLMSubsystem` (and, at U6, the GPU subsystem):
- `bool RequestPrefix(const FSuperSLMPrefixSpec& Spec, FSuperSLMPrefix& OutPrefix, FString& OutError)`.
  `FSuperSLMPrefixSpec { TArray<int32> Tokens; FName SchemaName; FSuperSLMPrefix Parent; }` (`Parent` invalid for a
  root; nested specs are refused `NestedRequiresBudgetMode` until U6; *rev 1.13:* PC-6's plan now lands first, and the value is its public one, paging §3.6). Returns a leased handle whose
  phase moves to Ready as today; refused synchronously only for a malformed spec (empty, out of vocabulary, over
  `context_cap`), as `CreatePrefix` refuses. An unknown `SchemaName` is found on a worker lane (resolving it is a
  Layer-1 call) and the request resolves `UnknownSchema` by name at a tick (*rev 1.1*). A request whose charge
  cannot fit resolves `BudgetExhausted` (§5.4, *rev 1.2*).
  **One handle per lease** (*rev 1.2*): each `RequestPrefix` mints its own `FSuperSLMPrefix` from the plugin's
  existing prefix-id counter (`GNextPrefixId`, `SuperSLMSubsystem.cpp:84-87`). Several leases on one entry are
  several handles, and a cache handle can never share an id with a `CreatePrefix` handle.
- `ESuperSLMPrefixSource GetPrefixSource(const FSuperSLMPrefix&)`: `Memory`, `Cooked`, `Disk`, `Recomputed`,
  `Pending`. Diagnostic; no behaviour reads it.
- `bool ReturnPrefix(const FSuperSLMPrefix&)`: ends the lease; the prefix stays cached until evicted. Returns
  false for a handle it does not end a lease on (§5.10).
- `void RegisterPrefixSet(USuperSLMPrefixSet*)` / `UnregisterPrefixSet`: the cooked tier.
- Diagnostics: resident entries and bytes, hits per tier, evictions, fallbacks by reason.
- `AdoptPrefix` accepts a leased cached prefix unchanged. Today it finds handles only in `State->Prefixes`
  (`SuperSLMSubsystem.cpp:2476`), and cache entries are not there, so U2 adds a cache lookup to it (*rev 1.2*).
  `CreatePrefix` and `ReleasePrefix` keep their behaviour for uncached prefixes. Cache-owned handles against the
  existing verbs: §5.10.
- `ESuperSLMRestoreResult` gains `PrefixLeaseEnded`, appended last with its Blueprint mirror (the enum's own
  append rule).
- Config (`FSuperSLMRuntimeConfig`): `PrefixCacheBudgetBytes`, `bPrefixDiskCacheEnabled` (default false, PC-10),
  `PrefixDiskCacheDirectory`, `PrefixDiskCacheMaxBytes`.
- Blueprint: `Request Prefix`, `Return Prefix`, `Get Prefix Source`, `Register Prefix Set`.

**Explicit residency control** (*rev 1.18*, PC-16 (a)). Two verbs let a caller put a spec in memory ahead of need and
take one out now. Both are additive: no existing verb, result or promise changes, and neither can end, invalidate or
reach through a lease. Both return at once and resolve at a tick, because the key is hashed on a pool thread (§5.5).
- `bool WarmPrefix(const FSuperSLMPrefixSpec& Spec, FSuperSLMPrefixOp& OutOp, FString& OutError)`. **A request
  that takes no lease.** It is refused synchronously exactly where `RequestPrefix` is (a malformed spec;
  `NestedRequiresBudgetMode` before U6). Otherwise it runs `RequestPrefix`'s path unchanged: the same key, the same
  tier order (memory, cooked, disk, recompute), the same charge rule and eviction loop, deferred-refund admission and
  waiter rules (§5.4, §5.11 (a)), the same coalescing (a warm attaches to a load, build or waiter in flight for its
  key, and a later `RequestPrefix` attaches to a warm's), and the same write-back. A memory hit touches the entry in
  the LRU order. When it completes, the entry is resident and, holding no lease, **unpinned**: a warm is advisory,
  and the entry is evictable from then on like any entry whose leases have all been returned. While it is in flight
  or waiting it pins what any request pins (a nested warm pins its spec's parent from its decision, §5.4 *Pins*).
- `bool DropPrefix(const FSuperSLMPrefixSpec& Spec, ESuperSLMPrefixDropScope Scope, FSuperSLMPrefixOp& OutOp,
  FString& OutError)`, `Scope` ∈ {`Memory`, `MemoryAndDisk`}. Refused synchronously only for a malformed spec. The
  memory part is decided at the apply pass that receives the key:
  - *Resident and unpinned* (no lease, no queued or running adopt, save or child job, no waiter pinning it as a
    parent): the entry is **evicted at that apply pass**, an eviction decided outside any request's loop. It leaves
    the map there (§5.5), and its charge becomes a **pending refund with its family**, unclaimed, completing when the
    last release of its family has applied (§5.11 (a), *Pending refunds wait on their family*). Result `Dropped`. A
    parent dropped while cached children stay is evicted as §5.4 *Parents* says, its ghost charge included (U6); a
    drop never names or reaches a descendant.
  - *Pinned, or its key in flight or waiting*: nothing is released, cancelled or unpinned. **A drop of a leased entry
    never invalidates a live lease**: every lease stays valid, `AdoptPrefix` through it adopts, and the entry keeps
    serving it. The key is **marked**, and the mark is applied, as the drop above, once the entry is resident and
    unpinned, in the order *When a mark is applied* states (below; *rev 1.18.2*). Result `DropDeferred`, resolved at
    once; the mark's later application is counted in the diagnostics.
  - *Not resident, not in flight and not waiting*: `NotCached`.
  - A later `RequestPrefix` or `WarmPrefix` for a marked key **clears the mark**: the live request wins, and is
    served as it would be without the drop. A drop never cancels engine work, a waiter or a coalesced attach.
    (*Rev 1.18.2:* "later" is issue order, below, and the request clears the mark when it is applied before the
    mark's application, which *When a mark is applied* places.)
  - *What "later" means* (*rev 1.18.1*, plan review F8, audit G10; the conductor's ruling): **issue order**, the order
    of the calls on the game thread, never the order in which their keys' hashes complete on pool threads. Every
    `RequestPrefix`, `WarmPrefix` and `DropPrefix` takes an issue sequence number from one game-thread counter at the
    call. Since an op's key is unknown until its hash completes, the apply pass applies these ops in issue order
    across the subsystem: an op whose hash has completed waits behind any earlier-issued op whose hash has not (a cost
    of at most that earlier op's hash latency, within §5.5's delivery rule). So a drop issued before a request is
    decided before it (the mark is set, then the request clears it), and a drop issued after a request is decided after
    it (it finds the request's lease and marks). Cell 3.12 pins both orders against both completion orders.
  - *When a mark is applied* (*rev 1.18.2*; the strike on rev 1.18.1, Fracture 1; the conductor's ruling). A mark's
    application is not an op. What triggers it is a pin change: the entry's last pin going, or the entry becoming
    resident with no pin. That happens inside `ReturnPrefix` or at a completion, never at a hash. Rev 1.18.1 ordered
    op against op only. So a request issued after the drop whose hash was still pending at the pass where the last
    pin went lost to the mark and rebuilt (the strike broke 18 of 24 cells, and every cell with that hash held).
    **A mark is not applied while any op issued after its drop, and before the pass that finds the entry resident
    and unpinned, still has a pending hash. Once those hashes resolve, issue order decides.** The apply pass runs in
    this order:
    1. **Completions** are applied (lane, pool-thread and release results). A pin drained here is gone for the rest
       of the pass.
    2. **Mark applications are queued.** For each marked entry that is now resident and unpinned and has no
       application queued, its mark's application joins the issue order **after every op issued before this pass**.
    3. **The issue-ordered queue is drained** from its head while the head's hash has completed (*What "later"
       means*). An op is applied as before. A mark's application, when reached, applies the mark if the key is still
       marked and the entry still resident and unpinned. An op ahead of it that requested or warmed the key has
       cleared the mark, and the application is then void. Otherwise (a pin taken since, for example a nested
       request's parent pin, U6) the key stays marked, and step 2 queues it again at the next pass that finds it
       resident and unpinned.

    Nothing else applies a mark. Each of its six triggers reaches it only through step 2: a lease returned, a queued
    adopt drained, a write-back's save drained, a child job settled (U6), a warm's build completed, and a request's
    build completed after its only lease was returned while `Pending`. So a request issued before the pass that
    finds the entry unpinned (in the same frame as the last return, say) is applied before the mark and clears it,
    whatever its hash timing. A request issued after that pass finds the mark ahead of it, and the entry is gone once
    the earlier hashes resolve. The wait is bounded by the hash latency of ops issued before that pass. Ops issued
    after it do not hold the mark back, so a caller that issues a request every frame cannot keep a mark from
    applying. (The ruling's words, "any op issued after that drop", would let it: a fresh op with a pending hash
    is always present under such a caller. The pass bound is this plan's addition, and §15.20 records it.)
    *Executed:* `reviews/fold-rev1p18p2-probes/mark_order_model.py` is the strike's model with this rule added, and
    its first two output sections reproduce the strike's exactly.
    - The strike's 12 cells hold: six triggers, with the later request's hash completing before the pass and after
      it.
    - With marks applied as a stage before the ops, the strike's control rule holds 0 of 12. That is why the order
      is stated.
    - Six mutants each let one trigger's path apply the mark at once. Each is red on its own trigger's cell and on
      no other. That is why cell 3.13 covers all six.
    - A request issued after the unpin gets the same outcome under both hash timings.
    - The bound's one cost: a request for the key issued after that pass, while the mark still waits on an earlier
      op's hash, is a rebuild. The ruling's words would serve it from memory (section 3b).
    - Under a stream of one request a frame, each hash one frame late, the mark applies at the second frame. The
      ruling's unbounded wording never applies it in 40 frames.

    Cell 3.13.
  - The disk part (`MemoryAndDisk`) deletes the key's file on a pool thread, on the same per-key serialization as
    write-back and exclusive with the trim (§5.7). *Rev 1.18.2 (the strike on rev 1.18.1, Fracture 2 and lesser
    plane 2; the conductor's ruling):* the disk part covers what the memory part covers, **in flight or waiting**.
    Two terms are now fixed. A **write-back** is the save on a worker lane plus the file write after it. A disk
    **read** of the key is on the same per-key serialization. At the drop's apply pass:
    - a write-back of the key that is queued or running, its save or its file write, finishes first and is then
      deleted;
    - a disk read of the key that is queued or running finishes first, and serves if it hits; the delete then runs,
      so it never meets a file open for reading (on Windows that delete would fail as `Failed`);
    - a load or build of the key that is in flight or waiting **does not write back**. That covers a tier lookup, a
      waiter under deferred-refund admission (§5.11 (a)), a build queued or running, and a build refused
      `PoolExhausted` that is waiting to retry (§5.4), which counts as in flight.

    A write-back by a request admitted after the drop writes normally. *Rev 1.18.1 (plan review F9):* a request or
    warm issued after the drop that **attaches** to that load, build or waiter (*rev 1.18.2*: in flight or waiting)
    clears the mark and restores its write-back (it is served as it would be without the drop). The file is then
    written once that build completes, after the drop's delete on the key's serialization. *Executed:*
    `reviews/fold-rev1p18p2-probes/disk_states_trace.py` is the strike's census with a third row set for this rule. No
    reachable state leaves a file behind the drop (the strike's reading A left one in s3, s5 and s8; reading B left
    one in s3 and s8). Cell 3.10 (iv)–(vii). Deleting a missing file is `NoFile`, not an
    error; with the disk tier off the disk part is `DiskTierOff`; a delete the file API refuses is `Failed`, logged
    once. The cooked tier is never touched: a registered set's entry stays, and a later request restores from it.
  - *Why §5.11 (a)'s argument still holds.* A drop is an eviction that no request's loop decided. Under the approved
    rule (refund at decision) its charge would be free at once; here it is a pending refund in `U`, unclaimed, so the
    ledgers still differ by exactly `U`. A drop releases only unpinned entries and a mark pins nothing, so the pin
    sets and the evictable sets stay equal in both. The induction is unchanged. *Executed (rev 1.18; reworded in
    rev 1.18.1, audit G19)*: `reviews/fold-rev1p18-probes/drop_model.py` adds drops and marks to the rev 1.17 model
    (both model files copied unchanged); three seeds of 20,000 runs (about 136,000 requests and 38,000 drops, 29,000
    of them deferred), releases in random and in decision order: 0 divergences, 0 ledger excesses, 0 physical
    overdraws and 0 jobs against an evicted parent. Mutant m2 (a drop refunded at its decision) is caught by a harm
    observable, about 10,000 physical overdraws per seed. Mutant m1 (a drop that releases a pinned entry) is caught
    by the rule comparison, about 7,700 divergences per seed; the model's "pinned released" count is the mutant's
    own branch counting itself and is not an observable, and with runs stopped at their first divergence no later job
    could meet the released parent. *Rev 1.18.1:* `reviews/fold-rev1p18p1-probes/drop_model.py` (the same model, one
    switch added) continues m1's runs past the divergence: 537, 642 and 612 jobs against a released parent and 77, 75
    and 102 physical overdraws on the three seeds, so m1 is also caught by a harm observable; sections 1–3 and m2
    reproduce rev 1.18's output exactly. The product's harm observable for m1 is the release-counting seam in 7.18
    and 8.14. One model limit, named in its docstring: the model keys a ghost by its entry's key, so requests that
    re-create a key whose earlier instance is still a ghost (11–20 per seed) are skipped. The plugin keeps the two
    instances apart (entries are keyed by `FResidencyEntryId`, and a releasing entry is unfindable by key), and cell
    8.17 executes it.
  - *The shared core* (*rev 1.18.1*, plan review F15). The drop adds an evict-by-id and per-key marks to the
    kind-neutral residency core (`SuperSLMResidencyCore`, which today has `EvictLru` and `EvictForTeardown` only). That
    core is shared with the plugin's LoRA residency plan. The additions are additive and inert for a kind that issues
    no drop; a cross-plan note to that plan's owner is in `reviews/fold-rev1p18p1.md`.
  - *Re-`Configure()` clears every mark* is recorded as an equivalent mutant (audit N5): an entry becomes resident
    only through a request or warm, which is issued after the re-`Configure()` and clears the mark anyway. No cell
    is owed.
- `bool GetPrefixOpResult(const FSuperSLMPrefixOp& Op, FSuperSLMPrefixOpResult& Out)`: `Out.Result` ∈ {`Pending`,
  `Resident`, `Dropped`, `DropDeferred`, `NotCached`, `BudgetExhausted`, `PoolExhausted`, `UnknownSchema`,
  `Failed`, `Cancelled`, `Expired`}, and for a drop `Out.Disk` ∈ {`NotRequested`, `Deleted`, `NoFile`, `DiskTierOff`,
  `Failed`}. A warm resolves as a request would (`Resident` for any hit or completed load or build, or the request's
  named refusal). *Rev 1.18.5 (U2 code review, finding 4):* a warm whose build **fails** resolves **`Failed`**, and
  `Out.Reason` carries the §5.9 fallback reason its request would log (`BuildFailed` for an engine failure). In the
  C++ enum, `Failed` is appended last, by the enum's append rule. `Out.Reason` is meaningful only when `Out.Result`
  is `Failed`. **`Cancelled` stays
  re-`Configure()` only** (*Generations and teardown*), so a caller can always tell a failed build from a cancelled
  warm. At U6, a nested warm whose parent left resolves as its request does. An op is `Pending` until both its parts resolve. Op ids come from a counter that never repeats within the
  subsystem's lifetime; returns false for an id it never issued. *Rev 1.18.1 (audit G11, reading (b); plan review
  F16):* a **resolved** op keeps its result, across a re-`Configure()` too, until it falls out of the ring of the
  last 1,024 resolved ops, and then reads `Expired`; a **`Pending`** op is not in the ring and never expires. Rev 1.18
  also expired every op of an earlier generation, which made `Cancelled` unreadable; that clause is withdrawn.
- *Generations and teardown.* Re-`Configure()` clears every mark. *Rev 1.18.2 (the strike on rev 1.18.1, lesser plane
  1): every op still `Pending` at a re-`Configure()` resolves during it,* so none can read `Pending` forever under
  reading (b) above:
  - **A warm** resolves `Cancelled`, and reads `Cancelled` afterwards, as a queued cache request does (cell 1.7).
    This holds in every state it can be in: its hash pending, its tier check (cooked check or disk read) in flight,
    its restore or build queued or running on a lane (the job drains with the teardown, and its result is
    discarded), or waiting as a waiter.
  - **A drop whose hash is pending** resolves `Cancelled` with `Disk == NotRequested`. Neither part ran.
  - **A drop whose disk part is pending**, queued behind a write-back or a read or running, keeps its memory part's
    result. Its delete is **drained** with the subsystem's disk IO (§5.4 *Re-`Configure()` and teardown*: a delete
    is drained, never cancelled) after whatever it waits on. It is not re-run, and the op resolves with the delete's
    result. *Rev 1.18.5 (U2 code review, findings 1 and 2):* this holds on the host's real abandon-then-pool-destroy
    path. A delete that waits on a write-back's save is drained under the old configuration before the pool is
    destroyed, and is never carried into the next `Configure()` (§5.4).

  - **A cache request's lease cancelled at a re-`Configure()`** (*rev 1.18.6*, a stated limit) stays readable as
    `Cancelled`, as cell 1.7 reads it. This covers a lease whose request was queued, and the builder extends it to
    one whose build or restore was running.
    - The kept set is capped at **1,024 entries**, and the oldest is dropped first.
    - A lease dropped from the set reads `Unknown`, which callers see as `Invalid`.
    - So `Cancelled` is guaranteed only for the newest 1,024 cancelled leases, as an op's result is kept only within
      its ring of 1,024 (above).

  Drops are counted apart from evictions in the diagnostics (drops, deferred, marks applied, marks cleared; warms by
  result).
- Blueprint: `Warm Prefix`, `Drop Prefix`, `Get Prefix Op Result`.
- Left out (§6): a synchronous residency query, and cancelling a load or build in flight.

### 5.7 The disk wrapper

The wrapper for the disk tier and for cooked blobs, little-endian: magic `'S','L','U','P'`, wrapper version (1),
backend, Layer-1 tag and commit (as `'SLUW'`, S14), model artifact hash, the key, the full spec (schema name,
token count and ids, parent key), payload size, then the `SPX1` payload byte-for-byte. The plugin never interprets
the payload: format and integrity are Layer 1's (digest, §3.2). Load checks, on a pool thread: wrapper magic and
version, backend, Layer-1 tag and commit, artifact hash, key, and **spec equality**; any failure is a miss, the file
is deleted (disk tier) or the set entry is marked stale (cooked tier), and the reason is logged once. Writes go to a
temporary file and are renamed into place, so a crash leaves no partial file under a key name. Temporary names are
unique per subsystem instance and write (several instances share one directory under PIE, cell 3.8), and a read
that fails because a file vanished or is mid-rename is a miss. The directory is trimmed to
`PrefixDiskCacheMaxBytes`, oldest first, on a pool thread, at `Configure()` and after writes. *Rev 1.1:* "oldest"
is the file's modification time, which the cache sets itself on every write and every hit it serves. It never
relies on the file system's access time, which is often not updated. *Rev 1.18.5 (U2 code review, finding 7):* the
touch that sets it on a hit (`TouchDiskFile`) **joins the key's disk serialization**, beside the read, the
write-back and the delete. On Windows, UE's `SetTimeStamp` opens the file with read sharing only, not delete
sharing, so a drop's delete running beside a touch would fail as `Failed` and leave the file behind.

### 5.8 Cooking prefixes

- `USuperSLMPrefixSet` (data asset): the model asset it is for; a list of specs (name, token ids, schema name,
  parent entry name; *rev 1.18:* the input kind and text, below); and, per backend, the built wrapper bytes for each
  entry, with the Layer-1 tag they were built under.
- **Build:** an editor action on the asset and a commandlet (`-run=SuperSLMBuildPrefixes`), both in the Editor
  module (S18), build each entry by prefill against the asset's model on the editor's pinned Layer 1, save, and
  store the wrapper bytes. CPU entries build anywhere; GPU entries build only where the GPU backend configures, and
  are otherwise left empty with a warning (the runtime recomputes them).
- **Cook validation:** a set whose stored Layer-1 tag, artifact hash or entry specs no longer match is reported at
  cook as a warning naming the entries; the stale bytes ship and are refused at runtime (a recompute), never used.
- *Rev 1.17: token ids only; defining an entry by text was left out (§6).* **Entries defined by text** (*rev 1.18*,
  PC-16 (b); the §6 row is removed). An entry gains `ESuperSLMPrefixEntryInput Input` ∈ {`Tokens` (the default, and
  what every existing asset loads as), `Text`} and `FString Text`. **The token ids stay the runtime key**: text is an
  authoring input only, the runtime never tokenizes, and a text entry's key, wrapper, spec compare and request are
  exactly a token entry's with the same ids.
  - *Tokenized at build, against the asset's model.* The editor action and the commandlet tokenize each `Text` entry
    with `SuperSLM::TokenizeText(Model, Text, OutTokens, OutError)` (`SuperSLMDetokenizer.h`; *rev 1.18.1*, plan
    review F14, read at source: its body makes the same `sslm_tokenize` call with the same sizing as
    `USuperSLMSubsystem::Tokenize` (`SuperSLMDetokenizer.cpp:72-110`, `SuperSLMSubsystem.cpp:3546-3575`), and differs
    only in the mapping: the asset's own model through `AcquireModelMapping`, against the subsystem's configured
    model, so the two agree when the subsystem is configured with the set's model), with nothing prepended or
    appended, on the game thread (the registry is game-thread API), at the start of the build, before the background
    build (below) runs. A tokenize failure, an empty result, or ids over `context_cap` fail that entry by name and
    build nothing for it.
  - *What is stored.* The entry stores the token ids it built from, beside the text, the text's SHA-256 and the
    model artifact hash they were tokenized under. A nested text entry's `Text` is its **suffix**, tokenized alone;
    its ids are the entry's own suffix ids, as a nested token entry's are (so a boundary that a whole-text tokenize
    would merge is not merged). A parent and child are each `Tokens` or `Text` independently.
  - *Staleness.* Editing an entry's `Text`, or the asset's model, clears that entry's stored ids and built bytes in the
    editor. At cook, a `Text` entry whose stored text hash or model hash no longer matches is warned
    (`TextChanged`, naming the entry), as the stale-tag warning above, and the cooked package carries that entry with
    no ids and no bytes; the source asset is not modified. At runtime such an entry has no spec and is not served.
    *Rev 1.18.1 (audit G8 (a)):* for a `Text` entry a stale model hash takes this path, **not** the stale-artifact
    path of *Cook validation* above (which ships stale bytes and recomputes from the stored ids), because a `Text`
    entry's ids belong to the old tokenizer; `RegisterPrefixSet` skips such an entry. A `Tokens` entry keeps the
    stale-artifact path.
  - *How a game requests it* (*rev 1.18.3*, the re-check's m1: one signature, stated once):
    `bool USuperSLMPrefixSet::FindEntrySpec(const USuperSLMModel* ConfiguredModel, FName Entry,
    FSuperSLMPrefixSpec& Out, FName& OutParentEntry) const`. The caller passes its subsystem's
    `GetConfiguredModel()`, which both `USuperSLMSubsystem` and `USuperSLMGpuSubsystem` have, so one signature serves
    both backends. It returns the entry's stored ids and schema name in `Out` for `RequestPrefix` or `WarmPrefix`,
    and `OutParentEntry`, the parent entry's name (`NAME_None` for a root, and for every entry before U6, when nested
    specs are refused). `Out.Parent` is left invalid: the caller requests the parent entry itself and sets
    `Out.Parent` to that lease. It refuses an entry that has no stored ids, and a set whose model is not
    `ConfiguredModel` (below). *Rev 1.18.2 (lesser plane 7):* rev 1.18 said it returns "its parent's spec, at U6",
    which it cannot, since `FSuperSLMPrefixSpec.Parent` is a lease handle, not a spec.
  - *The set's model against the subsystem's* (*rev 1.18.2*, lesser plane 8; the conductor's ruling, the simpler of
    refuse and re-tokenize). A set's ids, `Text` and `Tokens` entries alike, belong to its model asset's tokenizer.
    Re-tokenizing at runtime would break *the runtime never tokenizes* above, so the cache **refuses** instead.
    `RegisterPrefixSet` refuses a set whose model's artifact hash is not the configured model's (logged once,
    `PrefixSetModelMismatch`, naming the set; nothing registered). A re-`Configure()` with another model unregisters
    such a set, and the caller registers it again under its own model. `FindEntrySpec` refuses, with the same reason,
    when its `ConfiguredModel` is not the set's model. So a mismatched set's ids never reach `RequestPrefix` through
    the set. The check is per set, not per
    entry, and covers both input kinds. Cell 2.18.
- **The build runs in the background** (*rev 1.18*, PC-15 Q9; the U3 draft's W5). The editor action's prefills ran
  synchronously on the game thread. The action now runs as a background task with a progress dialog: after the
  tokenize pass above, every prefill, freeze and save runs off the game thread, the dialog reports entries done and
  can cancel, and the set's stored bytes are replaced **only when every entry has finished**. A cancel leaves the
  previous build's bytes as they were. The commandlet is unchanged. Cell 3.11. *Rev 1.18.1 (plan review F6, audit
  G7; the conductor's ruling): failure is per entry.* An entry that fails to build, at the tokenize pass or in its
  prefill, freeze or save, **keeps its own previously stored ids and bytes** and is reported by name; every other
  entry still builds, and at the end the set stores each built entry's new bytes beside the failed entries' kept
  ones, in one replacement. A GPU entry left empty because the editor's GPU backend does not configure is reported
  as **not built**, not as failed, and keeps its previous bytes the same way. Cells 2.17 and 3.11 state this one rule.

### 5.9 Every fallback

| Cause | Detected by | Result |
|---|---|---|
| Different model, engine pin, backend, format, schema, tokens or parent | the key (§5.2) | a miss |
| Wrapper malformed, truncated, other version, spec differs (collision or tamper) | wrapper parse and spec compare (§5.7) | miss; file deleted / entry marked stale |
| Any byte of the payload corrupted | `SPX1` digest (§3.4 step 3) | `SSLM_INVALID_ARGUMENT` → miss; file deleted |
| Blob from another producer (engine version, GPU vendor or device) or model | §3.4 steps 5–6 | the named status → miss |
| Unknown `SchemaName` (*rev 1.1*) | the name lookup on a worker lane (§5.3 item 4) | resolves `UnknownSchema`; no entry, no file |
| A cache-owned handle given to `ReleasePrefix`, or an ended lease given to `AdoptPrefix` (*rev 1.1*) | §5.10 | the named refusal; nothing released or queued |
| Charge does not fit and nothing is evictable (*rev 1.2*), counting every pending refund it can claim (*rev 1.14*; *rev 1.15*: unclaimed refunds included) | the charge rule (§5.4; §5.11 (a)) | `BudgetExhausted`; nothing drawn. A request that fits only by its claims on pending refunds waits for their applies and is then admitted, never refused (§5.11 (a)) |
| Pool short | `SSLM_KV_POOL_EXHAUSTED` | evict and retry; then `PoolExhausted` (§5.4) |
| Disk read or write error | the file API | miss (read) / no write-back (write); logged |
| Nested spec before U6 (*rev 1.13*: rev 1.12 said "before PC-6's plan", which now lands first) | `RequestPrefix` | refused `NestedRequiresBudgetMode` |
| *(Rev 1.18.5; U2 code review, finding 4.)* A warm whose build fails | the build's completion (§5.6) | `Failed`, with `Out.Reason` the request's reason (`BuildFailed`); logged and counted as the request's, with `Warm` and the op id in place of the handle |
| *(Rev 1.18.)* A warm that cannot be charged, or meets a short pool or an unknown schema | `WarmPrefix`, as `RequestPrefix` (§5.6) | the request's named result; nothing drawn past the ledger. *Rev 1.18.4 (suite audit G4):* reason `BudgetExhausted`, `PoolExhausted` or `UnknownSchema`, the request's own; the log line is the request's for that reason with `Warm` and the op id in place of the handle, plus the key |
| *(Rev 1.18.)* A drop of a pinned entry, or of a key in flight or waiting | `DropPrefix` (§5.6) | `DropDeferred`; nothing released or cancelled; the mark applies when the entry is resident and unpinned, behind earlier pending hashes (*rev 1.18.2*, §5.6 *When a mark is applied*). *Rev 1.18.4 (G4):* not a fallback. No reason code and no log line; its counter is the drop diagnostics' `Deferred`, and `MarksApplied` when the mark applies |
| *(Rev 1.18.4; suite audit G4.)* A drop's disk delete that the file API refuses | `DropPrefix(MemoryAndDisk)`'s delete on the pool thread (§5.6) | `Out.Disk == Failed`; the memory part keeps its result. Reason **`DropDeleteFailed`**, logged once per op: the key, the file path and the file API's error |
| *(Rev 1.18.2.)* A prefix set whose model is not the subsystem's configured model | `RegisterPrefixSet` and `FindEntrySpec` (§5.8) | refused `PrefixSetModelMismatch`, logged once; nothing registered, no spec returned. *Rev 1.18.4:* the reason is also an `EPrefixFallback` value, with the set's name and both models' artifact hashes in its log line |
| *(Rev 1.18.)* A `Text` entry whose text or model changed since its build | cook (§5.8) | warned `TextChanged`; the entry ships with no ids and is not served |
| *(Rev 1.18.)* A GPU release skipped after a terminal device loss | the GPU subsystem's device-loss path (§5.4) | not re-issued; refunds nothing; the ledger leaves with the pool |

Every row ends in a recompute or a named refusal. None restores bytes the engine did not accept.

*Reason codes* (*rev 1.18.4*; suite audit G4, the red suite's reading 10). Each runtime row above counts under one
`EPrefixFallback` reason: one log line carrying that reason's name, and that reason's counter raised by one, the form
cell 5.4 grades. The warm rows reuse the request's reasons, because a warm is a request without a lease. Two reasons
are new: `DropDeleteFailed` (a refused drop delete) and `PrefixSetModelMismatch` (§5.8). Both are appended after the
existing reasons, before `Count`. A deferred drop is not a fallback and counts only in the drop diagnostics. The
`TextChanged` and device-loss rows keep the forms their own cells (2.17, 5.14) give them.

### 5.10 Handle dispositions against the existing prefix verbs

*Rev 1.1.* A cached prefix is an `FSuperSLMPrefix`, so every existing prefix verb can reach it. Each case has one
disposition:

| Call | Given | Disposition |
|---|---|---|
| `ReleasePrefix` | a cache-owned handle, leased or not | refused, `false`, error "owned by the prefix cache; call ReturnPrefix". Nothing is released, and the cache stays the only releaser. |
| `AdoptPrefix` | a handle whose lease has ended (returned, or its entry since evicted), even while another lease on the same entry is live | refused `PrefixLeaseEnded`; nothing queued |
| `ReturnPrefix` | a handle already returned, or one the cache does not own | `false`, one log line; nothing changes |
| `AdoptPrefix` | *(rev 1.18, PC-15 Q4)* a live lease whose phase is not yet Ready | refused `Malformed` ("not Ready"); nothing queued. This is the plugin 1.3.0 result; a dedicated not-ready result can be appended later if games need one |
| any prefix verb | a handle from before a re-`Configure()` | refused as today's verbs refuse any unknown handle ("not a live prefix handle", `SuperSLMSubsystem.cpp:2478`, `:2504`). The plugin never reuses a handle across `Configure()` (`SuperSLMSubsystem.cpp:84-87`, *rev 1.2*: rev 1.1 cited the wrong file), and cache handles come from the same counter (§5.6), so a stale handle cannot name a new entry. |
| `CreatePrefix`, `AdoptPrefix`, `ReleasePrefix` | handles `CreatePrefix` made | unchanged from 1.0.0 |

A test-only counting seam on the plugin's `sslm_prefix_release` call site records releases per engine handle.
Cell 2.10 asserts every handle is released at most once.

### 5.11 On the paging release: budget-mode adopters, region C, the save boundary

*Rev 1.13.* Paging §4 leaves this plan three cross-plan notes. They are folded here. None changes paging's text, and
only (c) asks the principal for anything (PC-13). *Rev 1.14:* (a)'s deferred refund does make paging §3.2's and §4's
"PC's ledger refunds C at eviction" stale (paging's safety argument still holds, and is stronger); a cross-plan note
tells the paging planner (`superslm-unreal-paged-kv/reviews/xplan-note-from-precache-rev1p14.md`). This section rests
on paging rev 1.7, which awaits the principal's approval of its delta since rev 1.3 (§14).

**(a) "Holds while adopters copy", in budget mode.** §5.4's argument that 1.0.0's calls stay admitted beside a full
cache rests on two things: the cache's pages never exceed its extra blocks, and adopters copy, so an evicted entry's
pages all return to the pool. Legacy mode keeps both, so the argument and cell 8.10 stand as written there. In budget
mode the argument is replaced, not stretched:
- The pool is paging's `S + P + H + A + C` pages. This plan's ledger is region C's ledger. Resident entries, ghost
  charges, and builds and restores in flight are charged to C by §5.4's charge rule, unchanged, so the cache's pages
  never exceed C.
- A budget adopter of a cache entry shares its full pages. Paging charges those `floor(L/B)` pages to region A at the
  entry's first adopt, before the adopt is enqueued, and keeps them charged until the entry's last dependent lets go,
  whether the entry is still cached or not (paging §3.2, UP-14). So when this plan evicts an entry and refunds C (once its release has applied, below), the
  pages its adopters keep are already A's. The cache may then fill C again, and the pages drawn never exceed
  `S + P + H + A + C` (paging's "charges never exceed the pool").
- 1.0.0-shaped calls in budget mode are paging's S and P requests, admitted by paging's ledger, not by PKV's legacy
  admission (§5.4 *Legacy admission*).
- **One ordering is this plan's to guarantee** (paging §13 item 15, R9). A cache charge is refunded only when the job
  that frees its pages has been applied. *Rev 1.17 (re-strike R1):* for pages shared down a family of prefixes,
  that job is the **last** of the releases of every handle that maps them (the entry's own handle and every
  descendant's), since `sslm_prefix_release` only decrements a shared page (`sslm_abi.cpp:2611-2615` at `3b6631f`);
  *Pending refunds wait on their family*, below, states the rule. Releases are not serialised: lanes still apply them
  in any order. An eviction is decided on the game thread and the entry leaves the map there
  (§5.5), but its charge stays on the ledger until the worker lane's `sslm_prefix_release` for it has run and its
  completion has been applied at a tick. A refund at the eviction decision would let a cache build admitted in the
  same tick draw a page that paging has charged to another region, and that request would then meet engine
  exhaustion, which paging counts as a ledger fault.
- **This is a new admission rule, not a reading of §5.4** (*rev 1.14*; rev 1.13 called it "§5.4's own rule read
  exactly", which overstated it). Keeping an evicted entry's charge until its release applies is what §5.4's "the
  ledger counts every page the cache holds" asks, since the cache holds those pages until the release runs. But §5.4
  as approved also said the cache "evicts … until the charge fits" and otherwise resolves `BudgetExhausted`, and under
  a deferred refund no eviction makes a charge fit in the tick it is decided. So a literal §5.4 would refuse every
  request that needs an eviction. Rev 1.14 therefore adds a **deferred-refund admission** rule, and §5.4, §5.5 and
  §5.9 are amended to it:
  - *Pending refunds are claimed.* An eviction is **pending** from its decision until its release's completion is
    applied; its charge stays on the ledger throughout. Each pending refund is split into **claimed** portions, each
    earmarked to one waiting request, and an **unclaimed** remainder. *Rev 1.15 (re-check R2):* rev 1.14 earmarked a
    pending refund whole to the request whose loop decided it and barred every other loop from counting it. That
    left two kinds of refund that no loop could count: a waiter's surplus (the part of its victims' charges beyond
    what it lacked) and the evictions of a request that was then refused. A request could then be refused where the
    approved rule admits it (§15.15's E/W/X example). So a request now covers its charge in this order, and stops as
    soon as it is covered:
    1. the free charge;
    2. **unclaimed pending refunds**, oldest eviction first, each claimed portion earmarked to this request;
    3. evicting least-recently-used unpinned entries that are not already pending, one at a time, each new pending
       refund claimed by this request up to what it still lacks, the rest left unclaimed.

    It never evicts to cover a refund that is merely late, and never evicts while unclaimed refunds would cover it.
  - *Admitted now, waiting, or refused.* If the free charge alone covers the charge, the request is admitted and
    charged at once, as before. If the free charge plus its claims covers it, the request becomes a **waiter**: it
    takes the free charge it uses onto the ledger in its own name at once, draws nothing, holds no engine handle and
    queues no job. *Rev 1.16 (strike F1):* a nested waiter **pins its spec's parent from its decision** until its job
    completes (the pin passes to the job at admission), exactly as the approved rule's queued job pins it from the
    same decision (§5.4 *Pins*). Rev 1.15 left the parent unpinned while the request waited, so another request
    could evict it and the waiter's job would later start against a released handle (caller UB, §3.7). If nothing more is evictable and the three sources still fall short, it resolves `BudgetExhausted`
    by name and draws nothing, as §5.4 says. Its claims are released back to unclaimed, and the evictions it decided
    stand, as they did under §5.4; any later request may claim their refunds.
  - *No double counting.* Every pending refund page is claimed by at most one waiter or is unclaimed, never both.
    When a pending refund **completes** (*rev 1.17*: when the last release it waits on has applied, *Pending refunds
    wait on their family*, below), each claimed portion passes to its waiter's charge, and the unclaimed remainder
    returns to free charge. The ledger's total never rises at a completion, so it never exceeds the budget. A waiter
    is admitted at the apply pass in which the last pending refund it holds a claim on completes, and it is enqueued on a worker lane in that pass. Two
    waiters completed in the same pass are enqueued in arrival order. A waiter cannot be short at that apply, because
    it stopped only when free charge plus its claims covered it, and claims are fixed amounts of refunds that will
    apply; so it never re-runs the loop and never resolves `BudgetExhausted` after waiting. Cell 8.14 asserts the
    charge is complete at admission.
  - *Coalescing.* A second request for a key whose request is waiting attaches to the waiter, as §5.4's coalescing
    attaches to any in-flight load or build. One waiter per key, so one claim.
  - *Re-`Configure()` and teardown.* A waiter holds no engine state, so it resolves by name exactly as a queued cache
    request does (cell 1.7), draws nothing, and its partial charge leaves with the ledger, which `Configure()`
    rebuilds. The releases it was waiting on are lane jobs and drain as §5.4 says. *Rev 1.16 (strike L1):* draining a
    release job is not applying its completion. Every pending record, and every completion posted for it, carries the
    configuration generation it was decided under; `Configure()` drops every pending record, and a completion whose
    generation is not the current one is **dropped at the apply pass and refunds nothing** into the rebuilt ledger,
    whose entries it never charged.
  - *Why the principal's decision is unchanged* (*rev 1.15*: rev 1.14's argument was incomplete, re-check R2). PC-11
    says a request "that cannot be charged resolves `BudgetExhausted` by name", under §5.4 as approved, where an
    evicted entry's charge was refunded at the decision. The claim is: **given the same requests and the same job
    completions, this rule admits exactly the requests the approved rule admits, refuses exactly the ones it
    refuses, and evicts the same entries; the only difference is that a waiter draws at the apply of its last
    claimed release rather than at its decision.** Proof, by induction over decisions:
    - Compare the two ledgers at a decision. Let `P` be the pending refunds, split into claimed portions `e_w` for
      each waiter `w` and the unclaimed remainder `U`; let `f_w` be the free charge `w` took. The approved ledger
      holds each waiter's whole charge `f_w + e_w` and none of `P` (refunded at decision). This rule's ledger holds
      `f_w` for each waiter and all of `P`. Everything else (resident entries, in-flight builds and restores, ghost
      charges) is the same in both, by the induction hypothesis. So this ledger exceeds the approved one by exactly
      `U`, and the approved rule's free charge is this rule's free charge plus `U`. *Rev 1.16 (strike F2):* a ghost
      that the approved rule refunds at a descendant's eviction decision is here a pending refund (*rev 1.17:* of its
      own, completing when its whole family's releases have applied, *Pending refunds wait on their family*, below),
      so it lies in some `e_w` or in `U`, and the difference is still exactly `U`. The same holds for an evicted
      entry's own charge held for its family.
    - *Rev 1.16 (strike F1):* the **pin sets** are equal. The approved rule pins a nested request's parent from its
      decision (its job is queued); here the waiter pins it from the same decision. So the evictable entries are
      the same set, in the same LRU order, in both.
    - The approved rule admits the request iff free charge plus the charges of the evictable entries covers it,
      evicting least-recently-used entries until it does. This rule covers it from free charge, then `U`, then the
      same evictable entries in the same order (entries already pending are the ones the approved rule has already
      removed). Free plus `U` is the approved free charge, so both stop after the same evictions, and both refuse
      exactly when even every evictable entry falls short. The request is admitted at once in both when free charge
      alone covers it, and otherwise is admitted by the approved rule at once and becomes a waiter here, with the
      same charge.
    - After the decision the hypothesis holds again: the same entries left the map, the same parents are pinned,
      and every page of `P` is still either claimed or in `U`.

    *Executed (rev 1.17; this fold's re-check).* The re-strike's own model and generators
    (`reviews/restrike-rev1p16-probes/restrike_model.py`, copied unchanged to `reviews/fold-rev1p17-probes/`) run
    against this directory's `fold_refund_model.py` with the rev 1.17 rules. Its page counter is refcount-accurate
    (rev 1.16's credited all of a parent's shared pages to one descendant, which hid R1), and its releases apply in
    random order. Three seeds of 20,000 runs each (405,247 requests, 154,004 nested, 77,457 waits, 20,817 coalesced
    onto waiters, 10,760 re-`Configure()`s mid-wait, 18,628 ghost endings): **0 divergences, 0 ledger excesses, 0
    physical overdraws**, and no job against an evicted parent (`restrike_model.out`). The same runs with the rev
    1.16 ghost rule give 608, 612 and 777 physical overdraws; with dependency sets on ghosts only, 23, 7 and 23
    (`fold_runs.out` E). With releases in decision order, 0 everywhere, as before.

    *Executed (rev 1.16).* The strike's model (`reviews/strike-rev1p15-probes/refund_model.py`), extended with the
    pin-on-wait and claimable-ghost rules and configuration generations
    (`reviews/fold-rev1p16-probes/refund_model.py`, `refund_model.fixed.out`), compares the two rules at every step
    of random nested sequences (roots and children of resident entries, leases and returns, release applies, job
    starts and completions, re-`Configure()`), with equal feasible completions. Three seeds of 20,000 runs each
    (297,115 requests, 125,042 nested, 51,015 waits, 11,128 ghost endings) show **no divergence**: equal decisions
    (or "now" against "wait"), equal eviction lists and equal pin sets at every step, this rule's ledger never above
    C, the cache's physical pages never above C, and no job starting against an evicted parent. Without the two
    rules, 3,410 of 20,000 runs diverge. The strike's F1 and F2 constructions now give the approved rule's
    decisions, and the U2 root runs (3,000) are unchanged. (*Rev 1.17:* that model's page counter hid R1; its
    "physical never above C" is withdrawn and replaced by the run above.)

    So no request within the approved rule's reach is refused. What does differ is lane timing: a waiter's build or
    restore starts later, so its freeze (when a build's charge drops to `ceil(L/B)`) or its failure (when its charge
    returns) is applied later than it would have been had it started at the decision. A later request can therefore
    meet a ledger that the approved rule, under a different completion history, would have shown with that drop
    already applied. That is the same difference two runs of the approved rule show when the lanes run at different
    speeds, and PC-11 never fixed lane timing: its admission was always relative to what had been applied. The bound
    (the ledger never exceeds the budget) and the cost (a named refusal, never an overdraw) are unchanged, and the
    one difference a caller sees, given equal completions, is latency: §5.5's delivery rule for any queued
    operation. So this plan raises no PC for it.
  - In legacy mode the same rule keeps 8.10's reading exact. Cell 8.14 pins it in both modes.
- **Pending refunds wait on their family** (*rev 1.17*, re-strike R1; it replaces rev 1.16's *Ghost charges under
  the deferred refund*, which tied an ending ghost to the apply of the eviction that ended it, so with releases
  applied out of order a sibling's or the ghost parent's own handle could still map its pages and the cache's handles
  could hold more than C; rev 1.15 said ghosts were "unchanged", which left their end unspecified).
  - Every pending refund carries a **family**: the set of pending releases among its entry's own handle and the
    handles of that entry's descendants, each of which maps the entry's pages. For an evicted entry's own charge it is
    its own release plus every pending release of a descendant evicted before it. A ghost (§5.4, U6) that ends at a
    descendant's eviction gets a pending refund of its own, whose family is the ghost parent's own release (if not yet
    applied) plus every pending release of its descendants, the ending one included.
  - A pending refund **completes** when the last release in its family has applied, whatever order the lanes apply
    them in. Until then it stays on the ledger and stays **claimable** under the claim rule; at completion its claims
    pass to their waiters and its remainder returns to free charge.
  - Refunding earlier would let a request draw pages a family member's handle still maps (paging R9's hazard);
    holding a ghost unclaimable would refuse a request the approved rule admits. Neither happens: the induction above
    places every such refund in some `e_w` or in `U` until it completes.
  - From U6 a nested parent with sequence dependents is charged on both C (its ghost) and A (paging's pre-charge),
    which is safe (paging §3.2). *Rev 1.18 (PC-15 Q6; the U6 draft's R36):* the A side of that is not yet built (the
    cache cores never see A, and the CPU adopt path is U2's); it is **built at U6**, where nested entries ship, and
    cell 8.14's nested-A leg reads it with an adopter on a ghost's child.

**(b) Region C's wiring (U2).**
- *Budget mode.* The CPU subsystem's pool create (paging's `sslm_kv_page_pool_create(S + P + H + A + C)`) takes C from
  `CacheRegionPages()`, which U2 fills: `ceil(PrefixCacheBudgetBytes / block bytes) × P_blk` pages, paging's figure,
  where a block is paging's P/A block of `P_blk` pages. This plan's ledger charges and refunds C through the ledger
  core's region-C seam (`ISuperSLMCacheRegionCharge`), so paging's form-1 grading reads C as it reads its own regions.
- *Legacy mode.* The extra blocks are added to the `sslm_kv_pool_create` call that paging leaves unchanged, as §5.4
  says.
- *Adopt and hint.* `AdoptPrefix`'s cache lookup (§5.6) resolves the cache handle and then goes through the
  subsystem's one adopt entry point, so paging's A pre-charge fires on every adopt of a cache entry. The cache never
  calls `sslm_seq_adopt_prefix` itself. A leased cache handle passed as a restore hint is pinned by its lease and
  pre-charged as paging §3.2 specifies. *Rev 1.18 (PC-15 Q5; the U2 draft's V9):* the draft ignores such a hint, so
  those restores run private: correct, never sharing. The hint is **wired at U2 on the box**, and cell 8.15 grades it.
- *Enum order.* Paging appends `KvPagesExhausted`, `KvModeMismatch` and `ContextCapMismatch` after `OutOfMemory`;
  `PrefixLeaseEnded` (§5.6) is appended after them, still last.
- *Paging's [cache] cells.* U2's gate runs their box legs against this plan's API: paging's 8.3, 8.4, 8.5 and the
  cache kinds of its 1.2, 1.3, 5 and 11 (paging §6, §7).

**(c) The second save boundary (PC-13).** The paging release's saves are `'SLUW'` v2 wrappers tagged with the
vendored Layer-1 version, 1.13.0, around `SSB5`/`SSB6` (CPU) or `SLM5`/`SLM6` (GPU) payloads. The CPU-cache release
re-pins to 1.14.0, and under the wrapper's rule (S14; paging §3.5 *Version protection*) it would refuse every one
`Layer1Mismatch`: a second boundary, after the paging release's own 1.11.0 → 1.13.0 one (paging R11). This plan
removes it:
- *Engine, E5.* 1.14.0 keeps 1.13.0's four sequence formats byte for byte (§3.8). Sequence blobs carry no engine
  version, so the engine accepts 1.13.0 saves already. Cells 9.9 (CPU) and 9.10 (GPU, box) show that a pinned 1.13.0
  save restores on 1.14.0, re-saves byte-identical, and continues to the 1.13.0 build's own tokens. *Rev 1.18
  (principal, 2026-10-20):* the pinned saves are written by **v1.13.1** (`d697927`), the version the paging release
  vendors; 1.13.1 keeps 1.13.0's formats (§3.8).
- *Plugin, U2.* The `'SLUW'` reader gains one compatibility entry. A plugin vendoring 1.14.0 accepts a wrapper whose
  Layer-1 tag (and commit, where the wrapper records one) is 1.13.0's, when its payload magic is `SSB5`, `SSB6`,
  `SLM5` or `SLM6`. Every other tag still resolves `Layer1Mismatch`, the 1.11.0 tag included, and the reader still
  refuses every wrapper version but its own. The entry lives in one table, and only a release whose E5 re-ran 9.9 and
  9.10 against its predecessor adds a row. Cell 9.11. *Rev 1.18 (PC-15 Q8; the principal's 2026-10-20 retarget):*
  each row lists **both** commits a wrapper of that version can carry: the release commit and the tag's peeled
  commit. The 1.13.1 row is `d697927` (release commit and peeled commit are the same; the tag object is `034a76a`),
  and it is the row the paging release's saves meet. *Rev 1.18.1 (the conductor's ruling on PC-15 Q8):* **the 1.13.0
  row is dropped.** No released plugin carried 1.13.0: the plugin's `develop` vendored `v1.13.0` only from 2026-10-15
  to 10-17 and then the 1.13.1 candidates (`002094a6`, `2efa82d3`) and `v1.13.1`, and no plugin release tag vendors
  1.13.0. A development save tagged 1.13.0 is refused `Layer1Mismatch` by name, never misread (PC-13's own stated
  alternative). The table therefore has one row, 1.13.1 (`d697927`); the U2 draft's rows for `a5f3ad1` and
  `002094a6` leave it.
- *Why it can hold, and when it would not.* It holds because 1.14.0 adds two prefix verbs and changes no sequence
  writer, reader or forward path (§3.8). If 1.14.0 had to change one of those formats, or the forward numerics an
  adopter continues under, 9.9 or 9.10 would go red. That format then leaves the entry, the release note says so,
  and its saves keep the boundary by name. A save is never misread.
- *What it does not touch.* The cache's own `SPX1` blobs keep PC-7. Disk and cooked entries are keyed by the
  Layer-1 commit, so the re-pin makes them misses that recompute once (R6).
- *At 1.16.0 (U6)* (*rev 1.18*: rev 1.17 said 1.15.0). The same question returns at the GPU cache's re-pin, and
  1.16.0 adds GPU prefix code beside the GPU save path. F4 re-runs 9.9 and 9.10 against the pinned 1.14.0 and 1.13.1
  saves (and 1.15.0's, if a plugin release vendors it); each predecessor that passes joins U6's compatibility table,
  and one that fails keeps its boundary, stated in the release note.

---

## 6. Left out, and dated deferrals

Deferred 2026-09-30. Each re-enters at the release review of 1.16.0 (E/F slices) or plugin 1.4.0 (U slices)
(*rev 1.18*: rev 1.17 said 1.15.0 and plugin 1.2.0, renumbered by PC-1).

| Item | Why not now | Re-entry gate | What doing it now would take |
|---|---|---|---|
| Automatic prefix matching (find the cached prefix from a token stream) | PKV §6 deferral stands: the game knows its prefixes and names them | a consumer ask (PKV §6) | a radix index in the plugin over the §5.2 key's inputs, 1–2 days |
| Pre-paging stopgap (cache through today's whole-block sequence save) | the principal ruled it out | none | — |
| The plugin's move onto paged KV (PC-6) | a separate consumer adoption with its own scope (scheduler, lifecycle budget, save formats) | the principal commissions that plan | 2–4 days of plugin work, underived, plus its gate path |
| Cross-backend and cross-vendor blobs (a CPU blob restored on GPU; a blob from one GPU restored on the other) | the public claim of CPU/GPU K/V identity (`docs/platform-support.md:311-315`) is backed at source only at position 0, in a run-only package (S21; PC-5) | F3's cells 4.6 pass on both GPUs; F4 offers the relaxation | drop the backend or vendor check (one line each); keying GPU entries by backend only |
| Delta blobs (a child's rows only, needing its parent at restore) | not self-contained; the principal's default is self-contained | measured disk pressure from nested sets | a parent-key field and a hard requirement on the hint |
| Adapter-shaped prefixes | prefixes are base-only (PKV §6) | PKV §6's gate | PKV §6's cost, then an adapter field in `SPX1` |
| Blob compression | K/V int8 compresses poorly and load is already ms | a measured disk or package-size ask | a codec in the wrapper |
| *(Rev 1.18.)* A synchronous residency query (is this spec resident now?) | the key is hashed off the game thread (§5.5), so an answer in the call would hash on the game thread; `GetPrefixSource` and `GetPrefixOpResult` answer at a tick | a consumer ask | an O(tokens) hash on the game thread per query, or a key handle returned by `WarmPrefix` |
| *(Rev 1.18.)* Cancelling a load or build in flight | a drop marks such a key and never cancels engine work (§5.6); a cancel needs a lane-job cancel seam and a ledger rule for a half-built prefix | a consumer ask | a cancel point between prefill chunks and a refund-on-cancel rule under §5.11 (a) |

Rejected, not deferred: storing a content hash or the parent's tokens to identify a parent (§3.5); caching by the
full token list alone without the parent key (§5.2).

---

## 7. Coverage Model

Every dimension of the coverage catalog, crossed with this plan's mechanisms: the `SPX1` writer and reader, the
two CPU verbs, the nested re-share, the GPU twins, the version header and its check, and the plugin cache (key,
tiers, budget and eviction, pins, coalescing, disk wrapper, cook, threading; *rev 1.18:* residency control and
`Text` entries, whose cells are 1.10, 2.16, 2.17, 3.10, 5.12, 7.18–7.20, 8.14's drop legs, 9.12, 10.7 and 11.7;
*rev 1.18.1:* also 3.12, 7.23 and 8.17, with 5.15 for the release-status rule and 7.22 for the GPU labels;
*rev 1.18.2:* also 2.18 and 3.13).
Each cell carries an id and its
**green step** in brackets; the test author writes every cell red first in the step's red suite (E1, F1, U1).
`B = 16` unless stated; engine cells also run at PKV's test-only `B = 4` view. *Rev 1.13 (re-anchor F2):* at 1.13.0
neither is the engine's B any more. Each is now the test-only pin's (`KvPagesTestPinPagePositions`, S30 (f)), applied
to every fixture those cells map and checked for `SSLM_OK` and the resulting B. The pin keeps the cells' shapes and
their executed figures, and it needs B to divide the cap. So no pinned cell reaches the rule B, a partial last page or
the contiguous option. Those are reached by 4.7, 8.13, 9.8, 7.12's pool-kind leg and E4's real cells, which run
**unpinned**. *Rev 1.14 (re-gate review S4, audit S1), how an unpinned cell gets the rule B:* `superslm_precache_e1`
is compiled with `PKV_PIN_GENERATED_FIXTURES=1` (`precache.cmake:32`), so `GetFixture` pins pkv_def, pkv_qk and
pkv_32k to 16 at their one cached mapping (`pkv_common.h:296-310`; pkv_odd is skipped), `SyntheticFixture` and
`LoadSyntheticShape` pin every synthetic shape to 16 (`precache_common.h:247`, `:274`), the pin is sticky (S30 (f)),
and `Fixture::B()` returns 16 for any cap divisible by 16 unless `pinned` is set or `PKV_FIXTURE_B_FROM_RULE` is
defined (`pkv_common.h:236-242`), so every test-side helper (`CapPages()`, `R()`, `PageBytes()`, `PageRunOffset`,
probe budgets) follows it. Therefore:
- each unpinned cell maps **its own model instance**, never through `GetFixture`, `SyntheticFixture` or
  `LoadSyntheticShape`;
- it feeds the test-written rule B to every test-side helper, either by building in a target compiled with
  `PKV_FIXTURE_B_FROM_RULE` and without `PKV_PIN_GENERATED_FIXTURES` (as PKV's C6 suite does,
  `tests/paged-kv/paged_kv.cmake:246-252`), or by setting the fixture's test-side `pinned` to the rule B (which sets
  only `Fixture::B()` and makes no engine call). *Rev 1.15 (re-check R1):* the define goes only on a target of its
  own, never on `superslm_precache_e1`, whole or per file: `GetFixture`'s pin leaves `pinned` at 0, so the pinned
  cells there would read the rule B, and a per-file define is an ODR violation (10.1);
- it **asserts** `sslm_kv_page_positions(model) == PageRuleB(D, cap, elem)`, computed in the test, and `fx.B()`
  equal to it, before any other check;
- *receipt:* a 9.8 written through `GetFixture("pkv_def")` goes red on that assertion (it reads 16, not 128).

Paths that key on the pool's B but run only pinned are recorded, not required (audit N4): 5.10's private-nested
admission figures at the rule B (3 against 10; 5 against 19), which are executed arithmetic only; 8.1's physical
share count; and 8.3's `SSB6` `restore_shared` against a restored prefix. Each reads B from the pool exactly as it
does under the pin. *Rev 1.1:* the red suites are
**stubbed**, not link-red: each declared verb is defined to return a value outside the status list
(`(sslm_status)0x7FFF` on the engine; a plugin stub resolves a result no cell accepts). Every cell registers and
runs to a stub-caused red before the build, and no refusal cell goes green on unbuilt code. For E1, each [E2]–[E4] cell's
fixture and complete deciding assertions are authored before E2 implementation starts, including assertions
unreachable through the stubs. The coverage auditor checks their design against T-3060 before E1 closes;
stub-caused reds establish construction only. Built-verb assertion reach and discrimination are required at each
cell's green step. *Rev 1.2:* a cell whose
only assertion is an absence would pass against a stub, so each has a positive clause: 3.7 counts the IO in flight
at `Deinitialize()` (> 0); 3.9 and 7.8 read their instrument's checked-call count (> 0); 8.9 reads cache hits (> 0). **Fixtures** for 6.x and 9.x
use a prefix whose five saturation counts are non-zero and pairwise distinct (PKV's census fixture), whose walk
state is not the schema's initial state, and, when nested, whose `origin` is non-zero, so a field-dropping mutant
cannot hide behind a default value.

**Grading rule** (PKV §7's, applied here). A verdict's operands come only from: statuses; token streams compared
with the **prefill twin** (the same prefix built by prefill in the same process, a path this plan does not change)
or the pinned references of §8; raw pool bytes read through PKV's page-index peek seam and addressed by the formula
written in the test; blob bytes and sizes checked against §3.2's formula written in the test; the PKV fill probe
(§8) for page counts. `*out_shared_pages` and pool stats are diagnostics and decide no verdict, except in the cell
whose subject they are (9.3's diagnostic half).

**1. Lifetime and reuse — applicable.**
- 1.1 [E2] 1,000 cycles of restore → adopt (both modes) → decode → release, mixed lengths: the fill probe reads the
  pool's initial free count at the end; no page leaked or double-freed.
- 1.2 [E3] A nested child restored against a live parent; release the parent, then the child, then the adopters,
  in each of the 6 orders: decode stays equal to the prefill twin throughout, and the pool ends at its initial
  count.
- 1.3 [E2] A pool reused across restore → release → restore of a different blob: the second prefix's rows (peek)
  equal its blob's; no row of the first survives into a page the second maps (write a known pattern, release, and
  confirm the page reads 0xCD before reuse, PKV's leak check). *Rev 1.1:* the pages released are the **restored**
  prefix's scatter-written ones (PKV poison-fills only `dirty` pages); mutant: the scatter does not mark its pages
  dirty → red.
- 1.4 [U2] Cache churn: 500 requests over 40 keys with a budget of 10 entries, random leases and returns, entries
  both recompute-built and restore-built. *Operand (rev 1.1):* the cache's pages read through the fill probe, as the
  pool's used pages minus the sequences' reservation measured at the start, taken with no adopter live, at every
  10th tick boundary and at the end; never the cache's own diagnostic. The reading includes the pages of builds and
  restores in flight, which the charge rule counts (*rev 1.2*). *Reading form (rev 1.6, §8's rule, D-SLM7987):*
  "never exceed the budget" is one-sided, so each tick's reading is an admission lower bound. With the worker lanes
  parked at their dequeue, between jobs, with no job executing on any lane (§8's lane-park seam; *rev 1.8*,
  D-SLM7996), nothing draws while the reading runs, and in-flight work keeps what it has drawn. Creates totalling `F_min = N − R_seq − floor(PrefixCacheBudgetBytes / page size)` must all be
  admitted, and are then released. That shows the cache holds at most the budget's pages. `N` is the pool's page
  count and `R_seq` is the sequences' reservation, measured at the start. The fixture sizes the pool so `F_min ≥ 2`
  at every reading. The end reading is exact, by the one-state form at `k ≥ 3`. Cache pages × page size never exceed the budget,
  with no exception for pinned entries; a request that cannot be charged resolves `BudgetExhausted`; pinned entries
  are never evicted; the pool ends at its initial count after `ReturnPrefix` of all and a re-`Configure()`.
  *Rev 1.18.1 (audit N2):* with CPU builds one lane job (§5.3 item 4), a lane parked at its dequeue has either not
  started a build or frozen it, so "the reading includes the pages of builds … in flight" is vacuous for CPU builds
  here (restores in flight still count). Only 8.10 (c), through the build hold, reads a drawn and unfrozen CPU build.
- 1.5 [U2] A warm subsystem (cache populated) re-`Configure()`d: every cached handle is released, the next request
  is a miss served from the next tier.
- 1.6 [U2] Disk tier warm across two subsystem lifetimes: the second lifetime's request is a `Disk` hit.
- 1.7 [U2] *(Rev 1.1.)* Re-`Configure()` while a cache restore, and separately a cache build, is queued and while it
  is running on a worker lane: the request resolves by name. *Rev 1.18.8 (§15.27):* the build leg also runs with the
  re-`Configure()` **between two chunk jobs** of a sliced build: the request reads `Cancelled`, no further chunk is
  issued, the unfrozen prefix is released exactly once, and the pool destroy returns `SSLM_OK`. Mutant: a chunk
  issued after the teardown, or the unfrozen prefix left live, turns the leg red. *Operands (rev 1.2):* the fill probe reads the
  initial count before the pool is destroyed, after every lane has stopped (*rev 1.7*: a teardown reading, taken
  with no lane left to compete, never under the lane-park seam); the pool destroy returns `SSLM_OK`, not `SSLM_POOL_HAS_LIVE_HANDLES`;
  the release-counting seam shows each handle released exactly once.
  *Rev 1.18.2 (the strike on rev 1.18.1, lesser plane 1; §5.6 *Generations and teardown*):* re-`Configure()` with
  one op held in each `Pending` state, using the hash-completion hold, the pool-thread hold and the build hold
  (§8). The states are a warm with its hash held, a warm with its disk read held, a warm with its build queued, a
  warm with its build running, a warm that is a waiter, a drop with its hash held, and a `MemoryAndDisk` drop whose
  delete is queued behind a held write-back. After `Configure()` returns and the held write-back is released, no op
  reads `Pending`. Each warm reads `Cancelled`. The held-hash drop reads `Cancelled` with `Disk == NotRequested`. The
  queued delete has run once (the exclusion counter) and its op reads `Deleted`, and the warm's discarded build left
  no entry and no file. Mutant: any one state left unresolved → that op reads `Pending` after 1,100 later ops resolve
  → red.
- 1.8 [F2] *(Rev 1.1.)* GPU lifetime: restore A, release it, restore B so that it lands in A's pages. *Rev 1.2:* B
  lands there because the pool is left with exactly `ceil(L/B)` free pages, A's; the test confirms it by page index.
  *Rev 1.6:* `L` has `ceil(L/B) ≥ 2` (at `B = 16`, `L ≥ 17`). F0 records the GPU pool's probe form and its floor (§14),
  and the cell uses that form.
  A's and B's rows differ on every page. B's adopter tokens equal B's twin, and the device rows of B's private
  pages, read back, equal the host mirror. Mutants: the upload skipped, and the upload of the last page skipped →
  each red.
- 1.9 [U6] *(Rev 1.1.)* Ghost charges: nested entries in budget mode, parent evicted while its children stay; then
  the cache admits new entries up to the budget (*rev 1.2*: without this step the mutant has nothing to overrun).
  Also three levels (*rev 1.2*): P ⊂ C ⊂ G, evict P, then C, then admit up to the budget. The fill-probe reading of
  1.4, in 1.4's form (lanes parked between jobs, admission lower bound at `F_min ≥ 2`; *rev 1.6*, *rev 1.7*), stays within the budget. Mutants: the ghost charge dropped, and the ghost cleared at the last child rather
  than the last descendant → each red.
- 1.10 [U2] *(Rev 1.18; PC-16 (a).)* 1,000 cycles of warm → request → adopt → return → drop over mixed lengths and
  keys, with drops of leased and unleased entries interleaved: at the end, after every lane has stopped (1.7's
  teardown reading), the pool is at its initial count, the release-counting seam shows no handle released twice and
  none left unreleased, and the ledger reads 0. *Mutants (rev 1.18.1, audit G18):* a drop of a leased entry releases
  its handle and the lease's return releases it again → a double release on the seam → red; a dropped charge never
  refunded → the ledger is not 0 at the end → red.

**2. Trust boundaries and hostile inputs — applicable.**
- 2.1 [E2] Truncation, then one byte over: at synthetic `L = 1` (184 + 896 + 6,144 bytes) at every offset; at
  `L = 1,000` at every header offset and every 4,096th byte after it. Each `SSLM_INVALID_ARGUMENT`, nothing drawn
  (fill probe), `*out` null.
- 2.2 [E2] *(Rev 1.1: exhaustive.)* At synthetic `L = 1`, every byte flipped in turn, the final byte and the digest
  field included: refused `SSLM_INVALID_ARGUMENT`, the magic bytes by step 2 and every other byte by the digest.
  `SSB6`, `SSB5` and `SLM*` blobs refused by magic; an `SPX1` given to `sslm_seq_restore` and
  `sslm_seq_restore_shared` refused (X9).
- 2.3 [E2] Hostile fields with the digest recomputed by the test: each rule of §3.4 steps 4–8 fires its named
  status, one case per rule: backend 1; engine major, minor and patch each differing alone; adapter vendor alone
  and device alone non-zero; other model hash; other `kv_precision`; unknown schema hash; walk ≥ `state_count`;
  non-sentinel walk unbound; `origin > L`; `origin = −1`; `L = −1`; `L > cap`, including an `L` whose
  `L · bytes_per_token` would wrap the size formula onto `n` (refused by the cap clause, *rev 1.2*);
  `current_token` −1, −2 and `INT32_MIN` with `L > 0`; `current_token ≥ vocab`; `m` at `INT32_MAX + 1` and
  `INT32_MIN − 1`; size ≠ formula.
- 2.3b [E2] *(Rev 1.1; claim (iii).)* Edited but in-domain, digest recomputed, run under ASan and UBSan: `e` at
  `INT64_MIN`, `INT64_MAX` and 0; `m` at `INT32_MIN`, `INT32_MAX` and 0; the counts at 0 and `UINT64_MAX`; residual
  and K/V bytes at −128 and 127; the walk at `state_count − 1`; `current_token` at 0 and `vocab − 1`. Each case is
  either refused with a named status, or adopted and decoded for 16 tokens ending in tokens or a named forward
  status. No sanitizer report. *Routed to the test author at E1 (rev 1.6):* the coverage audit T-3024's F6, an
  equality clause for the in-domain `m` and `e` cases against an unedited restore, source-grounded by S20, to be
  authored in E1 and run against built verbs in E2 (§9).
- 2.4 [E2] Save: null arguments, unfrozen prefix, short buffer (size written, nothing else). *Rev 1.1:* a canary
  after the buffer on success shows exactly `*n` bytes written.
- 2.5 [E2] Restore: null `out`, `model`, `pool`, `*pool`, `buf`; pool of another model.
- 2.6 [E3] Hints failing each of (b)–(f) alone, including a parent of another pool and of another model, and a
  parent whose rows differ in the last byte of the last shared page: private restore, `SSLM_OK`, and the foreign
  parent's pool unchanged (fill probe). *Rev 1.1:* matching hints come from a parent built by legacy begin, one built
  by budgeted begin, and one that was itself restored.
- 2.7 [F2] GPU twins of 2.1–2.6 with §4's statuses, plus a cross-context `model` and `pool`.
- 2.8 [U2] Disk tier: truncated wrapper, other wrapper version, other Layer-1 tag, spec mismatch under an equal key
  (a test seam forces the collision), zero-length file, a foreign file in the directory, a file replaced between
  scan and read: each a miss, the file deleted, one log line, and the request served by recompute with tokens equal
  to the prefill twin.
- 2.9 [U3] Cooked set with a stale tag, a stale artifact hash, and a spec edited after build: cook warns naming the
  entries; runtime serves each by recompute.
- 2.10 [U2] *(Rev 1.1; §5.10.)* `ReleasePrefix` on a leased and on an unleased cache handle; `AdoptPrefix` on a
  returned lease and on one whose entry was since evicted; `ReturnPrefix` twice and on a `CreatePrefix` handle; each
  verb with a handle from before a re-`Configure()`; *(rev 1.2)* lease A returned while lease B on the same entry is
  live, where A's `AdoptPrefix` is refused `PrefixLeaseEnded` and B's adopts. Each gives §5.10's disposition. After teardown the pool is at
  its initial count (fill probe, read after every lane has stopped, as 1.7's, *rev 1.7*), and the release-counting seam shows no handle released twice.
- 2.11 [U2] *(Rev 1.1.)* `RequestPrefix` with an unknown `SchemaName`: resolves `UnknownSchema`, no cache entry, no
  disk file under that key.
- 2.13 [E2, F2] *(Rev 1.3; PC-12's must-reject.)* The strike's two-name artifact: an otherwise valid model whose
  schemas are `P1z16@?^_s(Q` (index 0) and `abcdefgh` (index 1), both with at least two DFA states. It is refused at
  map on the CPU and on the GPU (F2), with reason `DuplicateSchemaNameHash`. A model with a name hashing to 0 is
  refused `ZeroSchemaNameHash`. The printable 29-byte name in D-SLM8002 hashes to zero; the independent writer
  produced its full-model artifact (T-3030 §3). The C++ parser cell may use its separate hash-substitution seam.
  The same artifact with index 1 renamed to a non-colliding name maps. Mutants: each parser check removed → red.
  *Construction (rev 1.6, D-SLM7985):* the two-name and zero-hash artifacts' `SCM1` sections are written by §8's
  independent test-only writer, never by `_serialize_scm1`. The rest of the model is assembled as the synthetic
  full-model fixtures assemble theirs. The renamed pair is serialized by the production `_serialize_scm1` and maps
  (must-accept). The same bytes serve 2.13, 2.15, 11.6 and the certified-artifact audit's must-reject.
  The converter's must-reject (*rev 1.5*: `_serialize_scm1`, §3.8): asked to serialize the two names, and
  separately the D-SLM8002 zero-hash name, it fails naming both schemas (or the one); asked to
  serialize the renamed pair, it
  succeeds. Mutant: the serializer's check removed → red. *Rev 1.6:* `_serialize_scm1` has no argument that skips
  the check. The cell asserts its signature is unchanged from `5d5ca58`'s `(named_schemas, vocab_size)`, so a
  bypass parameter turns the cell red. A bypass through the environment or a module global is not visible to the
  signature check. Refusing one is the code reviewer's job at E2.
- 2.14 [E2] *(Rev 1.3; the shipped `SSB5` path.)* With the two-name artifact refused at map, no `SSB5` blob against
  it can exist. Regression: on a three-schema model with distinct hashes, a sequence bound to the index-2 schema
  saves as `SSB5` and restores bound to index 2, with adopters' tokens equal to the unsaved sequence's.
- 2.15 [U2, U3] *(Rev 1.3; plugin tiers.)* `Configure()` with the two-name artifact fails by name; no cache tier is
  reached. The editor's prefix-set build refuses it, naming the model. On a three-schema model with distinct hashes,
  a spec bound to the index-2 schema is served from the cooked tier [U3] and the disk tier [U2]. Its adopters'
  schema dispositions and tokens equal the recompute's (6.7's operands).
- 2.12 [U2] *(Rev 1.1; key mutation.)* For each key component that can vary within one subsystem (tokens, schema
  name, and the length prefix; the parent key at [U6], since nested specs are refused before it, *rev 1.2*): request
  spec A, then spec B differing only there. *Rev 1.2:* the test author derives the length-prefix pair from §5.2's
  actual field order (schema name before tokens) and the name's encoding. Rev 1.1's example assumed the reverse
  order and does not collide. If no colliding pair can be built, the mutant is recorded as equivalent. B's source is not `Memory`, B's adopter tokens equal B's prefill twin, and the
  spec-mismatch counter reads 0. Mutant: that component dropped from the key → the counter reads 1 → red.
- 2.16 [U2] *(Rev 1.18; PC-16 (a).)* `WarmPrefix` and `DropPrefix` with each malformed spec `RequestPrefix` refuses
  (empty, out of vocabulary, over `context_cap`): refused synchronously, no op issued. `GetPrefixOpResult` on an id it
  never issued returns false; on an op from before a re-`Configure()`, `Expired`; on an op older than the last 1,024
  resolved, `Expired`. `DropPrefix` on a spec that differs from a resident entry's in one key component only (2.12's
  components) is `NotCached` and the resident entry stays a `Memory` hit. *Rev 1.18.1 (audit G11, G12, G18; plan
  review F16):* "from before a re-`Configure()`, `Expired`" is withdrawn (§5.6): a resolved op from before the
  re-`Configure()` still reads its own result, and a warm cancelled by it reads `Cancelled`. The ring boundary is a
  pair: the 1,024th most recent resolved op reads its result and the 1,025th reads `Expired`. An op still `Pending`
  when 1,100 later ops resolve still reads `Pending` (it never expires). Mutants: a ring of 1,023 → the 1,024th reads
  `Expired` → red; a drop matched by token ids only, ignoring the schema → the resident entry is evicted → red on its
  `Memory` hit; a warm that accepts a malformed spec → an op is issued → red.
- 2.17 [U3] *(Rev 1.18; PC-16 (b), §5.8.)* `Text` entries at the edges: empty text, text whose tokenization exceeds
  `context_cap`, and a model with no mapped artifact each fail that entry by name and build nothing for it, while the
  set's other entries build; an edit to an entry's `Text` and a change of the asset's model each clear that entry's
  ids and bytes in the editor; at cook, an entry whose stored text hash no longer matches is warned `TextChanged`, the
  cooked package carries it with no ids and no bytes, the source asset is byte-unchanged, and at runtime
  `FindEntrySpec` refuses it. A nested `Text` entry's ids are its suffix's tokenization alone.
  *Rev 1.18.1 (audit G7, G8, N4; plan review F6):*
  - **Per-entry failure** (§5.8): the failing entries keep their own previously stored ids and bytes and are
    reported by name; the other entries build and are stored. In this cell's fixture each failing entry was built
    once before with known bytes, which must be byte-equal after.
  - "A model with no mapped artifact" is replaced by an artifact with **no tokenizer section** (`TokenizeText`'s
    `SSLM_ARTIFACT_REJECTED` branch), which leaves the set's `Tokens` entries buildable.
  - The cook-time text-hash staleness is built by editing the stored text hash in the asset outside the editor's
    change hook (which would clear the ids). The cook-time **model-hash** staleness: a set with one `Text` and one
    `Tokens` entry, its model hash made stale the same way, cooked: the `Text` entry is `TextChanged`, no ids, no
    bytes, `FindEntrySpec` refuses it and `RegisterPrefixSet` skips it; the `Tokens` entry follows 2.9. Mutant: the
    `Text` entry takes 2.9's path → served by recompute from stale ids → red.
  - The nested fixture's parent and suffix texts satisfy `tokenize(parent ‖ suffix) ≠ tokenize(parent) ‖
    tokenize(suffix)`, stated and asserted in the test, so the mutant "tokenize the whole text, then split" is red.
  - [U6] `FindEntrySpec` on a nested entry returns its parent's spec, equal to `FindEntrySpec` of the parent entry.
    *Rev 1.18.2 (the strike on rev 1.18.1, lesser plane 7; §5.8):* replaced. `FindEntrySpec` on a nested entry gives
    `OutParentEntry` equal to the stored parent entry name and `Out.Parent` invalid. On a root entry it gives
    `NAME_None`. A request for the parent entry, then the child's spec with `Out.Parent` set to that lease, is served
    with tokens equal to the twin.
- 2.18 [U3] *(Rev 1.18.2; the strike on rev 1.18.1, lesser plane 8; §5.8 *The set's model against the
  subsystem's*.)* A set built for model A holds one `Text` entry and one `Tokens` entry, and the subsystem is
  configured with model B (another artifact hash). `RegisterPrefixSet` refuses the set: `PrefixSetModelMismatch` is
  logged once, and a request for either entry's ids gets no `Cooked` source. `FindEntrySpec(Subsystem->GetConfiguredModel(), …)` refuses
  both entries with the same reason. After a re-`Configure()` with model A and a fresh registration, both entries are
  found and served `Cooked`. A set registered under A is unregistered by a re-`Configure()` with B. Mutants: no model
  check in `FindEntrySpec` → A's ids reach a request under B → red on the refusal; the check on `Text` entries only →
  the `Tokens` entry is returned → red; the set kept registered across the re-`Configure()` to B → red on the
  registration count.

**3. Concurrency — applicable** (TSan on Linux in the cloud; the MSVC suite on the box).
- 3.1 [E2] 8 threads: concurrent saves and adopts of one frozen prefix while 8 more threads restore other blobs
  into the same pool; every saved blob byte-equal; every restore keeps; fill probe initial after release.
- 3.2 [E3] Restores using one parent as hint on 8 threads while other threads adopt and `begin_from` the parent:
  every child keeps and shares `floor(origin/B)`.
- 3.3 [E3] A parent released on one thread after its child's restore returned on another (legal, §3.7): the child
  decodes equal to the twin.
- 3.4 [F2] GPU restore issued while a sequence is Submitted returns `SSLM_BUSY` with nothing drawn; after
  `sslm_gpu_ready` it succeeds.
- 3.5 [U2] Coalescing: 16 requests for one key in one tick → one build or load (counted), 16 Ready handles.
  *Rev 1.1:* also when the shared load fails (a corrupt disk file, then recompute) and when it ends `PoolExhausted`,
  and when the first requester returns its lease before Ready. All 16 resolve by name within a bounded number of
  ticks.
- 3.6 [U2] Eviction racing a queued adopt of the LRU entry: the entry is pinned; the adopt succeeds; the eviction
  takes the next entry.
- 3.7 [U2] `Deinitialize()` with disk reads and writes in flight (counted > 0 at the call, *rev 1.2*): no crash, no leaked handle, no file under a key
  name left partial (the temporary is removed or abandoned by name).
- 3.8 [U2] *(Rev 1.1.)* Two subsystem instances in one process (PIE) sharing the default directory write the same
  key and trim at once: one valid file per key, no partial file, and a read that races a rename is a miss, never a
  bad parse.
- 3.9 [U2] *(Rev 1.1; §5.5 ownership.)* An instrumented build asserts each mutation of cache state runs on the game
  thread. *Rev 1.2:* the assert sits in the one accessor through which the cache's private state is reachable, so an
  unwrapped write does not compile; the cell also reads the accessor's call count (> 0). Mutant: eviction's map
  update moved to the pool thread → red. *Rev 1.18:* the drop's decision and mark go through the same accessor, and
  its count includes them.
- 3.10 [U2] *(Rev 1.18; PC-16 (a), §5.6's disk part.)* With the disk tier on: `DropPrefix(MemoryAndDisk)` issued
  while the key's write-back is running (a pool-thread seam holds it mid-write): the write-back completes, then the
  delete; after the op resolves no file exists under the key and no temporary is left. A drop issued while the key's
  build is in flight: the build completes and serves, and no file is written for it. A drop racing a trim (both held
  by seams, released in each order): one runs at a time (an exclusion counter never reads 2), and the directory ends
  without the key's file and within its limit. A request admitted after the drop writes back normally. Mutant: the
  delete not serialized behind the write-back → the file exists after the op resolves → red.
  *Rev 1.18.1 (plan review F9; audit G11, G12):* (i) a request issued after the drop that **attaches** to the build
  in flight is served, and that build's write-back is restored: the file exists after it completes, written after the
  delete; mutant: the attach leaves write-back suppressed → no file → red. (ii) While the delete is held, the op reads
  `Pending`; after release, `Disk == Deleted`. (iii) *(Rev 1.18.5, U2 code review findings 1 and 2: run through the
  host's abandon-then-pool-destroy order, the product's path, as well as the core's `BeginTeardown()`. There are
  three runs, each with the delete waiting on a write-back's save that no lane has yet run, and a file already under
  the key: a re-`Configure()` with the disk tier turned off, one with another directory, and `Deinitialize()`. In
  each, the file is gone after the call and the op reads `Deleted`. Mutant: the delete carried into the next
  `Configure()` → `DiskTierOff` or `NoFile`, with the file still there, or at `Deinitialize()` never run → red. This
  is the review's probes P2a and P2b as a cell.)* A delete held at a re-`Configure()`, then released: the file is
  gone, the exclusion counter shows one delete ran, and the op reads `Deleted` after (§5.6); mutant: the delete
  re-run after `Configure()` → two deletes counted → red.
  *Rev 1.18.2 (the strike on rev 1.18.1, Fracture 2 and lesser plane 2; §5.6's disk part).* With the disk tier on,
  each leg is a request issued before a `DropPrefix(MemoryAndDisk)` of its key and held in one state at the drop's
  apply pass. After the drop's op resolves, the request is served, then every hold is released and the cache
  settles:
  - (iv) the request is a **waiter** under deferred-refund admission (§5.11 (a); it draws nothing and no job is
    queued). It is admitted and served, and no file exists under the key. This is the strike's keeper, its state s3.
  - (v) the request's build has applied and its **save** is queued on a worker lane that the lane-park seam holds
    (§8), with no file write issued yet. The save, then its file write, then the delete run in that order on the
    key's serialization, and no file exists.
  - (vi) the request's build was refused `SSLM_KV_POOL_EXHAUSTED` and is **waiting to retry** (§5.4). The pool is
    sized short of the build's draw, with one evictable entry, and that entry's release is held behind the lane-park
    seam. The build retries and serves, and no file exists.
  - (vii) a **disk read** of the key is held mid-read (a request that will hit the disk tier). The read completes and
    serves, then the delete runs: `Disk == Deleted`, never `Failed`, and the per-key overlap counter over the key's
    read, write-back and delete (§8) never reads 2. *Rev 1.18.5 (U2 code review, finding 7; §5.7):* the same with a
    disk hit's **touch** held in place of the read. The touch completes, then the delete runs, `Disk == Deleted`,
    and the counter, now over the touch too, never reads 2. Mutant: the touch left off the key's serialization →
    the counter reads 2, or on the box `Disk == Failed` with the file left → red.

  Mutants, each red on its own leg only, since each state is its own code path (the waiter list, the save lane, the
  retry list, the read): the waiter's write-back not suppressed → a file → red (iv); write-back read as the file
  write only → the save's later file write lands after the delete → a file → red (v); a retry wait not counted in
  flight → a file → red (vi); the delete not serialized with the read → the per-key counter reads 2, or on the box
  `Disk == Failed` → red (vii). A request that attaches after the drop to the waiter of (iv) or the retry of (vi)
  restores write-back as (i) does → the file exists, written after the delete.
- 3.11 [U3] *(Rev 1.18; PC-15 Q9, §5.8; the U3 draft's W5.)* The editor's Build Prefixes action on a set of 8
  entries returns to the editor's game thread before any prefill runs: a test-only counter of prefills on the game
  thread reads 0, and the progress task's entries-done count reaches 8. A cancel after 3 entries leaves the set's
  stored bytes byte-equal to the previous build's; a build in which one entry fails leaves them likewise; a complete
  build replaces them all at once. The commandlet's output is byte-equal to the action's. Mutant: the action's
  prefill on the game thread → the counter reads > 0 → red; the set replaced entry by entry → the cancel leg's bytes
  differ → red. *Rev 1.18.1 (plan review F6, audit G7, N4):* "a build in which one entry fails leaves them likewise"
  is replaced by §5.8's per-entry rule: the failing entry keeps its own previous bytes and is reported, and the other
  seven entries' bytes are the new build's, all stored at the end in one replacement. A GPU entry on an editor whose
  GPU backend does not configure is reported **not built** (not failed) and keeps its previous bytes, while the CPU
  entries are stored. The game-thread counter counts prefill, freeze and save calls, not prefill only. Mutants: a
  failing entry that clears its own bytes → red; a failing entry that blocks the others' store → red; a GPU-empty
  entry counted as failed → red on the report.
- 3.12 [U2] *(Rev 1.18.1; plan review F8, audit G10; §5.6 *What "later" means*.)* A seam holds each key's hash
  completion so the test chooses the completion order. For `DropPrefix(E)` issued before `RequestPrefix(E)` (E
  resident, unleased) in one frame, with the request's hash completing first and, in a fresh run, second: in both
  orders the drop is decided first (`Dropped`) and the request is then a miss that rebuilds, or, with E leased by an
  earlier lease so the drop marks, the request clears the mark and E stays resident after every return. For
  `RequestPrefix(E)` issued before `DropPrefix(E)`: in both completion orders the drop finds the request's lease,
  marks, and E is evicted when that lease returns. The same two legs with `WarmPrefix` for the request. Mutant: apply
  in hash-completion order → one completion order gives the other outcome → red.
- 3.13 [U2; T4's leg U6] *(Rev 1.18.2; the strike on rev 1.18.1, Fracture 1; §5.6 *When a mark is applied*.)* The
  strike's scenario for each of the six triggers. E is resident and leased, and `DropPrefix(E)` resolves
  `DropDeferred`. Then `RequestPrefix(E)` is issued **before** the pass at which E's last pin goes, with its hash held
  by the hash-completion hold past that pass. The triggers are:
  - T1, the last lease returned in the same frame;
  - T2, a queued adopt drained;
  - T3, a write-back's save drained (disk tier on);
  - T4, a child job settled [U6];
  - T5, a warm's build completing (E not yet resident at the drop);
  - T6, a request's build completing after its only lease was returned while `Pending`.

  For each trigger, a fresh run also releases the hash before that pass. In all 12 runs the request's source is
  `Memory`, or attached to T5's and T6's build, with no further prefill (`sslm_test_only_prefix_prefill_call_count`),
  and E is resident after the pass with the marks-cleared count up by one. A second leg, run for T1 and T5, issues the
  request **after** that pass with its hash held: E is dropped once the earlier hashes resolve (marks-applied up by
  one), the request recomputes, and the outcome is the same with the hash released. A third leg (T1): after the
  return, one request a frame for a fresh key, each hash held one frame. E's mark applies by the second frame.
  Mutants: marks applied as their own stage, before or after the ops, ignoring later pending hashes (rev 1.18.1's
  rule) → every held-hash run recomputes → red; one trigger's path applies the mark at once, outside step 2 → red on
  that trigger's run only, which is why all six are cells (`fold-rev1p18p2-probes/mark_order_model.py` section 2);
  the wait not bounded by the pass → the third leg's mark never applies within its watchdog → red.

**4. Shape and platform matrices — applicable.**
- 4.1 [E2] `L` ∈ {0, 1, B−1, B, B+1, 1,000, cap−1, cap} at both geometries (synthetic 0.5B and 1.5B shapes); each
  restores, saves byte-equal, and adopters match the twin. `L = 0` (empty prefix) adopts as today.
- 4.2 [E3] `origin` ∈ {0, 1, B−1, B, mid-page, page-aligned, `L`} (the last: `begin_from` then freeze with no
  prefill).
- 4.3 [E4] Platforms: MSVC on the box and gcc/clang in the cloud; the same blob built on each is byte-identical
  (the spike's inferred cross-CPU validity, pinned: blob SHA-256 equal across the three builds).
- 4.4 [F3] GPU cells on both certified GPUs.
- 4.5 [U4] The plugin on Windows x64 (the plugin's only built platform, per its changelog).
- 4.6 [F3] *(Rev 1.1; byte identity, PC-5.)* *Rev 1.2:* on the real 0.5B and 1.5B artifacts at `L = 1,000`,
  comparing every layer, head and position, because the existing evidence is one position (S21). (a) The same
  prefix built by prefill on the CPU and on each GPU: the
  `SPX1` rows and state fields are byte-equal (the header's backend, producer and digest excepted). (b) The same
  prefix on the RTX 2080 SUPER and on the RX 7900 XTX: blobs byte-equal past the producer and digest. (c) A blob
  saved on one GPU and restored on the other, with the producer check skipped by a test seam: adopter tokens equal
  the target's prefill twin. A difference is a finding, not a failed step: PC-5's refusal stays, and the difference
  goes to the principal as evidence against `docs/platform-support.md:311-315`.
- 4.7 [E2, E3] *(Rev 1.13; re-anchor F2/F4, the coverage gap.)* **The rule B and a partial last page, unpinned.**
  Models at their rule B (S30): pkv_def (`B = 128`), pkv_odd (`B = 256`, cap 4,100, 17 pages, the last holding 4
  positions), pkv_32k (`B = 256`), pkv_qk (`B = 64`), and the synthetic 0.5B and 1.5B shapes (`B = 64`). The test
  writes the rule and computes each B from the fixture's `D`, cap and element width, and requires
  `sslm_kv_page_positions` to equal it. *Rev 1.14:* each model is its own mapping, and the rule B feeds every
  test-side helper (§7 preamble). `L` ∈ {0, 1, B−1, B, B+1, `(cap_pages − 1)·B`, `(cap_pages − 1)·B + 1`,
  cap−1, cap}; on pkv_odd (4,096, 4,097, 4,099, 4,100) the last three reach the partial page. For each model and `L`, in a page pool sized so every
  reading is at least 2 free pages:
  - `SPX1` save → restore → save gives byte-identical blobs (6.4's operand);
  - the restore draws exactly `ceil(L/B)` pages (two-sided fill probe, before and after; S29's `cap_pages ≥ 3`
    holds for every model here);
  - the restored rows equal the blob's, by peek, addressed in the test with 1.13.0's run-slot rotation
    (`pkv::PageRunOffset`);
  - adopters of each mode decode `min(64, cap − L)` greedy tokens equal to adopters of the prefill twin built in the
    same pool; at `L = cap` (and `L = 0`) there is no continuation and the `SPX1` checks stand alone, as 4.1's code
    has it (`std::min(16, cap − L)`, `tests/precache/e2_root.cpp:26`, `:72` at `3b6631f`) (*rev 1.14*, audit N3).
  - *Nested leg* [E3], at the rule B: origin ∈ {B−1, B, B+1}, and on pkv_odd a parent ending in the last full page with
    the child running to `L = cap`; with a matching hint `s = floor(origin/B)`, the rows `[s·B, L)` come from the
    blob, and the probe reads `ceil(L/B) − s` drawn.

  Mutants: the restore's draw computed as `floor(L/B)` → red at `L = B + 1` and at every partial-page `L`; the table
  bound computed as `cap / B`, truncating (a premise mutant in PKV's `MapPagesFor`, run to show the exemption in
  §3.4 step 3 rests on a fact) → pkv_odd's partial-page cases refused `SSLM_INVALID_ARGUMENT` → red; a scatter that
  ignores the run-slot rotation → red at every `L > B`. This is the scratch probe of the re-anchor record (§3,
  1,448 checks) made a cell, with the nested leg and the mutants added.

**5. Failure and rejection paths — applicable.**
- 5.1 [E2] Pool one page short of `ceil(L/B) − s`: `SSLM_KV_POOL_EXHAUSTED`, nothing drawn, no refcount changed
  (fill probe before and after; for the nested case, the parent's pages still free exactly when the parent is
  released). *Shapes (rev 1.6; the probe reads only `k ≥ 2`, §8):* root `L = 4B + 1`, `s = 0`, so
  `ceil(L/B) = 5` and the pool has 4 free at `B = 16` (`L = 65`) and at the `B = 4` view (`L = 17`). Nested: the
  PKV cohort shape (origin 1,000, `L` 1,200, `B = 16`) with a matching hint, `s = 62`, 13 to draw, so the pool has
  12 free. Both are executed arithmetic, and both readings sit at least 2 above the probe's floor. The nested case
  is [E3], since it needs the hint.
- 5.2 [E2] Allocation failure after the draw, using the **restore-site fault slot** at its allocation position (§8, *rev 1.2*; named *rev 1.6*), never the general
  slot, which fires inside SHA-256 at step 3, before any draw: `SSLM_ALLOCATION_FAILED`, and the slot's recorded
  free-page count is below the initial count (the draw happened). Every page and shared refcount is returned (fill
  probe). Mutants: the drawn pages not returned → red; for a nested restore, the shared refcounts not returned →
  red (the parent's pages stay held after the parent is released); *(rev 1.7, T-3027 F13)* the failure path
  returning only the reserve (`HolderReturnReserve`) rather than releasing the holder → red, because the slot
  fires after the map, when the drawn pages are mapped and the reserve is empty (§8 must-reject (4)): the root case
  at `L = 4B + 1` reads 5 pages short. *Rev 1.6:* the nested case is [E3]. It uses a
  matching hint at `s ≥ 1` (the cohort shape, `s = 62`), and the slot's recorded shared-entry count must equal `s`.
  This is what makes the shared-refcount mutant meaningful: a slot that fired before the share would record 0, and
  nothing would have been shared for the mutant to keep (§8's second must-reject).
- 5.3 [F2] GPU device-status paths of §4 item 3, step 12, at the upload, using PKV's fault seams: pages returned, the named
  status.
- 5.4 [U2] Each row of §5.9 exercised: the named result or recompute, the log line's content asserted, and the
  diagnostic counter for that reason raised by one.
- 5.10 [E3, F2] *(Rev 1.4; the re-strike's construction.)* `B = 16`, cap 4,096. A frozen parent of length 1,000 is
  resident (63 pages), and the pool has exactly `R(200) = 14` more pages free. Control: `begin_from(parent, 200)`,
  prefill to 1,200 and freeze are admitted; the child is saved and released. Then restore the blob with each
  private-hint disposition: null, unfrozen, another model, another pool, `context_length ≠ origin`, and rows
  differing in the shared span. Each → `SSLM_KV_POOL_EXHAUSTED`, nothing drawn and no refcount changed (fill probe
  before and after). Must-accept control: the matching hint shares 62, draws 13, and is admitted. GPU twin [F2].
  Mutant: a private restore that draws only `ceil(L/B) − floor(origin/B)` pages → admitted where the cell expects
  the refusal → red. *Erratum (§15.30, P3):* 5.10 does not kill that mutant. The underdrawn holder is refused at the
  map with the same `SSLM_KV_POOL_EXHAUSTED` and releases everything, so 5.10 stays green. **2.6** kills it (also
  6.3, 7.5 and 9.3), which meets T-3060 F8. 5.10 keeps its refusals and control.
- 8.12 [U6] *(Rev 1.4.)* The cache's nested restore under a hint that fails (a plugin test seam passes a null hint):
  the charge taken before the draw is `ceil(L/B)`, the ledger never exceeds the budget (fill probe, as 1.4), and a
  charge that cannot fit resolves `BudgetExhausted` by name. Mutant: charging `ceil(L/B) − floor(origin/B)` before
  the draw → the fill-probe reading exceeds the budget → red.
- 5.5 [U2] *(Rev 1.2.)* Every entry pinned and one more request: `BudgetExhausted`, no wait, nothing evicted,
  nothing drawn (fill probe). `PoolExhausted` is reached by a test that fills the pool from outside the cache.
  *Shape (rev 1.7, T-3027 F17):* the pool carries 8.10's spare block and no 1.0.0 prefix is live, so with the
  cache at its budget both one-state readings see at least `cap_pages ≥ 3` free pages, meeting the form's
  `k ≥ 3` floor. Both readings are taken with the lanes parked behind the barrier (§8).
- 5.6 [E2] *(Rev 1.1; never drew, not drew-and-returned. Rev 1.6 rewrites the population and the pool, D-SLM7986.)*
  **Population:** every restore refusal case of 2.1, 2.2, 2.3 and 2.5, run twice more. 2.4's save refusals are
  excluded because a save has no pool to draw from, and 2.5's cases with a null pool are excluded because no
  mutant can draw without one.
  - **Against a pool with exactly 0 free pages**, each case must give its validation status and never
    `SSLM_KV_POOL_EXHAUSTED`. The pool is fresh, with a known page count `N = block_count · cap_pages`. Budgeted
    creates totalling `N` are admitted by PKV's `AdmitPages`, and every one is admitted. Since each create of
    budget `(r − 1)·B` draws exactly `r` pages, that leaves **exactly 0** free, shown by admission alone, never
    read from the probe (S29). A 2-page create refused `SSLM_KV_POOL_EXHAUSTED` confirms it.
  - At exactly 0 free, every blob with `L ≥ 1` needs at least one page. So the `L = 1` blobs of 2.1 and 2.2 now
    discriminate, and are kept: a draw-first mutant is refused for capacity on them. Rev 1.5's 2-page blob rule
    and its "at most one page free" pool are withdrawn.
  - 2.3's cases use the synthetic default `L = 1` unless the case's subject is `L`. The `L > cap` case still
    discriminates, because its draw exceeds every pool. Cases where no `L ≥ 1` can be read do not discriminate the
    draw-first mutant, and are recorded so (*rev 1.7*, T-3026 N-2): the `L = −1` case (a mutant drawing for a
    negative `L` draws no page); 2.1's truncations shorter than 120 bytes, which end before `L`'s field (offset
    112, 8 bytes, §3.2); and 2.5's null `buf`. The draw-first mutant is killed by every other case.
  - *(Rev 1.7, T-3026 N-2, T-3027 N8.)* 2.5's "pool of another model" case passes the other model's pool, so that
    pool is the one held at exactly 0 free, built as above on a second fixture model.
  - **With the restore-site fault slot armed at its allocation position** (§8; *rev 1.7*, D-SLM7993), each case
    gives its validation status and the slot reports it **did not fire**; the test then disarms it. The mutant this
    kills is a handle allocated before validation, which no draw precedes and the 0-free half cannot see. The share
    and map positions are not armed here: each needs a draw before validation, which the 0-free half already
    kills, and at E2's root blobs (`s = 0`) the share position is never consulted. The share position's statuses
    also include `SSLM_INVALID_ARGUMENT`, the validation status of most of the population, so "never the slot's
    status" there would fail a correct build (T-3026 W-3, T-3027 N7).

  Mutants:
  - draw placed directly after step 2 (reading `L` from the unverified header) → 2.2's flips past the magic red,
    every flip that leaves `L ≥ 1`;
  - draw placed after step 3 → every 2.3 case with `L ≥ 1` red;
  - draw first, pages returned on refusal → every case red.

  Precondition: `cap_pages ≥ 3` (S29). Each fixture states its cap. At 5.10's cap of 4,096, `cap_pages` is 256 at
  `B = 16` and 1,024 at the `B = 4` view.
- 5.7 [E2, F2] *(Rev 1.1; precedence; rev 1.2 completed.)* Two faults at once. **Rule:** where consecutive steps
  share a status, pair **each** of them with the nearest step on either side that gives a different status;
  adjacent pairs alone pin nothing inside a run of equal statuses.
  - CPU:
    - each of steps 1, 2, 3 and 4 + a producer mismatch → `SSLM_INVALID_ARGUMENT`. At CPU-only E2, step 4 uses
      an `SPX1` made from a known-good E2 CPU save: preserve its model hash, precision, schema, state, length,
      origin and rows; set backend to 1 and vendor/device to fixed non-zero values; recompute the digest. The
      unedited CPU save must restore, and an independent header parser must verify those preserved fields,
      the GPU-shaped backend/adapter fields, exact size and digest before the status is read. With the current
      engine version, this operand must refuse at step 4. An additional backend-only control changes only
      backend from the good CPU save, leaves CPU adapter IDs at zero and recomputes the digest; it too must
      refuse at step 4. Edit only the GPU-shaped operand's version, recompute the digest, and
      require the same status for the step-4 + step-5 pair. Swapping steps 4 and 5 must turn that pair red.
      This constructed operand proves CPU validation order; F2 repeats the pair with a real GPU save whose
      unedited model, precision, schema, domains, size and digest are checked against its GPU producer fixture;
    - producer + model → `SSLM_RESTORE_PRODUCER_MISMATCH`;
    - model + precision → `_MODEL_MISMATCH`;
    - precision + schema → `_KV_MISMATCH`;
    - schema + domain → `_SCHEMA_MISMATCH`;
    - domain + a pool with exactly 0 free pages (built as 5.6, *rev 1.6*), with a blob of `L ≥ 1` → `SSLM_INVALID_ARGUMENT`.
  - GPU [F2]:
    - each of steps 3, 4 and 5 while a sequence is Submitted → `SSLM_BUSY`;
    - a cross-context `pool` while a sequence is Submitted → `SSLM_SEQUENCE_KV_BUFFER_MISMATCH`;
    - each of steps 3, 4 and 5 + a producer mismatch → `SSLM_SEQUENCE_KV_BUFFER_MISMATCH`;
    - each of steps 8, 9 and 10 + a model mismatch → `SSLM_RESTORE_MODEL_MISMATCH`;
    - step 10 + an exhausted GPU pool → `SSLM_SEQUENCE_KV_BUFFER_MISMATCH`.
  - Mutant: steps 4 and 5 swapped → 5.7 and 8.7 red.
- 5.11 [E2, E3, F2] *(Rev 1.6; the refusals after a draw, D-SLM7984.)* The restore-site fault slot (§8) armed at
  each post-draw position, each on a fresh fixture, with the fill probe (two-sided, `k ≥ 2`) reading the pool's
  free count before and after.
  - *Share* [E3]: a nested blob with a matching hint at `s ≥ 1` (cohort shape). Armed once to refuse as
    `kInvalidArgument`, once as `kRefcountUnderflow`. Statuses `SSLM_INVALID_ARGUMENT` and
    `SSLM_ARTIFACT_REJECTED`; `*out` null; free count after equals before. The parent's page refcounts are
    unchanged: once the parent is released, its pages return to the pool, which the probe reads as the initial
    count.
  - *Map* [E2]: a root blob, `L = 4B + 1`. The slot empties the reserve before step 3. `SSLM_KV_POOL_EXHAUSTED`,
    `*out` null, and the free count after equals before. A nested variant [E3] also returns the shared refcounts,
    as in the share case.
  - *(Rev 1.7, D-SLM7992; T-3026 W-2, T-3027 F12.)* Every case also asserts, once every handle the case made is
    released, that `sslm_kv_pool_destroy` returns `SSLM_OK`, not `SSLM_POOL_HAS_LIVE_HANDLES` (1.7's operand). The
    reason is the root map case: the map position returns the **whole** reserve before step 3
    (`HolderReturnReserve`, `kv_pages.cpp:379-411`), so on a root blob the holder then holds no page, and a leaked
    holder leaks no page for the probe to see. It still holds one `live_refs` count, which only `ReleaseHolder`
    drops (`sslm_abi.cpp:1622-1627`; `CreateHolder` takes it at `:1616`), and the destroy refuses on it
    (`:1226-1231`). The precedent this position
    copies, `sslm_pkv_test_only_drain_reserve`, grades a status and a missing row, never a release, so the
    release is graded here on its own operand rather than inherited from that sibling.

  Mutants:
  - the holder not released on either refusal → the pool destroy is refused in every case, and the probe reads
    short in the share case and the nested map case → red (the root map case is red on the destroy alone);
  - the refusal path returning only the reserve (`HolderReturnReserve`) rather than releasing the holder → the pool
    destroy is refused → red;
  - the share's or the map's status ignored and the restore continued → `SSLM_OK` or a crash where a refusal is
    expected → red;
  - the share refusal mapped to another status → red.

  The pool-status leg is pinned too: the map refusal must read as `SSLM_KV_POOL_EXHAUSTED`. The plugin reads that
  status as a capacity miss (§5.9), which is harmless here, because the case is unreachable outside the seam.
  - *GPU half* [F2], **pending F0** (§4 item 2): if F0 finds a GPU share that can refuse after the draw, the GPU
    twin of the share case, with F0's status, every drawn page returned and the debug layer clean, plus a 5.7
    GPU pair (that refusal against its nearest differing neighbour). If F0 finds none, the half is recorded not
    applicable with F0's citation. *Rev 1.18 (PC-15 Q3):* settled, no longer pending. The GPU share can refuse after
    the draw, and both page-module refusals read `SSLM_SEQUENCE_KV_BUFFER_MISMATCH` (§4 item 2, item 3 step 11a): the
    GPU twin armed once as `kInvalidArgument` and once as `kRefcountUnderflow`, each with that one status, every
    drawn page returned (the GPU pool's fill reading, R7's two-check form) and the debug layer clean, plus the 5.7
    GPU pair. Mutant: either refusal mapped to another status → red.
- 5.8 [U2] *(Rev 1.1.)* A fault seam between the disk write and the rename: no file under the key name; the next
  request recomputes.
- 5.9 [U4] *(Rev 1.1.)* A restore delayed on its lane by a test seam past its tick's slice raises the hitch
  counter by one and still delivers. *Rev 1.18 (Q7):* also a CPU cache build of `L = 2,000` on the real 0.5B
  artifact, now one lane job: its hitch count is recorded; a count above zero is the trigger for slicing (§5.3 item 4),
  not a red. *Rev 1.18.8 (§15.27): the trigger tripped at U4, and the build is sliced. The leg is re-graded:*
  - (a) **Sliced.** The build of `L = 2,000` runs as more than one lane job, and the ledger's chunks sum to `L`, in
    order, on one lane. Each chunk is the chunk planner's chunk at the planner's price, with the depth term at the
    chunk's start, and its K is the planner's K for that price. *Rev 1.18.10 (§15.29):* a chunk priced above the
    tick's share must be exactly the planner's floor, `min(MinPrefillChunk, MaxChunk)` (at least 1), where
    `MaxChunk = min(MaxPrefillChunkBudget, tokens left)`; the floor earns a larger K, as `CreatePrefix`'s does. Rev
    1.18.8 read "none is above the tick's slice", which no chunk can meet when one token alone costs more than the
    share.
  - (b) **Byte-equal.** The frozen prefix's save blob, and an adopter's tokens, equal those of a one-job build of the
    same ids (a host twin and the box).
  - (c) **Interleaved.** A generation on the build's lane delivers tokens between the build's chunk jobs.
  - (d) **Chunk pricing** (*rev 1.18.11, §15.31*; rev 1.18.8 read "the hitch count over the build, expected 0").
    - **What is read:** the build's own chunk jobs, from the ledger. For each chunk the leg records its price, K,
      worker ms and whether it overran (worker ms > K × `TickBudgetMs`).
    - **Graded:** after calibration, the median chunk's worker ms is at most its K × `TickBudgetMs`, so the price
      is not systematically low.
    - **Recorded, not graded:** the overrun count, the subsystem's hitch counter over the build (every job on the
      lane, with a late delivery and an overrun each counting once), and its breakdown by job kind.
    - **What a failure means:** the cache chunk is mispriced, and the case goes back to calibration (§5.5), not to
      slicing. *Rev 1.18.12 (§15.32):* the price it grades is the `prefix_prefill` arm's, timed under the build's
      conditions. A margin factor on the price is not a remedy.

  Mutants: the build run as one job, which turns (a) and (c) red; a chunk priced at `PrefixBeginCostMs`, which turns
  (a) red; a freeze before the last chunk, which turns (b) red.
- 5.12 [U2] *(Rev 1.18; PC-16 (a), §5.6, §5.9.)* **Residency fallbacks.** `WarmPrefix` with every entry leased and a
  charge that cannot fit resolves `BudgetExhausted`, nothing drawn (fill probe behind the barrier); with an unknown
  `SchemaName`, `UnknownSchema`, no entry, no file; with a nested spec before U6, refused `NestedRequiresBudgetMode`
  synchronously. `DropPrefix(MemoryAndDisk)` with the disk tier off: the memory part applies, `Out.Disk ==
  DiskTierOff`; on a key with no file, `NoFile`; with a delete the file API refuses (a test seam), `Failed`, one log
  line. A drop on a cooked entry's key leaves the set registered, and the next request is a `Cooked` hit. Mutant: the
  drop's disk part reaches the cooked tier (marks the set entry stale) → the `Cooked` hit is lost → red.
  *Rev 1.18.1 (audit G9, G12):* a warm with the pool filled from outside the cache (5.5's construction) resolves
  `PoolExhausted`; `DropPrefix(Memory)` reads `Disk == NotRequested`; `DropPrefix(MemoryAndDisk)` on a key that is
  not resident but has a disk file reads `NotCached` and `Deleted`, and the file is gone (the "purge a stale file"
  use).
  *Rev 1.18.4 (suite audit G4; §5.9 Reason codes):* each warm and drop row is graded in 5.4's form:
  - the named result;
  - **one log line** carrying the row's reason name and the key;
  - **that reason's counter raised by exactly one**, with every other reason's counter unchanged.

  The rows are the warm's `BudgetExhausted`, `PoolExhausted` and `UnknownSchema`, and the refused delete's
  `DropDeleteFailed`. Rev 1.18's "one log line" for the refused delete becomes that line, with its reason. A
  deferred drop raises the drop diagnostics' `Deferred` by one, logs nothing and leaves every reason counter
  unchanged. Mutants: a row logged under another reason → red on the line's reason and on the two counters; a
  counter not raised → red.
  *Rev 1.18.5 (U2 code review, finding 4; §5.6 `GetPrefixOpResult`):* a warm whose build fails (the builds-fail
  seam, the review's probe P3) reads `Failed`, with `Out.Reason == BuildFailed`, one `BuildFailed` log line and that
  counter up by one. It does not read `Cancelled`. Mutant: a failed build mapped to `Cancelled` → red.
- 5.13 [U6] *(Rev 1.18; PC-15 Q7, §5.3 item 4; the U6 draft's W4/R40.)* **GPU builds in chunks.** A test-only seam
  lowers the submission thread's lock-holder bound so that a one-job build of the fixture's longest prefix would
  overrun it. The chunked build completes, each chunk job ends inside the bound (the seam records each job's
  duration against it), the thread is never marked unresponsive and the device-loss path is never taken; the
  build's adopters equal its prefill twin's tokens, and it holds one charge (the ledger's in-flight count reads one
  build throughout). Mutant: the build submitted as one job → the lock-holder check trips → red. *Rev 1.18.1 (audit
  N4):* a second request for the key issued between two chunks attaches to the build (one prefill sequence counted)
  and is delivered with the first.
- 5.14 [U6] *(Rev 1.18; PC-15 Q9, §5.4 *After a terminal GPU device loss*; the U6 draft's W3.)* A simulated
  terminal device loss (the GPU subsystem's existing test seam) with a cache release queued: the skipped release is
  issued exactly once (a release-issue counter), no `ReleaseFailed` line is logged after the loss tick, its refund
  stays pending, requests fall back by name, and teardown and re-`Configure()` complete without waiting on it.
  Mutant: re-issue kept after device loss → the counter rises every tick → red. *Rev 1.18.5 (U2 code review,
  finding 3; §5.4 *a skipped release is not a release that returned*):* the skipped release refunds nothing, logs
  no `ReleaseFailed`, and its handle stays in the live-handle set. At teardown the sweep releases it exactly once
  (the release-counting seam). The same holds for a job skipped because the submission thread is marked
  unresponsive. Mutant: a skipped job delivered as a returned `Failed` → the refund completes at once and the handle
  leaves the set, so the sweep releases nothing → red on the ledger and on the count. *Rev 1.18.1 (plan review F1, audit
  G13 (a)):* rev 1.18's beside-leg ("an engine-failed release refunds nothing, is issued again and then applies") is
  withdrawn: its seam kept the handle alive, which no engine release does, so it certified the defect. The GPU legs
  of 5.15 run here instead, on a real GPU release.
- 5.15 [U2, U6 GPU legs] *(Rev 1.18.1; plan review F1, audit G13 (a); §5.4 *What a release's status means*.)* **A
  release that returned has applied.** The seam models the engine's real failing release: it calls the real
  `sslm_prefix_release` (so the handle is consumed) and then substitutes a non-`OK` status, a page-module refusal
  (`SSLM_INVALID_ARGUMENT`, `SSLM_ARTIFACT_REJECTED`). Asserted: the release-counting seam shows **one** release of
  that handle and never a second; `ReleaseFailed` is logged once; the family's pending refund completes at that
  apply; a waiter holding a claim on it is admitted at that apply pass; at teardown the pool destroy returns
  `SSLM_OK` within a watchdog bound. [U6] GPU legs: a release that returns `SSLM_BUSY` is issued again at a later
  tick, its refund pending until that issue returns, and the handle is released exactly once in all; a GPU release
  returning any other status is not re-issued. *Rev 1.18.2 (the strike on rev 1.18.1, lesser plane 3; §5.4):* the
  `SSLM_BUSY` leg is forced by the GPU **busy-release substitution** seam (§8). The seam returns `SSLM_BUSY` for a
  chosen number of issues **without calling** `sslm_gpu_prefix_release`, which is exactly the engine's `SSLM_BUSY`
  contract: the handle is untouched. It then lets the real release run. The engine's own path to `SSLM_BUSY` (the
  prefix's internal sequence `Submitted`) is not reachable from the cache, so "driven by a Submitted sequence" is
  withdrawn. Mutants:
  re-issue on a non-`BUSY` failure (the U2 draft's V12 rule, `e7811590e`) → a second release of a freed handle on
  the seam (and, under ASan, a use-after-free) → red; refund withheld on failure → the waiter never wakes within the
  watchdog → red; no re-issue on `SSLM_BUSY` → the GPU handle is never released → red.

**6. Numerical edges and determinism — applicable (the spine).**
- 6.1 [E2] For each shape of 4.1: rows of the restored prefix (peek, formula in the test) byte-equal the saved
  prefix's; residual, scale, counts, `current_token`, schema and walk equal. *Operand (rev 1.1):* the header of an
  `SSB5`/`SSB6` save of an adopter of each prefix, which is blob bytes under the grading rule; fixtures per the
  preamble. A fresh post-adoption `SSB6` writes the no-current-token sentinel for a nonnegative prefix token;
  this header does not grade that token's value. Cell 6.4 grades it in the `SPX1` blob.
- 6.2 [E2] Adopters of the restored prefix and of the prefill twin, 64 tokens each, greedy and damped-greedy, with
  and without a bound schema: identical token streams. Both adopter modes.
- 6.3 [E3] A nested child restored with and without sharing, and built by `begin_from` + prefill: the three
  children's adopters emit identical tokens; the shared restore's pages are the parent's (peek addresses equal).
- 6.4 [E2] Save → restore → save: the two blobs are byte-identical (round-trip bit-equality), for root, nested,
  empty and schema-bound prefixes. An isolated mutant that changes only a nonnegative restored `current_token`
  must fail this blob comparison; the test reads the field from `SPX1` on both sides. E1 authors the mutant and
  assertion; E2 first records the unmutated pass, then runs the isolated mutant and records its failure.
- 6.5 [F3] GPU: 6.1–6.4 on both vendors; GPU adopter tokens equal the GPU prefill twin's.
- 6.6 [E4] Multithreaded decode (1.11's parallel-for hook) over a restored prefix equals single-threaded decode.
  *Rev 1.14 (audit S3):* the existing leg (`Parallel66`, `e4_real.cpp:410-453` at `3b6631f`) maps pkv_def through
  `GetFixture`, which pins it to 16, and is recorded as a **pinned** leg. A **rule-B leg** is added on the real 0.5B
  artifact (rule `B = 128`), mapped on its own and asserting its B (§7 preamble): a prefix of `L ≥ 2B` (`L = 300`,
  three pages, the last partial) restored in a rule-B page pool, then 32 tokens decoded multithreaded and compared
  with 32 decoded serially from an identical restore. *Rev 1.18.4 (suite audit G9):* rev 1.14's mutant, "a
  parallel-path K/V read that addresses pages at a B other than the pool's", is **withdrawn**. The parallel-for hook
  splits only the token finish's logits rows and, with `SSLM_PARALLEL_FOR_MATVEC` (which both legs set), the
  one-row projections (`include/superslm/parallel_for.h:10-18`). K/V reads are never split, so serial and parallel
  run the same attention read: a K/V addressing defect is common-mode, and this equality cannot see it. The
  mutant is now a **matvec-split defect**: the last projection task, which the split rule lets be smaller, has its
  row range ended one row short. That row keeps its stale value on the parallel path only, so the parallel tokens
  differ → red on both legs. The rule-B leg's added value is running the hook at real geometry. K/V addressing at
  rule B is graded by 10.1 and 10.2 against licensed references, and by this leg's restore and re-save identity.
- 6.7 [E2, F2] *(Rev 1.3; collision-class fidelity, the strike's census as a cell.)* On a three-schema model with
  distinct hashes, prefixes bound to each schema (index 2 included), with the walk advanced past state 0, are saved
  and restored. For each restored prefix and its prefill twin, adopters bound to each of the three schemas and an
  unbound adopter give the same disposition (`SSLM_OK` or `SSLM_RESTORE_SCHEMA_MISMATCH`, PKV §3.9 A18) and, where
  admitted, the same 64 tokens. The restored prefix's schema index is read through an adopter's `SSB6` header hash
  and walk (6.1's operand). Mutant: restore resolves "the first schema whose `state_count` admits the walk" rather
  than by hash → red. The collision itself is unreachable under PC-12; 2.13 pins that.
- 6.8 [F2] *(Rev 1.18; PC-15 Q1, §4 item 7; `pcg_cell_budget_snapshot`, built on the software GPU at `fdbb97b` and
  `83a82c2`.)* **A prefix stopped after landing tokens restores with the same prefill snapshot.** Two legs, each a
  prefix whose last prefill lands tokens and then stops: (a) on its budget (budget 20, 30 tokens in one call, 20
  land, `SSLM_GPU_KV_BUDGET_EXCEEDED`); (b) on its bound schema rejecting a token after earlier ones land
  (`SSLM_SEQUENCE_REJECTED` from the DFA exit). Each is frozen, saved and restored; an adopter of the original and an
  adopter of the restore each read `sslm_gpu_seq_read_prefill_final_hidden` (status, codes, scale), which must be
  equal and `SSLM_OK`, and decode 8 greedy tokens, which must be equal. A plain sequence (kind 0) stopped the same
  way still reads `SSLM_PREFILL_HIDDEN_UNAVAILABLE` (*rev 1.18.1*, audit G4: the built `pcg_cell_budget_snapshot`
  has no such leg yet; the test author adds two, a pooled plain GPU sequence stopped on its budget, budget 20 and 30
  tokens, and one stopped by a schema rejection after landing tokens, each reading
  `SSLM_PREFILL_HIDDEN_UNAVAILABLE`; mutant: the refresh applied to every sequence kind → red). Mutant: no refresh on either exit → the original's adopter reads
  `SSLM_PREFILL_HIDDEN_UNAVAILABLE` and the restored one's `SSLM_OK` → red (the recorded pre-fix run: 2 failures of
  17 checks on leg (a)). Box: F2's MSVC run on both certified GPUs (§9).
  *Rev 1.18.4 (code review round 1; §4 item 2's freeze-time setting):* three multi-call legs, built at `50bab8cd`.
  Each compares the original with its restored copy:
  - **A:** a budget stop that lands 20 tokens, then a retry that lands none. The adopters' snapshot reads must be
    equal.
  - **B:** 10 tokens land `SSLM_OK`, then token 256 is refused `SSLM_TOKEN_ID_OUT_OF_RANGE`. The adopters' snapshot
    reads must be equal.
  - **D:** an empty prefix, a zero-count prefill, freeze, and a restore at `L = 0`. A `begin_from` child's
    `sslm_gpu_prefix_set_schema(0)` must be admitted on both.

  Before the fix, the original and the copy split on each leg. The mutant `exit-refresh-only` (refresh at every
  failing exit, nothing set at freeze) is red through D only. Once the review's remedy lands, the mutant is
  re-anchored as "the freeze-time setting removed". That mutant is red through D, and also through A and B if the
  remedy removes the exit refresh.

**7. Contract claims — applicable, each mapped to a cell that fails if the claim is false.**
- 7.1 [E2] "Adopters of a restored prefix are bit-identical to adopters of the prefill-built prefix" (§3.4 (i),
  *rev 1.3*: within PC-12 and the version scope; *rev 1.18*, Q2: in pools of the restore pool's page size, with the
  adopter in a pool of another page size refused `SSLM_INVALID_ARGUMENT`, which 8.13 grades for restored and built prefixes alike) → 6.1–6.4, 6.7; mutant: restore skips copying the residual → 6.2 red; mutant: restore skips the saturation counts, which
  no decode reads → 6.1 and 6.4 red (*rev 1.1*: rev 1 routed it to 6.2, which cannot see it).
  *Rev 1.8, D-SLM7998; rev 1.9, T-3042 N1:* a restore mutant that changes **only `hidden_scale`** is graded by
  6.1's independent `SSB6` header operand and 6.4's blob equality, never by 6.2, 8.x or 10.x token streams.
  The five saturation counts use the same two operands. A wrong nonnegative `current_token` value is graded by
  6.4's `SPX1` blob equality alone: a fresh adopter's `SSB6` header writes the no-current-token sentinel, so 6.1
  cannot distinguish that mutant. Decode reads only the token's sign and overwrites its value before the next
  embed. The test author keeps these field-isolation mutants distinct from the token-stream mutants and runs
  the current-token mutant against 6.4 explicitly after its unmutated E2 case passes.
- 7.2 [E2] "Every validation refusal draws nothing and changes no refcount" → 2.1–2.5, 5.1, 5.6; mutant: draw
  first and return the pages on refusal → 5.6 red. *Rev 1.6:* "every refusal after a draw returns every page and
  refcount it took" (§3.6) → 5.2, 5.11.
- 7.3 [E2] "Any accidental change to a blob is refused" (§3.4 (ii)) → 2.1, 2.2; mutant: digest computed over one
  byte too few → 2.2's final-byte case red.
- 7.4 [E2] *(Rev 1.4: narrowed.)* "A root restore, or a nested restore with a matching parent, never draws more than
  the build it replaces" → in a pool left with exactly `ceil(L/B)` free pages (fill probe; `ceil(L/B) ≥ 2`), a root
  restore is admitted, and a 2-page create is then refused `SSLM_KV_POOL_EXHAUSTED` (*rev 1.5*: rev 1.4 read "0
  free" from the fill probe, which cannot certify fewer than 2); mutant: restore draws `R(L) = ceil(L/B) + 1` → refused → red. Nested twin at
  [E3]: a pool with exactly `ceil(L/B) − floor(origin/B)` free and a matching hint → admitted. *Rev 1.6:* the shape
  is fixed with `ceil(L/B) − floor(origin/B) ≥ 2`: the PKV cohort shape (origin 1,000, `L` 1,200) gives 13 at
  `B = 16` and 50 at the `B = 4` view (executed). "A private nested
  restore needs `ceil(L/B)` free pages" → 5.10.
- 7.5 [E3] "The hint is never required for correctness, and shares only under (a)–(f)" → 2.6, 9.3; mutant: (f) compares only
  layer 0 → 2.6's last-page case red (the differing byte is placed in the last layer's V half).
- 7.6 [E2] "A blob restores only on the producer that wrote it" → 2.3; mutants: the version check removed, and
  the check comparing only major and minor → each red.
- 7.7 [E2] "The version header equals CMake's version" → `check_version_header.py`'s own must-reject (a header one
  patch behind) and must-accept.
- 7.8 [U2] "No hashing, file IO or Layer-1 call on the game thread" → a **new** off-game-thread assert (the
  plugin's existing checks assert only *on* it), registered in §8 with its must-reject: an instrumented call made
  on the game thread trips it. Mutant: key hashing moved into `RequestPrefix` → red. *Rev 1.18.1 (audit G12):* the
  assert also covers `WarmPrefix` and `DropPrefix`; mutant: the key hashed inside `DropPrefix` to decide synchronously
  → red.
- 7.9 [U2] "Every fallback ends in a recompute or a named refusal, never in bytes the engine refused" → 5.4.
  *Rev 1.18.1 (audit N1):* §5.9's rev 1.18 rows map by their step: the warm and drop rows to 5.4 and 5.12 [U2], the
  `TextChanged` row to 2.17 [U3], the device-loss row to 5.14 [U6].
- 7.10 [U2] *(Rev 1.2: the pinned exception removed by the charge rule.)* "The cache's pages, resident and in
  flight, never exceed the budget" → 1.4, 5.5 and 8.10 (and 1.9 at U6), on the fill probe. Mutants: a build draws
  without being charged → 8.10 red; admission over budget when every entry is pinned → 5.5 red.
- 7.15 [E2] *(Rev 1.3.)* "A model maps only if its schema-name hashes are distinct and non-zero" (PC-12) → 2.13;
  and "the shipped `SSB5` path is closed by the same rule" → 2.13, 2.14.
- 7.16 [E2] *(Rev 1.6.)* "`SSLM_RESTORE_PRODUCER_MISMATCH` is appended at 30 and no existing ordinal moves, in
  the X-macro and in every hand-written mirror" (§3.8) → the existing Gate C ctest entries
  `t2139_gate_c_must_accept_compiles`, `t2139_gate_c_real_suite_side_check_compiles` and
  `t2141_gate_c_t2138_suite_side_must_accept_compiles`, with their commissioned must-rejects
  (`t2139_gate_c_must_reject_fails_to_compile`, `t2139_gate_c_sentinel_must_reject_fails_to_compile`,
  `t2141_gate_c_t2138_suite_side_must_reject_fails_to_compile`). No new instrument. A mirror left one behind
  fails its sentinel comparison.
- 7.17 [E5] *(Rev 1.7, D-SLM7991, superseding rev 1.6's D-SLM7989; T-3026 W-1, T-3027 F15.)* "Every status
  ordinal `docs/api.md` states is that status's ordinal, and the new status is stated at 30" → a source check that
  finds every statement in `docs/api.md` pairing a backticked status name with "ordinal" and an integer on one line,
  and requires each integer to equal that name's value in the `SSLM_STATUS_ENUM_LIST` X-macro. At `5d5ca58` there
  are two, both `SSLM_KV_BUDGET_EXCEEDED` at 29 (`:553`, `:787`; S27). It also requires
  `SSLM_RESTORE_PRODUCER_MISMATCH` to be stated, at 30. It refuses when it finds fewer pairs for a name than its
  recorded floor (`SSLM_KV_BUDGET_EXCEEDED` twice, `SSLM_RESTORE_PRODUCER_MISMATCH` once), so a statement reworded
  past the locator drops out loudly rather than letting the check pass on fewer statements. Rev 1.6 compared every
  stated ordinal with `SSLM_STATUS_NEXT_FREE − 1`; once 30 is appended, both true statements of 29 fail that, and a
  doc edited to call `SSLM_KV_BUDGET_EXCEEDED` "ordinal 30" passes it. Registered as an instrument (§8).
  Must-rejects: `SSLM_KV_BUDGET_EXCEEDED` edited to 30; the new status stated at 31; one `SSLM_KV_BUDGET_EXCEEDED`
  statement reworded so the locator misses it. The locator takes the integer **immediately following**
  "ordinal", not another integer on the line (which may include a version such as `1.12.0`). Must-accept:
  a constructed `docs/api.md` from `5d5ca58` with one correct statement of
  `SSLM_RESTORE_PRODUCER_MISMATCH` at 30, checked against E2's X-macro. The release tree is the cell's own
  reading after the edit. The check is written red by the test author at E5's start, before the release edits
  `docs/api.md`. It is red then for its own reason: the new status is not yet stated. Its constructions are
  the debunker's (§9 E5).
  *Rev 1.18.4 (suite audit G11):* the one-line locator misses a statement whose name is on the next line. At
  `d42cec75`, `docs/api.md:518` read "One covers precached prefixes (1.15.0, ordinal 30):", with
  `SSLM_RESTORE_PRODUCER_MISMATCH` on `:519`; after the relabel it is `:521`. Edited to 31, the check stayed green
  (executed). The locator is widened, so that **every "ordinal N" in `docs/api.md` is read**:
  - A line with "ordinal N" and no backticked status name pairs with the first backticked status name on the
    next line of the same paragraph.
  - If that line has none either, the statement is **unpaired** and the check fails.
  - Ranges ("ordinals 24 to 29") are still not statements.

  `SSLM_RESTORE_PRODUCER_MISMATCH`'s floor rises to 2. A new must-reject is the paired sentence edited to 31, and
  another is an "ordinal N" with no name on its line or the next. *Also corrected (the red suite's reading 6):* the
  red-first premise above is stale. `docs/api.md:1138` (at `d42cec75`) has stated `SSLM_RESTORE_PRODUCER_MISMATCH` at 30 since
  `31e41d82` (2026-10-03), so the check is green when written. Its commissioning starts from the constructed
  mutants (§9 E5), not from a red tree.
- 7.11 [E2] *(Rev 1.1.)* "A hostile blob is refused or accepted without undefined behaviour" (§3.4 (iii)) → 2.3,
  2.3b; mutant: step 8's `current_token` lower bound removed → 2.3's −2 case red.
- 7.12 [E2] *(Rev 1.1; format conformance.)* "The blob is the §3.2 table" → an independent parser (python
  `struct`, not the engine's reader) reads a saved blob. It asserts every field at its offset against values the
  test knows, the rows in canonical order against the peek, and the
  digest recomputed with `hashlib` over the blob with the digest zeroed. *Operands per field (rev 1.2; every
  field):*
  - version and producer: the known build;
  - `L`, `origin`: the fixture's;
  - model hash: the artifact's;
  - `kv_precision`, `backend`: the known values;
  - schema hash: `Fnv1a64` of the known schema name; walk: the known state;
  - `m` and `e`: the header of an `SSB6` save of the prefill twin's adopter;
  - the five counts: test-owned expected values fixed before either serialization, nonzero and pairwise distinct.
    Pass those values to the parser's `--counts` operand on the built blob; the adopter's `SSB6` is corroboration,
    not the count oracle. An isolated symmetric count swap must turn the independent comparison red;
  - `current_token`: the known nonnegative last prefill token from the fixture input, checked directly at its
    `SPX1` offset. A fresh adopter's `SSB6` writes a sentinel for this field and is not its oracle.

  The fixture is bound to a known nonzero schema hash and advanced walk and has five nonzero pairwise-distinct
  counts. The independent parser's must-accept is a built-verb save of that fixture. An isolated token edit and a
  writer-and-reader-symmetric field swap each turn this parser red, with separate receipts, even when the engine
  writer and reader agree with each other. So a symmetric swap of two counts, of `m` and `e`, or of the schema
  hash and the walk goes red.
  And "carries no page size" (§3.1): use the same fixture's B=16 model and a B=4 artifact twin made by changing
  only CFG1 `kv_block_size` and recomputing the artifact integrity hash. Freeze equal token inputs in both models;
  independently assert the pre-save semantic fields and peeked K/V rows agree. Parse both saved `SPX1` blobs,
  verify each digest and each model-hash field against its own artifact, and require the two model hashes to
  differ. Compare every byte of the two blobs **outside** digest offsets [28,60) and model-hash offsets [60,92)
  for equality, including all canonical rows. A B-dependent serialization of any other field or row turns this
  cell red. Whole-blob equality across these two model artifacts is impossible because their B field changes
  model identity; the independent per-blob format checks still apply to both. *Rev 1.13 (re-anchor F2):* CFG1's
  `kv_block_size` no longer selects B, so the twin's patch now only makes its model hash differ; both views take
  their B from the test-only pin (16 and 4), and the cell asserts each pin's `SSLM_OK` and B, as it already checks
  `B() == 16` and `B() == 4` (`e2_format.cpp:171-188` at `3b6631f`). *Pool-kind leg (rev 1.13, re-anchor F5):* one
  model, unpinned, at its rule B. *Rev 1.14 (audit S7):* the model is the synthetic 0.5B shape at its rule `B = 64`,
  mapped on its own (§7 preamble), and the prefix is 7.12's own `L = 65` fixture, so `L ≥ B + 1`: two pages in the
  rule-B pool, one in the contiguous pool. The leg asserts `sslm_kv_page_positions == 64` against the test-written
  rule first. At `L ≤ B` the prefix is one page in both pool kinds, and a page-ordered writer would be canonical
  there. Build a prefix by prefill in a rule-B page pool and its twin by prefill in a
  contiguous pool, from equal tokens; save both, then restore each blob into the **other** pool kind and save
  again. All four blobs must be **byte-identical, digest and model hash included**, because the model is one
  artifact. This is the production-path proof that no page size reaches the blob, with no patched artifact. Mutant:
  a writer that serializes rows in the holder's page order → red here and in 4.7. *Routed to the test author at E1
  (rev 1.6):* the coverage audit T-3024's F7, §3.1's "carries no mode or budget" half, to be authored from that
  audit's stated properties (§9 E1).
- 7.13 [U2] *(Rev 1.1; the memory tier.)* Memory-hit oracle: request, Ready, request again; source `Memory`, the
  restore-count and prefill-count seams unchanged, tokens equal the twin. LRU oracle: fill to budget, touch the
  entries in an order that is neither the insertion order nor its reverse (*rev 1.2*), admit one more; the evicted
  key is the least recently touched. Mutants: evict the most recent → red; evict in insertion order → red.
- 7.14 [U2] *(Rev 1.1; rev 1.2 serves a hit.)* "The disk tier is trimmed to its limit, oldest first, and a hit
  refreshes an entry" → write A, B and C in that order, serve a `Disk` hit on A, then write past
  `PrefixDiskCacheMaxBytes`. After the write, and after a re-`Configure()`, the directory is at or under the limit;
  A survives, and B is the first removed. Mutant: no time refresh on a hit → red.
- 7.18 [U2] *(Rev 1.18; PC-16 (a).)* "A drop of a leased entry never invalidates a live lease" → with two leases on
  entry E, one adopt of it queued and a child job of it queued [U6 leg], `DropPrefix(E)` resolves `DropDeferred`; both
  leases still adopt (equal tokens to the twin), the release-counting seam shows E unreleased; after the queued work
  drains and lease 1 is returned, E is still resident; after lease 2 is returned, E is evicted at the next apply
  pass, once the hashes of ops issued before that pass have resolved (§5.6 *When a mark is applied*; *rev 1.18.3*,
  the re-check's m3: this fixture issues no op with a pending hash, so that is the same pass) (the drop
  diagnostic's marks-applied count up by one), its refund pending until its release applies. A drop on a
  key whose build is in flight resolves `DropDeferred`, the build completes and serves its requester, and the mark
  applies when that lease is returned. A `RequestPrefix` for E between the drop and the last return clears the mark,
  and E stays resident after every return. *Rev 1.18.2 (the strike on rev 1.18.1, Fracture 1):* so does a
  `RequestPrefix` for E issued after the last return and before that frame's pass, with its hash held past the pass
  (3.13's T1 run). *Rev 1.18.1 (audit G12):* a drop of a key whose request is a waiter reads
  `DropDeferred`; the waiter is admitted and served, and the mark applies at its lease's return. Mutants (the model's m1 as a cell): a drop releases a pinned entry at once
  → a lease's adopt is refused or the seam shows E released under a lease → red; the mark not cleared by a later
  request → E evicted after that request's lease returns → red.
- 7.19 [U2] *(Rev 1.18; PC-16 (a).)* "A warm is a request that takes no lease" → `WarmPrefix(S)` on a cold cache
  resolves `Resident` after one build (the prefill-count seam up by one) and leaves no lease (a `ReturnPrefix` of
  nothing exists; the entry is unpinned); a following `RequestPrefix(S)` is a `Memory` hit with no prefill; a warm of
  a resident key touches the LRU (it survives an eviction that takes an older entry); a warm and a request for one key
  in one tick share one build (coalescing, one prefill); with the disk tier on, a warm writes back as a request does.
  With the budget filled by warmed entries and nothing leased, a request evicts them like any unleased entry.
  Mutants: a warm that leases → the warmed entry is pinned and the last eviction refuses → red; a warm that does not
  coalesce → two prefills → red. *Rev 1.18.1 (audit G9, G18):*
  - **Tier order**, read through the restore-count, disk-read and prefill-count seams (a warm reads `Resident` from
    every source): a warm on a key with a registered cooked entry restores once, prefill count 0; a warm on a key with
    a disk file reads it once, prefill count 0; each is followed by a `Memory` hit. Mutant: a warm that always
    recomputes → prefill count 1 → red.
  - **Waiting**: a warm whose charge fits only by an unclaimed pending refund waits, draws nothing before that
    release's apply, and is admitted there. Mutant: the warm refused instead → red.
  - **Coalescing, both orders**: a request then a warm, and a warm then a request, each in one tick, one prefill
    each; a warm issued while a request for its key is a waiter attaches to that waiter.
  - **LRU**: mutant: a warm hit that does not touch the LRU → the warmed entry is evicted first → red.
- 7.20 [U3] *(Rev 1.18; PC-16 (b), §5.8.)* "The runtime never tokenizes, and a text entry is keyed by its stored ids"
  → a set with one `Text` entry and one `Tokens` entry whose ids are the text's tokenization, both built: their keys
  are equal (one disk file, one cooked wrapper key), the text entry is a `Cooked` hit through
  `FindEntrySpec`, and a test-only tokenize-call counter reads **0** across
  `Configure()`, `RegisterPrefixSet`, `FindEntrySpec`, the request and the adopt (and > 0 at the editor build, its positive clause).
  *Rev 1.18.1 (plan review F14, audit G6):* the counter covers **every** `sslm_tokenize` call the plugin makes, both
  sites (`SuperSLMDetokenizer.cpp`'s `TokenizeText` and `SuperSLMSubsystem.cpp`'s `Tokenize`), and its must-reject
  makes one call through each site and reads each. The subsystem is configured with the set's own model asset, the
  fixture under which the two sites map the same artifact.
  The built ids equal `USuperSLMSubsystem::Tokenize` of the text on the same model. Mutants: the runtime re-tokenizes
  `Text` through either site → the counter reads > 0 → red; the build prepends a token → the ids differ from `Tokenize` → red.
- 7.21 [U2] *(Rev 1.18; PC-15 Q4, §5.10.)* "`AdoptPrefix` on a lease not yet Ready is refused `Malformed`, nothing
  queued" → a lease whose build is held (the build hold, §8) is adopted: `Malformed` with the "not Ready" error,
  nothing queued (the adopt-queue count unchanged), and the same lease adopts once Ready. Mutant: the adopt queued
  before Ready → red. *Rev 1.18.1 (audit N3):* also a live lease whose request resolved to a refusal
  (`BudgetExhausted`) and has not been returned: `Malformed` ("not Ready"), nothing queued, the disposition the core's
  `NotReady` already gives.
- 7.22 [M1, E5; the 1.15.0 release] *(Rev 1.18.1; plan review F4, F5, audit G3; §4 preamble.)* "In 1.14.0's tree, and
  in 1.15.0's, the GPU twins are labelled unreleased until 1.16.0" → a source check shaped like 7.17, registered in
  §8: every mention of `sslm_gpu_prefix_save` and `sslm_gpu_prefix_restore` in `gpu_1p0.h` and `docs/api.md`, and
  the status-30 sentence, carries "unreleased until 1.16.0"; no release number other than 1.16.0 stands beside this
  plan's GPU code in any file of §4 preamble's list (or any file the grep finds that the list missed). It runs at M1,
  at E5's publish gate and at the 1.15.0 release's publish gate; F4 runs its inverse after lifting the label.
  Must-rejects (the debunker's): a stale "(1.15.0)"; one mention left without the label. Must-accept: the labelled
  tree.
  *Rev 1.18.4 (the red suite's reading 7, the suite audit's ruling on it, and N6): the unit is the paragraph or the
  comment, not the line.* This is what was built, and it is ruled before M1's relabel runs. A **mention unit** is a
  paragraph of `docs/api.md` (text between blank lines) or one comment in `gpu_1p0.h` (a `/* … */` block or a run
  of `//` lines). Each unit that names either verb, or is the status-30 sentence, carries "unreleased until 1.16.0"
  at least once. Under the line reading, a line such as `gpu_1p0.h:1034` or `:1044` (at `d42cec75`) would need its own label, though
  it sits in the unit that `:1029`'s label covers. The relabelled tree was accepted under this unit (9 mention
  units, 13 label sites).

  "Any file the grep finds" is defined as the sweep that was built: a "(1.15.0" label in an unlisted file whose path
  contains `gpu`, or in a `tools/` file within two lines of a GPU identifier. *Known limit:* the label-site locator is
  lexical (a line naming "precached prefix" or either verb). A stale label phrased otherwise, such as "GPU prefix
  save and restore (1.15.0)" beside `sslm_gpu_prefix_save_impl`, escapes it (executed). The debunker's commissioning
  records this limit and does not widen the check.
- 7.23 [U2] *(Rev 1.18.1; audit G17.)* "Each Blueprint node gives what its C++ verb gives" → an automation test
  calling each Blueprint library function (`Warm Prefix`, `Drop Prefix`, `Get Prefix Op Result`, and the existing
  `Request Prefix`, `Return Prefix`, `Get Prefix Source`, `Register Prefix Set`) and the C++ verb on the same input:
  both `Scope` values, the op ids, and every `GetPrefixOpResult` output field equal. Mutants: a `Drop Prefix` node
  that passes a fixed `Scope` → red; a `Disk` output never written → red.

**8. Composition — applicable.** Each pair crossed.
- 8.1 [E2] Restored prefix × legacy (copy) adopter and × budget (share) adopter: both equal the twin; the budget
  adopter shares `floor(L/B)` pages.
- 8.2 [E3] Restored prefix as `begin_from` parent; restored prefix as restore hint; restored child as parent of a
  grandchild restore.
- 8.3 [E2] A sequence adopted from a restored prefix, saved (`SSB6`), and restored with `sslm_seq_restore_shared`
  against the restored prefix: shares, keeps (PKV §3.9 A4).
- 8.4 [E2] Schema-bound prefix × adopter with the same schema / another schema / unbound: PKV §3.9 A18's outcomes,
  identical to the twin's.
- 8.5 [E2] Restored prefix × an adapter bound on the adopter: tokens equal the twin's.
- 8.6 [E2] Restored prefix in a pool made by the old verb, where every other handle is legacy and the restored
  prefix (recorded `budget`, *rev 1.2*) has no budget sharer: legacy admission counts unchanged (PKV §8's
  legacy-create admission count).
- 8.7 [F2] A CPU `SPX1` given to the GPU restore → `SSLM_SEQUENCE_KV_BUFFER_MISMATCH`; a GPU one given to the CPU
  restore → `SSLM_INVALID_ARGUMENT` (*rev 1.2*: statuses named; backend precedes producer, 5.7).
- 8.8 [U2] Cache × sequence save/restore: a sequence adopted from a cached prefix saves and restores as today.
- 8.9 [U4] Cache × the plugin's determinism self-check and its query path: unchanged results, with cache hits
  > 0 during the run.
- 8.10 [U2] *(Rev 1.1; PC-11; rev 1.2 made it dynamic.)* 1.0.0's behaviour beside a working cache, in three
  phases:
  - (a) Static: the cache filled to its budget, then `BlockCount` warm sequences created and adopted from
    `CreatePrefix` prefixes, and `CreatePrefix` admitted up to `PrefixBlockCount` with the next refused as today.
  - (b) Every entry leased and one more request: `BudgetExhausted`; then (a)'s calls all admitted.
  - (c) K ≥ 8 requests for distinct uncached keys in one tick (no coalescing). While they run on the lanes, call
    `CreatePrefix` up to `PrefixBlockCount` and reset a released warm slot.

  None of 1.0.0's calls is refused under its cap. The cache's pages, including those in flight, never exceed the
  budget's pages at any sampled tick, read by the fill probe in 1.4's admission form.
  - *Subtrahend (rev 1.7, D-SLM7995; T-3026 W-5, T-3027 F10).* 1.4's `F_min` omits pages that 8.10 creates
    between readings: its 1.0.0 prefixes are neither the sequences' reservation nor the cache's. 8.10's reading is
    therefore `F_min(t) = N − R_seq − P₁₀(t) − floor(PrefixCacheBudgetBytes / page size)`, where `P₁₀(t)` is the
    pages held by live 1.0.0 prefixes at the reading.
    - A legacy `sslm_prefix_begin` draws `cap_pages` (`sslm_abi.cpp:2155-2170`), and its first freeze returns the
      reserve, leaving `ceil(L/B)` (`:2301-2306`).
    - `R_seq` stays constant: the warm sequences are whole-reserve holders created at `Configure()`, and a reset
      keeps their pages (plugin `SuperSLMSubsystem.cpp:2095-2101`, `:3261-3263`, in this worktree's tree).
    - `P₁₀(t)` must be exact, not an upper bound. An overstatement of it is a page the cache can overrun unseen.
    - It is made exact by construction. Every 1.0.0 prefix has a length with `ceil(Lᵢ/B) = cap_pages`, so frozen and
      unfrozen hold the same, and `P₁₀(t) = cap_pages · live(t)`. The test-only counter inserts each handle returned
      by a successful 1.0.0 `CreatePrefix` begin job into a set. It removes a handle and decrements `live(t)` only
      when a release job is for a handle in that set. Cache evictions, failed builds, restored prefixes and
      teardown releases of other handles do not change `live(t)`. Read the set behind the lane-park barrier.
    - The small-cap plugin fixture states `B = 64`, `cap = 256`, and `cap_pages = 4`; `B` divides `cap`.
      *Rev 1.13 (re-anchor F2):* rev 1.12 stated `B = 16`, `cap = 64`, which 1.13.0's rule cannot give for the
      fixture's `D = 64`, int8 K/V (at cap 64 it gives `B = 64`, one page per block; *rev 1.14:* that holds for
      `D·elem ≤ 128`; a wider head, e.g. `D = 512` int8, gives `B = 16` at cap 64, but not this fixture), and the plugin's fixtures have no pin. So the fixture is chosen so the
      rule gives the shape: head width `D = 64`, int8 K/V, cap 256, hence `B = 64` (S30 (a)). Every
      1.0.0 prefix has `Lᵢ = cap`, so its frozen and unfrozen holdings are both four pages. Phase (c)'s cache
      builds instead use `L = 2B = 128`: `L ≡ 0 (mod B)`, `L ≤ cap − B`, and `R(L) = 3` versus
      `ceil(L/B) = 2`. The fixture may use a larger divisible cap with `cap_pages ≥ 3` if it records the same
      margins for each build. A cap `B` does not divide gives a partial last page (S30 (b)), so a full-cap
      prefix's holdings no longer state the margins exactly, and it cannot serve this cell.
    - The counter is registered in §8 with its must-reject.
  - *Floor (rev 1.6; rev 1.7 corrects the slack, T-3026 N-6).* §5.4's sizing leaves phase (a)'s free count at the
    extra blocks' pages minus the budget's pages. That slack is anywhere in `[0, cap_pages − 1]`, below the probe's
    floor at its low end. So the fixture adds **one spare block** that no 1.0.0 call can take: warm sequences are
    capped at `BlockCount` and live prefixes by count at `PrefixBlockCount`. With the exact subtrahend,
    `F_min ≥ cap_pages ≥ 2` at every reading.
  - *What the spare block hides (rev 1.7, T-3027 F16).* It hides nothing from the probe reading, which is a lower
      bound on free pages at one-page resolution, so a cache one page over the budget fails it. It does hide an
    overdraw of up to the slack plus `cap_pages` pages from 8.10's other operand, "none of 1.0.0's calls is refused".
    At K ≥ 8 builds and `cap_pages` ≥ 8, phase (c)'s mutant overdraws K pages and no 1.0.0 call is refused. So that
    mutant rests on the probe reading alone. The fixture sizes entries and builds so the ledger sits at the
    budget, to the page, while the K builds are in flight. Rev 1.6's "also caught by its own operand" is withdrawn.
  - Phases (a) to (c) are read with the lanes parked between jobs, behind the barrier (§8; *rev 1.7*, D-SLM7996).
    Phase (c)'s in-flight pages belong to **builds begun and not yet frozen**. *Rev 1.17:* a test-side batch gate
    held dispatch of their prefill/freeze jobs until all K distinct begin jobs had completed. *Rev 1.18 (PC-15 Q7):* a
    CPU build is one lane job (§5.3 item 4), so there is no job boundary between begin and freeze. The test-only
    **build hold** (§8) stops each build job after its begin returns and before its first prefill, and the lane-park
    barrier counts a lane stopped at the hold as parked: it has drawn its reserve and makes no Layer-1 call until
    released. *Rev 1.18.1 (plan review F7, audit G15; the conductor's ruling):* the lane count is read at source:
    lanes = `Clamp(MaxConcurrentCalls, 1, BlockCount)` (plugin `SuperSLMSubsystem.cpp:2397`,
    `SuperSLMRuntimeConfig.h:68`). The fixture configures **K + 1 lanes**, `MaxConcurrentCalls ≥ K + 1` and
    `BlockCount ≥ K + 1` (K ≥ 8 stands), with phase (a)'s warm sequences, `R_seq` and phase (c)'s margins re-derived
    for that `BlockCount`. The K builds reach the hold on K lanes; phase (c)'s `CreatePrefix` calls and the warm-slot
    reset run on the free lane and complete before the switch parks it, so the required reading has K held builds
    and `P₁₀ > 0`. The switch then parks the other
    lanes. One required reading occurs with **K builds drawn and zero frozen**, witnessed by per-build
    begin/freeze state and the ledger's in-flight count. If the window is absent, the cell fails; a later
    post-freeze reading cannot substitute. Release the hold after the reading and drain the jobs.

  Mutants: a build draws without being charged → red; a build charged `ceil(L/B)` rather than `R(L)` → red at (c)
  on the probe reading, with every build at `L ≡ 0 (mod B)`.
  - *Commissioning of the reading (T-3027 F10; rev 1.8 F22; rev 1.9 T-3041 W-1/T-3042 F1).* Must-accept:
    phase (a) with the cache exactly at the budget and all `PrefixBlockCount` prefixes live is admitted. For
    the must-reject, use the stated `B = 64`, `cap = 256` fixture (*rev 1.13*; rev 1.12's `B = 16` figures
    `L = 16` and `L = 17` scale to `B` and `B + 1`); fill the ledger and physical cache holdings
    exactly to the budget with an **unpinned one-page frozen entry of `L = 64` as the LRU victim** and at
    least one more unpinned, newer one-page entry for the healthy two-page-charge control. Restore `L = 65`:
    the production path draws `ceil(65/64) = 2` pages, while
    the test-only cache-charge seam reports one. Admission evicts that one-page victim; under §5.11 (a) the restore
    then waits for the victim's `sslm_prefix_release` to apply and is admitted at that apply, with the victim's charge
    passed to it (*rev 1.14*). Before the probe,
    assert the victim was evicted, the ledger returned to exactly the budget (budget − 1 + 1), and physical
    holdings reached budget + 1 (budget − 1 + 2). Also assert the 1.0.0 prefix set and `R_seq` did not change.
    The spare block admits the faulty draw; `F_min` is one page above actual free pages, so the admission
    lower-bound probe must refuse. A construction using a two-page LRU victim verifies that undercharge by
    itself is insufficient: its ledger ends at budget − 1 and physical holdings at budget, so it cannot be
    counted as the must-reject. These counts are derived from §5.4's admission rule and the anchored
    `ReservationPages` formula, not executed on a cache. The debunker constructs and runs the one-page
    production-path fault independently of the probe builder. The seam's must-accept runs the same restore
    with its full two-page charge and passes.
- 8.11 [E2] *(Rev 1.1.)* A restored prefix and its prefill twin each given every prefix verb: `sslm_prefix_prefill`
  (frozen), `sslm_prefix_freeze` (idempotent), `sslm_prefix_set_schema` on an empty restored prefix,
  `sslm_prefix_release`, and `sslm_kv_pool_destroy` while it lives (`SSLM_POOL_HAS_LIVE_HANDLES`). Equal
  dispositions.
- 8.13 [E2] *(Rev 1.13; re-anchor F5, X13.)* **Adopt across page sizes, restored against built.** One model,
  unpinned: a rule-B page pool and a contiguous pool. *Rev 1.14:* the model is its own mapping and the cell asserts
  `sslm_kv_page_positions == PageRuleB(D, cap, elem)`, computed in the test, and **rule B < cap**, so the two pool
  kinds have two page sizes (§7 preamble; review S4); every prefix has `L = B + 1` (two pages in the page pool, one
  in the contiguous pool). Four prefixes: a prefill-built prefix and a restored prefix
  (from its blob) in each pool kind. For each prefix, an adopter in the **other** pool kind, once whole_reserve and
  once budget: the restored prefix and the prefill-built one in the same pool give the same disposition,
  `SSLM_INVALID_ARGUMENT`. After the refusal the adopter is unchanged: its `SSB*` save is byte-equal to its save
  before the call, `sslm_seq_set_schema` is still admitted on it (the bind was not revoked), and nothing was drawn
  in either pool (page pool: two-sided fill probe; contiguous pool: the single-page count of 9.8). Control: an
  adopter in the prefix's own pool kind adopts, and its 64 tokens equal the twin's adopter's. Mutant: PKV's
  page-size check in adopt removed (A22) → the whole_reserve cross adopt is admitted → red. The cell's subject is
  the equal disposition, so it fails if a restore ever binds its handle to a geometry other than its pool's.
  *Rev 1.14 (review N4, audit N1), which legs discriminate A22:* only the **whole_reserve adopter in the contiguous
  pool of a page-pool prefix**. The budget legs are refused by A16 (`share_adopt && prefix->pool != seq->pool`,
  `sslm_abi.cpp:2860`) whatever A22 does, and the page-pool adopter of a contiguous-pool prefix at `L = B + 1` is
  refused by the mapped-count check (`adopt_pages > HolderMapped`, `:2893`) with the same status. Those legs stay,
  for the equal-disposition subject, and are recorded as non-discriminating for A22, as 5.6 records its own. The A22
  mutant's receipt is taken on a Release build: under ASan the discriminating leg reports a heap overflow from the
  cap-sized copy rather than a cell red. *Erratum (§15.30, P2):* the **whole_reserve adopter in the page pool of a
  contiguous-pool prefix** discriminates A22 as well. Its status stays `SSLM_INVALID_ARGUMENT` (the mapped-count
  check), but with A22 removed, adopt revokes the adopter's schema bind (`sslm_abi.cpp:2862`) before that refusal
  (`:2893`), so the cell's "bind not revoked" clause turns red (`build/e2-ruleb-cells.md` §8 F1; reproduced on the
  post-M1 tree). Only the budget legs are non-discriminating. Whether adopt should revoke before refusing is a code-review
  matter, not a cell change.
- 8.14 [U2] *(Rev 1.13; §5.11 (a), (b); rev 1.14 rewrote it: re-gate review S2, audit S4–S6.)* **The cache on
  paging's ledger.** In budget mode, with paging's ledger and zero slack outside C and A (paging's form 3); the
  cache's charges are visible through the region-C seam and graded by paging's form 1 at every reading, taken with the
  lanes parked between jobs behind the barrier (§8).
  - *The reachable construction (rev 1.18.6; box-fixes review, plan note).* Builds cannot fill C exactly.
    - **Why not.** Until it freezes, a build is charged `min(ceil(L/B) + 1, P_blk)`, so the last build of a
      construction peaks one page above where it settles. The one exception is a capped build with
      `ceil(L/B) = P_blk`, which fills all of C by itself and so cannot serve a construction of several entries.
    - **How each leg below reads.** Wherever a leg says "C is filled exactly", "C full" at setup or "free charge 0",
      the cell builds C to settle with **one page free**, and each request it issues is **one page larger** than
      the leg states.
    - **What that leaves unchanged.** Each request takes the free page first (§5.11 (a): free charge, then unclaimed
      refunds, then evictions), and then claims exactly the refunds the leg names, so every gap is unchanged.
      - During the required window, C is full (`Free() == 0`), with E's `k` still carried.
      - The overdraw from the refund at the eviction decision is still `k − f = 1`, because the free page sits in the
        request's charge, not in the slack.
      - Form 3's zero slack outside C and A still holds.
      - The waiters leg's needs (4 and 3 in the built form) still meet re-check §4's condition.
    - **The readings, outcomes and mutants are unchanged.** The review's sweep kills the three mutants named here
      (refunds not earmarked; the loop ignores pending refunds; a loop that cannot claim unclaimed refunds) on the
      cells as built.
  - *Region C.* The test computes `ceil(PrefixCacheBudgetBytes / block bytes) × P_blk` itself, at a
    `PrefixCacheBudgetBytes` that is **not** a multiple of the block bytes, and requires the pool's C to equal it
    (not `CacheRegionPages()`'s own figure, which is the function under test). Mutant: `floor` for `ceil` → red.
  - *The eviction leg.* Entry E has `L = 2B + 1`, so the cache charges C `k = 3` pages and paging pre-charges A the
    `f = 2` full pages its adopters share. Two budget sequences adopt E through `AdoptPrefix`. C is filled exactly,
    with E the least recently used and at least one newer unpinned entry F. A cache build is then requested whose
    charge fits only with E's refund. A test-side batch gate holds dispatch of E's `sslm_prefix_release` job (as
    8.10 (c) holds its prefill and freeze jobs), so a build admitted early would be dispatched and draw first. One
    required reading is taken **between the eviction decision and the release's apply**: E is out of the map, C still
    carries E's `k`, the build is a waiter that has drawn nothing (its begin job not dispatched), and form 1 is
    clean. If that window is absent, the cell fails; a later reading cannot substitute. The gate is then released.
    At the apply pass of E's release the build is admitted with its charge complete (the ledger's total unchanged by
    the transfer, the build's charge equal to its `R(L)`), and it draws. Asserted: **exactly one** eviction (the
    eviction diagnostic), and F is still a `Memory` hit after the build is admitted; zero `LedgerFaults`; paging's
    probe readings match at every reading.
  - *Refused.* With nothing evictable (every entry leased) and no eviction pending, the same request resolves
    `BudgetExhausted`, nothing drawn.
  - *Waiters (§5.11 (a)).* Two requests for distinct keys in one tick, with free charge 0 and two unpinned entries of
    equal charge, each request needing exactly that charge (*rev 1.15*, re-check §4: the second request's need must
    not exceed the first victim's charge, and with no surplus nothing is unclaimed): each claims its own victim (two
    evictions), both are admitted at their own victim's apply, and the ledger never exceeds the budget.
  - *Unclaimed refunds (rev 1.15, re-check R2).* (i) **E/W/X:** C full, E (3 pages) the only unleased entry, both
    releases gated. W needs 2: it evicts E, claims 2 of E's refund and waits; 1 is left unclaimed. X, in the same tick,
    needs 1: it claims that 1 and waits, with no further eviction. At E's apply both are admitted, W with 2 and X with
    1; the ledger never exceeds the budget, and exactly one eviction occurs. (ii) **Orphaned refund:** C full, E (3
    pages) the only unleased entry. Y needs 4: it evicts E, still falls short, and resolves `BudgetExhausted`; E's
    eviction stands and its refund is unclaimed. Z, before E's apply, needs 3: it claims E's refund and waits, and is
    admitted at E's apply. Both are the approved rule's outcomes (§5.11 (a), *Why the principal's decision is
    unchanged*). A second request for a waiting key attaches to the waiter: one build, the prefill-count seam up by one. A
    re-`Configure()` while a request waits: it resolves by name as 1.7's queued request does, nothing is drawn, and
    the pool destroy returns `SSLM_OK`. *Rev 1.16 (strike L1):* the waited-on release's completion is posted after the
    `Configure()`; at the next apply pass it is dropped (its generation is stale), and the rebuilt ledger, read after
    that pass, holds exactly the new configuration's charges, with free charge never above C. Mutant: a stale
    completion applied → free charge above C (the strike's construction: C = 8, V of 5 evicted, W claiming 4, free
    9) → red.
  - *Legacy mode.* The same eviction-then-build sequence keeps 8.10's admission reading.
  - *A drop is an unclaimed pending refund* (*rev 1.18*, PC-16 (a); the model's m2 as a cell). E as in the eviction
    leg (`L = 2B + 1`, `k = 3`, `f = 2`, two budget adopters, then every lease returned), C full, E's release gated.
    `DropPrefix(E)` resolves `Dropped`: E is out of the map, C still carries E's `k`, no request claims it, and form 1
    is clean at a required reading between the drop's decision and its release's apply. A request whose charge fits
    only with E's refund, issued in that window, claims it and waits (it draws nothing, its begin job not
    dispatched); at E's apply it is admitted with its charge complete. Zero `LedgerFaults`, and the eviction
    diagnostic is unchanged by the drop (drops are counted apart). Mutant: the drop refunded at its decision → the
    request is admitted at once and draws while the adopters' and E's pages are still held → physical above the
    charges by `k − f = 1` under zero slack → red as the eviction leg's first mutant. *Rev 1.18.1 (audit G12):* the
    same leg in **legacy mode**, read by 8.10's admission reading; mutant: refund at the decision → the reading
    exceeds the budget → red.
  - *Nested entries [U6] (rev 1.16, strike F1 and F2),* in budget mode, releases gated, authored in U6's red commit:
    - **(F1) a waiter pins its parent.** C = 12; LRU order V (3), P (4), Q (3), and L (2) leased; free 0. W requests a
      child of P (charge 3): it evicts V, claims 3 and waits, and P is pinned from that decision. X, in the same tick,
      requests a root of charge 6: Q is the only evictable entry, so X evicts Q and is refused `BudgetExhausted`, as
      the approved rule refuses it. At V's apply W is admitted, and its build or restore runs against a live P (the
      release-counting seam shows P not released). Mutant: no pin while waiting (rev 1.15's rule) → X also evicts P
      and waits, and W's job starts against a released P → red on X's disposition, on the eviction list, and on
      the release-counting seam.
    - **(F2) an ending ghost is claimable.** C = 10; P (4 full pages) is evicted while its child D (2 private pages,
      sharing P's 4) stays cached, so P is a ghost of 4, and P's release applies; E (4) is leased. (i) X needs 6: it
      evicts D, claims D's 2 and the ghost's 4 from D's pending refund, waits, and is admitted at D's apply. (ii) In a
      fresh run, X needs 2 and Y needs 4 in one tick: X evicts D and claims 2, Y claims the remaining 4, both wait,
      and the cache's physical pages (the fill probe, behind the barrier) never exceed C before D's apply; zero
      `LedgerFaults`. Mutants: the ghost refunded at D's decision → in (ii) X is admitted at once and draws while D's
      handle still maps P's pages → physical above C, a paging exhaustion or `LedgerFaults` → red; the ghost held but
      unclaimable → in (i) X is refused → red.
    - **(Re-strike R1, rev 1.17) siblings, releases out of order.** C = 10; P (4 full pages) shared by cached
      children D1 and D2 (1 private page each); E (4) leased. P is evicted (a ghost of 4) and its release applies. X
      needs 1: it evicts D1, claims 1 and waits. Y needs 5: it evicts D2, which ends the ghost; Y claims D2's 1 and the
      ghost's 4 and waits. **D1's release is gated behind D2's**, so D2's applies first. At that apply Y is **not**
      admitted (the ghost's family still holds D1's release) and the cache's physical pages read by the fill probe
      stay at most C; at D1's apply both X and Y are admitted. Mutant: the ghost refunded at the ending descendant's
      apply (rev 1.16's rule) → Y is admitted at D2's apply and draws 5 while D1's handle still maps P's 4 → physical
      14 > C 10, a paging exhaustion or `LedgerFaults` → red.
    - **(Re-strike R1, rev 1.17) a chain.** C = 8; a chain e0 ← e1 ← e2 ← e3 of one page each, with e1 and L (4)
      leased, LRU order e0, e2, e3. A request needing 5 evicts e0 (a ghost), e2 (a ghost) and e3 (which ends e2's
      ghost), and is refused. e3's release applies before e2's own. A root request Z of 2 then waits (e2's ghost
      waits on e2's own release) and is admitted at e2's apply; physical never above C. Mutant: the ghost refunded at
      e3's apply → Z admitted at once → physical 9 > C 8 → red.
    - **(Rev 1.17) a parent evicted after its child, not a ghost.** C = 7; e1 (3, child of e0) LRU, then e0 (3); L (1)
      leased. W needs 4: it evicts e1 (claims 3), then e0, which is no ghost (e1 is no longer cached) but whose pages
      e1's handle still maps (claims 1; 2 unclaimed). X needs 2 and claims e0's remaining 2. e0's release applies
      first: nobody is admitted; at e1's apply both are. Mutant: family sets on ghosts only → X admitted at e0's
      apply → physical 9 > C 7 → red. (The re-strike's generators found this shape once the ghost rule was fixed:
      23, 7 and 23 overdraws on its seeds, `fold_runs.out` E.)
    - **(Re-strike R2, rev 1.17) the own-parent reading.** C = 7; P (4) unpinned, L (3) leased. A nested **restore**
      of a child of P charged 4 evicts P, is admitted, and runs with a null hint (`*out_shared_pages == 0`, private);
      mutant: the restore's loop skips its parent → refused → red. A nested **build** of the same shape is refused
      `BudgetExhausted`, P stays resident and unreleased (the release-counting seam); mutant: the build's loop evicts
      its parent → its `begin_from` starts against a released P → red on the seam (and, in the model, physical 11 > 7).
    - **(Rev 1.18, PC-15 Q6; the U6 draft's R36) region A for a nested parent.** P (2 full pages) with cached child D;
      a budget sequence adopts D through `AdoptPrefix`, so paging pre-charges A with D's full pages, P's included. P
      is evicted (a ghost on C) and its release applies; D stays. Form 1 at the required reading counts P's pages on
      both C (the ghost) and A (the adopter's dependency), and after D is evicted and its release applies, the
      adopter still decodes its twin's tokens while C has refunded the ghost and A still carries the pages. Mutant: A
      not pre-charged for the parent's pages through the child → after the refund the adopter holds uncharged pages →
      form 1 red.
    - **(Rev 1.18, PC-16 (a)) dropping a pinned parent.** A nested waiter W pins P (the F1 construction); `DropPrefix(P)`
      resolves `DropDeferred`, W's job runs against a live P (the release-counting seam), and P is evicted at the
      apply pass after W's job completes and P is unpinned, with its family's refund rule applying as for any
      eviction. Mutant: the drop releases P at once → W's job starts against a released P → red on the seam.
    - **(Rev 1.18.1, audit G13 (b)) dropping an unpinned parent.** P (4 full pages) with a cached child D, nothing
      pinned. `DropPrefix(P)` reads `Dropped`; D is still a `Memory` hit; admitting up to the budget behind the
      barrier keeps the fill probe within C. Mutants: the drop refunds P's whole charge with no ghost → physical above
      C → red; the drop cascades to D → D's hit is lost → red.
    - **(Rev 1.18.1, audit G13 (c)) a nested warm pins its parent.** The F1 construction with `WarmPrefix` as the
      nested waiter. Mutant: a warm waiter that does not pin its parent → its job starts against a released P → red on
      the release-counting seam.

  Mutants, each red with the release gated:
  - C refunded at the eviction decision → the build draws while E's pages are still held; physical use exceeds the
    charges by `k − f = 1` page under zero slack → engine exhaustion in a paging request, a `LedgerFaults` count
    above zero, or form 1 red at the required reading. (At `L ≡ 0 (mod B)`, `k = f` and this mutant would survive,
    which is why `L = 2B + 1`.)
  - `AdoptPrefix` bypasses the subsystem's one adopt entry point, so A is never pre-charged → after E's refund the
    adopters still hold `f = 2` uncharged pages → red the same way.
  - The eviction loop ignores the request's pending refunds → it evicts F too → red on the one-eviction assertion
    and F's `Memory` hit.
  - The waiters' refunds not earmarked (both count the first victim's) → the second request stops without evicting →
    red on the two-eviction assertion. This holds because the second request's need equals the first victim's
    charge (*rev 1.15*: were the victim smaller, the mutant would evict a second victim too and survive the count).
  - A loop that cannot claim unclaimed pending refunds (rev 1.14's rule) → X is refused `BudgetExhausted` in (i), and
    Z in (ii) → red (*rev 1.15*).

  Paging's own [cache] cells (8.3, 8.4, 8.5 and the cache kinds of 1.2, 1.3, 5 and 11) run beside it at U2's gate.
- 8.15 [U2, box] *(Rev 1.18; PC-15 Q5, §5.11 (b) *Adopt and hint*; the U2 draft's V9.)* **A leased cache handle as a
  restore hint.** In budget mode, parent entry P leased; a nested restore of its child's blob is requested with P's
  lease as the hint: the restore shares `floor(origin/B)` of P's pages (`*out_shared_pages`, bookkeeping, and the
  fill probe's draw count `ceil(L/B) − s`), P stays pinned by its lease for the restore's duration, and paging's A is
  pre-charged for the shared pages as paging §3.2 specifies (form 1 clean). The child's adopters equal the twin's
  tokens. The same request with P's lease already returned runs private (`s = 0`). Mutant: the hint ignored (the
  draft's behaviour) → `s = 0` and the draw count reads `ceil(L/B)` → red. *Rev 1.18.1 (audit G18):* mutant: the
  hint wired but A not pre-charged for the shared pages → form 1 red.
- 8.16 [F2] *(Rev 1.18.1; audit G5 (b); §4 item 2.)* **A restored GPU prefix's derived state, against its twin.** On
  GPU prefixes restored at `L = 0` and at `L > 0`, and their prefill twins: `sslm_gpu_prefix_set_schema` is admitted
  iff `L = 0`; an adopter is ready for logits iff `L > 0`; every verb's disposition equals the twin's (the GPU twin of
  8.11). Mutant: bind permission derived as for `L > 0` → the `L = 0` set_schema is refused → red.
  *Rev 1.18.4 (the red suite's reading 15):* **8.16 also carries 6.8's bind-permission leg D.** Its `L = 0`
  prefill twin is built that leg's way: an empty prefix given a zero-count prefill, then frozen. A twin frozen with
  no prefill call at all would leave the history on which the freeze-time setting matters unexercised. So this cell
  grades the restored copy against the one history where the original and the copy used to differ. The outcome is
  unchanged (set_schema admitted iff `L = 0`, on both). A second mutant is added: the freeze-time setting removed
  (bind permission left as the zero-count prefill set it) → the twin's set_schema is refused while the restore's
  is admitted → red. 6.8's leg D stays. It grades the original against its own copy, which 8.16 does not.
- 8.17 [U2; U6 twin] *(Rev 1.18.1; plan review F10, audit G20.)* **A key re-created while its earlier instance is
  still pending.** Budget mode, E's release gated: `DropPrefix(E)`, then `RequestPrefix(E)`. The request misses and
  builds a new instance, admitted or waiting on E's own refund; at the required reading (lanes parked, behind the
  barrier) the ledger holds the old pending `k` plus the new charge and form 1 is clean; at the old release's apply
  exactly `k` refunds, and the new instance stays resident and a `Memory` hit. [U6] twin: E is a ghost parent with a
  cached child. Mutant: instances keyed by key → the new charge overwrites the pending one → the ledger is below
  physical by `k` → red on form 1 and the fill probe.

**9. Persistence round-trip and version evolution — applicable.**
- 9.1 [E2] Round trip per 6.4, and a blob written by the cloud build restored on the box build (4.3).
- 9.2 [E2] A root blob with a hint: hint ignored (X2).
- 9.3 [E3] Nested with a matching hint: `s = floor(origin/B)`; diagnostic half reads `*out_shared_pages` (its
  subject).
- 9.4 [E2] Evolution: a blob from another producer refused (X5); an unknown magic `SPX2` refused
  `SSLM_INVALID_ARGUMENT`; `SSB*` and `SLM*` refused by the prefix verb and `SPX1` by the sequence verbs.
- 9.5 [F4] Across releases: a blob written by 1.14.0 is refused by 1.16.0 (*rev 1.18*: rev 1.17 said 1.15.0) with `SSLM_RESTORE_PRODUCER_MISMATCH` (run
  at F4 on both binaries; the pinned 1.14.0 blob of §8).
- 9.6 [U2] Plugin wrapper evolution: wrapper version 0 and 2 are misses; a blob built under another Layer-1 pin is a
  miss.
- 9.7 [U3] Cooked set rebuilt after a pin change serves hits again.
- 9.8 [E2, E3] *(Rev 1.13; re-anchor F5, X13.)* **One blob, both pool kinds.** Each model of 4.7's list that has
  `cap_pages ≥ 3`, one at a time: every blob, pool and adopter in a run belongs to that one model (*rev 1.14*,
  audit N2). Each is its own unpinned mapping and the cell asserts `sslm_kv_page_positions == PageRuleB(D, cap,
  elem)`, computed in the test, and **rule B < cap**, before anything else (§7 preamble; review S4, audit S1). `L` ∈ {0, 1, B, B+1, cap−1, cap}. A prefix built by prefill in a rule-B page
  pool is saved; its blob is restored into a **contiguous** pool, and that restored prefix is saved again and
  restored back into a page pool. Each restore keeps (§3.6); each re-save is byte-identical to the first blob. The
  page-pool restore draws `ceil(L/B)` pages (two-sided fill probe). The contiguous restore draws `ceil(L/cap)`
  pages, 0 or 1, read by a **single-page count**: in a contiguous pool of `N ≥ 3` blocks every budget-1 create
  reserves exactly one page (`ReservationPagesIn = 1`, S30 (d)), so `N − ceil(L/cap)` creates are admitted and one
  more is refused `SSLM_KV_POOL_EXHAUSTED`, on a state the cell rebuilds (PKV's `ProbeSplit` needs
  `cap_pages ≥ 3` and cannot read a contiguous pool, S29). Adopters of each mode in each pool kind decode 64 greedy
  tokens equal to adopters of the prefill twin built in that pool kind. Mutant: the restore takes B from the model's
  geometry (`ComputePageGeometry(model)`) rather than from the pool → at `L ≥ B + 1` the contiguous restore asks for
  `ceil(L/B) ≥ 2` pages of a pool whose page is the cap: the holder create is refused (a status other than
  `SSLM_OK`), or a reserve page is kept, which the single-page count reads as `N − 2` creates admitted rather than
  `N − 1` → red (*rev 1.14*, audit N2: the deciding operand is the status or the single-page count, and only at
  `L ≥ B + 1`; at `L ∈ {0, 1, B}` the mutant is equivalent). This is the matrix row X13's "keeps".
  *Memory (rev 1.14, audit N9):* on the synthetic 1.5B shape a contiguous block is 448 MiB, and the single-page count
  needs `N ≥ 3`, so about 1.3 GiB per contiguous pool; §9 E2 budgets it.
  - *Nested leg [E3] (rev 1.14; re-gate audit B1).* At each model's rule B (`B_r`), a nested child is built by
    `begin_from` in a rule-B page pool, with `origin ∈ {B_r, B_r + 1}` and `L = 2·B_r + 1`, and saved. Its blob is
    restored into a **contiguous** pool:
    - (i) with a hint that is a prefill-built parent of length `origin` in **that** contiguous pool, matching in every
      other respect: `s = floor(origin/cap) = 0`, `*out_shared_pages = 0`, private; the single-page count reads
      `ceil(L/cap) = 1` drawn; the re-save is byte-identical to the blob; adopters of each mode equal the adopters of
      the contiguous twin (the same child built by `begin_from` from that parent in the contiguous pool); and the
      parent's own re-save is byte-identical to its first save (nothing scattered into its page);
    - (ii) with the parent in a **page** pool: condition (d) fails, `s = 0`, private, the same counts;
    - (iii) the edge `origin = L = cap` (`begin_from` then freeze with no prefill, as 4.2's last case), with a matching
      contiguous parent of length cap: `s = 1`, `*out_shared_pages = 1`, the single-page count reads 0 drawn.

    Mutant: the hint's `s`, or (f)'s span, computed from the model's rule B rather than the pool's → at `origin = B_r`
    in (i) `s = 1`: the child shares the parent's single cap page and draws `ceil(L/cap) − 1 = 0`, so the single-page
    count reads one more create admitted, and the rows `[B_r, L)` are scattered into the shared page or skipped, so
    the parent's or the child's re-save differs → red. This is X13's nested case.
- 9.9 [E5] *(Rev 1.13; PC-13; rewritten in rev 1.18.1, audit G14 (c).)* **1.13.1's CPU sequence saves on 1.14.0.**
  The predecessor is **v1.13.1** (`d697927`, tag object `034a76a`), the version the paging release vendors (rev 1.18's
  retarget; rev 1.17 named `v1.13.0`, `a5f3ad1`). Captured once on a `v1.13.1` build (`d697927`) and pinned with
  their SHA-256s, with the decode each continues to on that build: `SSB5` saves of a whole_reserve sequence, and
  `SSB6` saves of budget sequences (a root, an adopter of a nested prefix with `origin > 0`, and one mid-turn), each
  with and without a bound schema. *Rev 1.18.4 (the red suite's reading 1, closed):* "mid-turn" means **mid-token**:
  saved between the layer calls of one decode step, as the strike probe this clause came from defines it ("one
  layer of a two-layer decode", `reviews/strike-rev1p15-probes/pc13_cross.cpp:122`). The pinned mid-turn blobs carry
  `layer_index = 1`, and every resting case carries 0. On the 1.14.0 release tree (`main` after M2, §9 E5): each restores through
  `sslm_seq_restore` (and the `SSB6` ones through `sslm_seq_restore_shared` with a matching hint), re-saves
  byte-identical to its pinned bytes, and decodes 64 tokens equal to the pinned 1.13.1 continuation. *Rev 1.18.4
  (the red suite's reading 2, and the suite audit's N1): the bound cases are graded in outcome form.* pkv_def's one
  schema admits token 0 and then dead-ends, so "64 tokens" cannot hold for a bound case. Each of the 64 steps
  records its outcome code: the token, −2 where the dead-ended schema admits nothing, or −1000 minus the status for
  a failing step. The codes must equal the 1.13.1 capture's. The capture reads `0, −2, −2, …` for three bound cases
  and 64 × `−2` for the bound mid-turn case. Bound cases save after the prompt, with no decode before the save. So
  for those four cases the continuation grades at most one token, and the byte-identical re-save is what carries
  the cell. A fixture whose schema admits a longer walk would give that leg back its power, and none is owed. The
  unbound cases keep "64 tokens". The oracle is
  the 1.13.1 build's own output, so no 1.14.0 code grades it. Mutants: one `SSB6` header field moved in 1.14.0's
  writer → red; the reader's exact-size rule **tightened** so a 1.13.1 save's size is refused → red (*rev 1.14*,
  audit N5: a relaxed rule breaks no PC-13 promise and is not this cell's mutant). With the 1.13.0 row dropped
  (PC-15 Q8), no 1.13.0 capture is owed (audit G14 (b) is moot).
- 9.10 [E5] *(Rev 1.13; PC-13; box.)* The GPU twin of 9.9: `SLM5` (dedicated) and `SLM6` (pooled, including a save
  between an adopt and the first finish) captured on a `v1.13.0` build on each certified GPU, restored on the 1.14.0
  release build on the same device, re-saved byte-identical, and continued to the pinned tokens. 1.14.0 changes no
  GPU source file (§3.8), so for the save formats this is a regression proof, graded the same way. *Rev 1.16 (strike
  L2):* it is not a proof that GPU behaviour is unchanged: PC-12's refusal in `schema_masks.h`, which
  `src/gpu/gpu_1p0.cpp:70` includes, changes which models the GPU maps. 9.10's captures use certified artifacts,
  which E0's audit found PC-12 admits (§8); a model PC-12 refuses is refused at map on both backends, before any
  restore. Rev 1.17 had F4 re-run 9.9 and 9.10 on 1.15.0 against the pinned 1.14.0 and 1.13.0 saves (§5.11 (c));
  superseded below. *Rev 1.18:* captured on `v1.13.1` (`d697927`) as 9.9 is.
  "1.14.0 changes no GPU source file" no longer holds (§3.8): the squash carries §4's GPU code, so this cell is now
  the proof, not a regression check of an unchanged file set, that `SLM5` and `SLM6` are unchanged. F4 re-runs 9.9
  and 9.10 on **1.16.0** against the pinned 1.14.0 and 1.13.1 saves (and 1.15.0's if a plugin release vendors it).
- 9.11 [U2] *(Rev 1.13; PC-13; rewritten in rev 1.18.1 for the one-row table, PC-15 Q8 and audit G14 (a).)* **The save
  boundary removed.** Wrappers written by the paging release (plugin 1.2.0, vendoring 1.13.1: tag `v1.13.1`, commit
  `d697927`) in legacy and budget mode on the CPU, and in dedicated and pooled mode on the GPU [U2 box leg], are
  restored by the CPU-cache release: each is accepted, nothing is refused `Layer1Mismatch`, and its continuation
  equals the paging release's recorded one. A wrapper written by a 1.0.x/1.1.x build (1.11.0 tag) is still refused
  `Layer1Mismatch` with nothing drawn; a wrapper tagged 1.13.0 (any commit) is refused `Layer1Mismatch` (the row is
  dropped); and 1.13.1-tagged wrappers whose payload magic is outside the four are refused as before, one case each
  for `SSB4`, `SSB3`, `SSB2` and `SPX1` (*rev 1.14*, audit N6). *Tag and commit (rev 1.14, audit S6; rev 1.18.1,
  audit G14 (a)), conditional on U2's read of the wrapper's commit field (S14, §14):* where the wrapper records a
  commit, a wrapper carrying the 1.13.1 tag with a commit other than `d697927` (for example `9e235965`) is refused
  `Layer1Mismatch`, nothing drawn. Mutants: the compatibility entry removed → the 1.13.1 cases red; the entry widened
  to any tag → the 1.11.0 and 1.13.0 cases red; the entry widened to `SSB4`, `SSB3`, `SSB2` or `SPX1` → that magic's
  case red; the commit comparison dropped from the entry → the foreign-commit case red. If U2 finds the wrapper
  records no commit, the foreign-commit case is recorded as not applicable, with its reason, and the entry matches by
  tag alone (§14).
- 9.12 [U3] *(Rev 1.18; PC-16 (b).)* **Prefix Set persistence.** A `USuperSLMPrefixSet` saved before the text fields
  existed (a pinned asset from the U3 draft, token entries only) loads with every entry `Input == Tokens`, its
  ids and built bytes unchanged, and serves as before. A set with `Text` entries survives save, reload and cook with
  its `Text`, stored ids, text hash and model hash unchanged, and its cooked entries are hits. Mutant: the new fields
  serialized without a default (an old asset loads as `Text` with empty text) → red.

**10. Functional achievement — applicable (the feature oracle).**
- 10.1 [E4] Real 0.5B artifact on the box: 10 prefixes of 128–2,000 tokens built by prefill and saved to files;
  process exits; a new process restores all 10 from the files; 50 adopters (**both adopter modes**, budget
  and legacy; on the DGC1 artifact, both greedy and damped-greedy decode modes) decode 64 tokens each;
  every stream equals the pinned reference streams of the same adopters over prefill-built prefixes (§8), and no
  prefix prefill ran in the second process (`sslm_test_only_prefix_prefill_call_count` reads 0, §8).
  *Geometry (rev 1.13, re-anchor F2; applies to 10.1–10.3; rev 1.18.4, suite audit G10 and the red suite's reading
  5: rev 1.14 said it applies to 4.3 and that "`:181` is 4.3's", which is stale. At `3b6631f`, `:181` is inside
  `Timing103`, so it is 10.3's. 4.3, `CrossCompiler43`, maps pkv_def through `GetFixture` at the pin, has no real
  fixture, and checks byte identity across compilers, which B does not affect):* the real cells run at the **rule B, unpinned**. The
  geometry assertion each makes (`e4_real.cpp:60`, `:181`, `:269`, `:322` at `3b6631f`) is re-derived: it requires
  `sslm_kv_page_size == B·bytes_per_token`, with B computed in the test from the rule of S30 (a) and the artifact's
  `D`, cap and element width: **128 at 0.5B** and **64 at 1.5B**, never read back from the engine as its own
  oracle. As written at `3b6631f` they require `16·bytes_per_token` and go red on the real artifacts. They are
  **re-derived, not pinned**, because 10.x is the feature oracle: it must run the geometry a shipped game runs, and
  10.3's speed bar must time the page size the rule gives the certification artifact (768 KiB at 0.5B, 896 KiB at
  1.5B; *rev 1.15:* the plugin ships the cap-4,096 0.5B artifact, whose rule page is 384 KiB, §14). The pin is a test-only seam
  of the engine's test library that no game can call, so a pinned 10.x would grade a configuration nobody ships,
  and the pinned B = 16 cells already cover those shapes.
  *Rev 1.14 (re-gate review S3, audit S2): the four assertions are not the only B-dependent sites.* The real
  fixtures are unpinned with cap 32,768, and `superslm_precache_e1` is not compiled with `PKV_FIXTURE_B_FROM_RULE`,
  so `Fixture::B()` returns 16 (`pkv_common.h:236-242`) and every test-side operand follows it. So E4:
  - sets each real E4 fixture's test-side `pinned` to the test-written rule B (`fx.pinned`, and `wf.pinned` for the
    1.5B worst case, in `Real101`, `Real102`, `Timing103` and 6.6's rule-B leg; *rev 1.18.4:* not 4.3, which has no
    real fixture), which sets only
    `Fixture::B()` and makes no engine call, and asserts it equal to `sslm_kv_page_positions` before anything else.
    *Rev 1.15 (re-check R1):* rev 1.14 compiled the E4 cells with `PKV_FIXTURE_B_FROM_RULE` and said the pinned
    cells sharing the binary were unchanged because `pinned > 0` is checked first. That is false: `GetFixture`'s pin
    calls `KvPagesTestPinPagePositions(f->model, 16)` and leaves `f->pinned` at 0 (`pkv_common.h:296-310`; only
    `PinFixtureB` sets it), so target-wide every `GetFixture`-pinned cell's `B()` would read the rule (128 on pkv_def)
    against an engine pinned to 16 and go red, `Parallel66` included; and defined for `e4_real.cpp` alone, the inline
    `Fixture::B()`, `CapPages()`, `R()` and `PageBytes()` would have two definitions in one program (an ODR
    violation; `e4_real.cpp` is linked into `superslm_precache_e1`, `precache.cmake:3-5`). Of the re-check's two fixes
    this plan takes `fx.pinned`, not a separate target: it touches only the real fixtures, which already build their
    `Fixture` by hand (`e4_real.cpp:57-59`); `real_two_process` keeps running the one binary it names
    (`precache.cmake:59-61`), so no runner, fixture setup or reference wiring moves; and the pinned cells in the
    binary are untouched by construction. A separate target would duplicate that wiring for no added coverage. The
    macro is never defined in `superslm_precache_e1`, target-wide or per file;
  - sizes its pools at the rule B: `e4_real.cpp:70-72` (`CapPages()`, `R()`), `:272` and `:330-337`. At B = 16
    those pools are about 2,180 pages of 768 KiB at 0.5B and 2,052 pages of 896 KiB at 1.5B: oversized, not wrong,
    but not the rule-B configuration 10.3 times. *Rev 1.18.4 (suite audit G10):* `:272` is
    `PagePool worst_pool(wf.model, wf.CapPages()+4)`, the 1.5B worst case, which follows the rule B through
    `CapPages()`. `Timing103`'s own `PagePool pool(fx.model, 300)` (`:184`) is a constant page count, is not a rule-B
    site, and is left unchanged: at B = 128, 300 pages are more bytes, not fewer, so it is not a defect;
  - re-derives 10.2's share assertion, `e4_real.cpp:388` (`shared == world_ids.size()/16`, 62), to `s = 7`
    (`floor(1000/128)`), with the draw asserted below.
- 10.2 [E4] Same with the PKV cohort's nested shape (world 1,000 → 10 personas of 200): children restored against
  the restored world share its pages and their adopters match. *Rev 1.13:* at the real 0.5B rule `B = 128`, each
  child shares `s = 7` pages and draws 3 (§3.5). *Rev 1.14 (audit S2):* both are asserted: `*out_shared_pages == 7`
  (`:388` re-derived), and `ceil(1200/128) − 7 = 3` pages drawn, read by the fill probe around each child's restore.
  *Rev 1.16 (strike L3):* the cell computes both from the files it reads, never from literals:
  `s = floor(|world.ids| / B)` and the draw `ceil((|world.ids| + |personaNN.ids|) / B) − s`, with B the test-written
  rule B, as `:388` already computes from `world_ids.size()`. 7 and 3 are the values for the box's 1,000-token world
  and 200-token personas.
  Mutant: the restore's hint computing `s` at any B other than the pool's → red on both.
- 10.3 [E4] Speed, the reason the feature exists: restoring each prefix of 10.1 from memory, digest included, is at
  least **1,000×** faster than building it by prefill on the same box. *Derivation (rev 1.1; executed arithmetic,
  rates assumed):* prefill ~19 ms per token (spike, 0.5B), so ~19 s for 1,000 tokens. Restore ≈ the digest over 5.86
  MiB plus the scatter: ~15.9 ms with a portable SHA-256 at an assumed 400 MB/s (~1,197×), ~4.6 ms with SHA-NI at an
  assumed 1.5 GB/s (~4,123×). Both costs scale with `L`, so the ratio barely moves with length. The spike's cloud
  ratio without the digest was ~10,000×. Measured with resolving power; also recorded at the worst reachable setting
  (a cap-length prefix at 1.5B geometry) as ms and MiB/s for the plugin's cost model.
- 10.4 [F3] GPU: 10.1–10.2 on both GPUs, and a GPU speed bar (*rev 1.1*: rev 1 carried the CPU's 1,000× over with no
  derivation) of **≥ 100×** restore-from-memory over batched GPU prefill, at 1.5B on both GPUs and at 0.5B on the
  2080 SUPER, at 128 and 1,000 tokens. *Derivation (executed arithmetic; rates assumed, figures cited):*
  - prefill (S22): 1.5B ~16.1 ms per token on the 2080 SUPER and ~13.6 ms on the 7900 XTX. 0.5B ~8.4 ms per token
    is *inferred*: batched prefill runs at about the decode rate at 1.5B, so the spike's 8.4 ms per token for 0.5B
    decode stands in for 0.5B prefill. *Rev 1.2:* rev 1.1 divided it by 7.19, which is the speed-up over the old
    per-token prefill path, not over decode. No 0.5B prefill figure is published.
  - restore ≈ digest (SHA-NI, assumed 1.5 GB/s) + host-mirror write (assumed 12 GB/s) + upload (assumed 10 GB/s)
    + a 0.5 ms fence.
  - predicted ratios at 128 / 1,000 tokens: 1.5B ~1,002× / ~1,272× on the 2080 SUPER and ~846× / ~1,073× on the
    7900 XTX; 0.5B ~919× / ~1,468× on the 2080 SUPER. Without the digest, 1.5B on the 7900 XTX would be ~2,083× /
    ~4,352×.
  - The binding case is 1.5B on the 7900 XTX at 128 tokens (~846×), about 8.5× above the bar on assumed rates.
    Why 100× rather than 1,000×: 1,000× would miss that case, and the purpose is seconds to milliseconds. 100× turns
    a ~8.4 s 0.5B prefill of 1,000 tokens into ≤ 84 ms.
  - If the box misses it, R1b names the folds in order.
- 10.5 [U4] Plugin, packaged game on the box: a level with 10 cooked prefixes loads; every request is a `Cooked`
  hit (source diagnostic), no prefix prefill runs (`sslm_test_only_prefix_prefill_call_count`), adopters' tokens equal the recompute run's, and the game thread's tick
  cost stays within the plugin's existing tick bound. Then the budget is set to 5 entries and the level is
  re-entered: evicted entries return from the cooked tier, tokens unchanged.
- 10.6 [U4] Disk tier on: first run recomputes and writes back; second run is all `Disk` hits with equal tokens.
- 10.7 [U4] *(Rev 1.18; PC-16.)* Plugin, packaged game on the box: at a level load the game warms its next level's
  10 prefixes (`WarmPrefix`, from a set with `Text` entries and `Tokens` entries); each warm resolves `Resident`, and
  every later `RequestPrefix` at the level's start is a `Memory` hit with no prefix prefill in that window
  (`sslm_test_only_prefix_prefill_call_count`) and the game thread's tick cost within the plugin's bound. At the
  level's exit the game drops them (`Memory`); the next level's requests are served from the cooked tier, with
  tokens equal to the recompute run's. *Rev 1.18.1 (audit G16):* the warms come from a cooked set and the prefill
  counter reads 0 across the warms themselves; after the drops apply, the resident-bytes diagnostic (the subject
  here) falls by the dropped entries' bytes; and the **10 dropped specs themselves** are re-requested: each source is
  `Cooked`, not `Memory`, and the restore count rises by 10. Mutant: a drop that does nothing → the re-requests are
  `Memory` hits and the resident bytes do not fall → red.

*The carried T-3060 F10–F13 portions at E4* (*rev 1.18.4*; suite audit G8). Their text is the coverage auditor's
E1 case file and its recheck in the Wizard records (`Claude/Mendeleev/t3060-precache-e1-suite-audit-2026-10-01.md`,
rows F10–F13, and `t3070-precache-e1-suite-recheck-2026-10-01.md`, rows F10–F13 and items A3 and A4), read with
`git grep` on `develop` and `origin/claude/project-thread-wt2dpx`, which agree. That text routes F10–F13 into E4.
The E4 portion of each, and the cell that carries it:
- **F10 → 4.3: three real compiler jobs.** The same `SPX1` fixture is produced by MSVC, gcc and clang jobs. Each
  digest is bound to its compiler, model hash, input ids, version and fixture revision before the bytes are
  compared. A renamed or copied receipt must not count as three measurements. A3 adds that each receipt is bound to
  an independently captured runner binary, hashed, and to the bytes its own job generated (not a length-only
  `runner_sha256`). F10's B cross-view half is E2's (7.12, §15.12) and 9.1's provenance half is E2's, so neither is
  E4's.
- **F11 → 10.1, 10.2: the two-process run.** An orchestrated capture process that exits, then a separate restore
  process. There are manifest hashes for the real blobs, the model and the prompts (21 blobs in E4's receipt), and
  the ctest merge-base fixture binds every reference cell (§8). Stale, missing or unlicensed populations are refused,
  with a positive graded count at E4's green. A4 adds that the reference header is parsed into exact, unique keys,
  so a missing, duplicate, misnamed or unknown identity field is refused, with a misnamed-key must-reject.
- **F12 → 10.3: the timing receipt.** The exact receipt identity of the independently commissioned timing
  instrument. Paired measurements at each of the ten lengths, enough to state a resolving interval for the 1,000×
  threshold, with an unresolved result refused. The box, geometry, thread count, bytes with the digest, and the
  cap-length 1.5B ms and MiB/s are recorded under that receipt. The debunker owns the timing must-accept and
  must-reject constructions.
- **F13 → 6.6: the hook ran.** The parallel hook's calls and tasks are counted and required to be above 0 while
  the 32 tokens are compared with serial decode, on the real artifact at E4. F13's 3.1, 3.2 and 6.3 halves are E2's
  and E3's.

**Authored state, reconciled at the engine's tip `8788a8ea`.**
- 4.3 writes and reads per-compiler receipts and refuses a copied receipt (`e4_real.cpp:212-240`).
- `precache_provenance.h` hashes the runner (`ShaFile`, `:34`).
- `precache_reference.h` refuses a missing, duplicate, misnamed or unknown identity field (`:131`). Its `Field`
  helper still locates keys with `find` (`:33-35`); the test author confirms that every identity field goes through
  the exact parse.
- 10.3 checks a 13-field `PC-TIMING-1` receipt and runs paired measurements (`:258-313`).
- 6.6 installs a counting hook (`:526`).

E4 runs each portion on built verbs, with its mutant or independent control, and the coverage auditor re-audits
discrimination (§9 E4). The test author maps each portion to its file and assertion in E4's receipt.

**11. Guard vitality — applicable.**
- 11.1 [E2] Each validation step of §3.4 shown able to fire in the shipping build configuration: its mutant (the
  check removed) turns its 2.x case red (digest 7.3, producer 7.6, `current_token` 7.11; one mutant per step 2–8,
  and one per clause of step 8, except the checked size arithmetic, which is defence in depth that no input can
  reach once `L` is bounded, §3.4; *rev 1.2*). *Erratum (§15.30, P1a):* step 8's `length < 0` clause is exempt
  too, for an equivalent reason: with it removed, a negative `L` still reaches `kDomain` through the checked size
  arithmetic, or else through the origin clause, which no origin satisfies when `L < 0`. No input can tell the
  mutant apart. *Rev 1.6 (D-SLM7984):* rev 1.5's exemption of restore's release on a
  refused share is withdrawn. That release is pinned by 5.11 through the restore-site fault slot, as PKV pins its
  own unreachable short-reserve branch through `sslm_pkv_test_only_drain_reserve`, and as 11.2 pins the save's
  unreachable check. So are the map refusal's release and status. One more branch is exempt, for the
  checked-arithmetic reason: `MapPagesFor`'s past-the-table refusal in §3.4 step 3, which no input reaches once
  step 8 bounds `L`, because the table has `ceil(cap/B)` entries (S26; *rev 1.13*, re-anchor F4: rev 1.12 said "and
  `B` divides the cap", which 1.13.0's partial last page makes false; cell 4.7 runs the partial-page case). The share check itself is PKV's module code (S26). A grep of
  `tests/paged-kv` at `5d5ca58` finds no cell that drives its maximum-refcount branch. 5.11's slot stands in for
  that check, so it pins the restore's handling of the refusal, not PKV's check, and this plan leaves PKV's branch
  as it is.
- 11.2 [E2] The save's `layer_index != 0` refusal: a test seam sets a frozen prefix's `layer_index` to 1; the save
  is refused. The check is a real branch, not an assert.
- 11.3 [E2] `check_version_header.py` must-reject and must-accept (7.7), registered as an instrument (§8).
  *Rev 1.1:* the constructions are authored by the debunker, not the builder, and add a must-reject for an
  unparsable `project(… VERSION …)` line (the check must fail, not pass vacuously).
- 11.4 [U2] The spec-compare on load: with the compare removed, 2.8's forced-collision case restores the wrong
  prefix and the cell goes red. *Rev 1.1:* the same for the memory tier's compare, with the collision forced by the
  same seam.
- 11.5 [F2] The GPU restore's `EntrySubmitGuard`: removing it turns 3.4 red (PKV §4.4's per-verb pin).
- 11.6 [E2, F2] *(Rev 1.3.)* PC-12's two parser checks fire in the shipping build configuration on both backends'
  model map (2.13's mutants), as real branches, not asserts.
- 11.7 [U2] *(Rev 1.18; PC-16 (a).)* The drop's pin check and its disk-delete serialization fire in the shipping
  build configuration: with each removed, 7.18 (the pin check) and 3.10 (the serialization) go red, as real branches,
  not asserts. The runtime's no-tokenize property (7.20) is a counter on a call site, not a guard, and is not listed
  here.
- 11.8 [F2] *(Rev 1.18.1; audit G5 (a); §4 item 1.)* **The GPU save's at-rest guard** (the GPU twin of 11.2). A test
  seam drives a frozen GPU prefix into each non-rest shape: `L > 0` below full depth; `L > 0` with `current_token ==
  −1`; `L > 0` with `current_token ≥ vocab`; `L = 0` with `current_token ≥ 0`. Each save returns
  `SSLM_GPU_PREFIX_REJECTED` with nothing written; both at-rest shapes, (a) and (b), save `SSLM_OK`. The check is a
  real branch in the shipping configuration, not an assert. Mutant: the at-rest check removed → red.

---

## 8. Instruments and references

- **Pinned references (independent of the code under test).** At E0, before any of this plan's code, from the
  build-ahead anchor **`5d5ca58`** (*rev 1.5*, on the principal's 2026-10-01 build-ahead call; rev 1.4 named the
  1.13.0 tag, else 1.12.0). The references are token streams of every adopter used in 6.2, 8.x and 10.1–10.2 over
  prefill-built prefixes, generated on the cloud synthetic shapes and on the real 0.5B and 1.5B artifacts on the box.
  The restore path cannot influence them. **D-SLM7997:** T-3022's files in
  `D:/_artifacts/SuperSLM/t3022-precache-refs/` prove only that the old capture reproduces; no §7 cell grades
  against them. The test author regenerates the population from T-3031 §5 on the unchanged anchor, including
  chat-template real-text retrieval prompts, page-unaligned lengths, live schema walks, adapter-off controls,
  a full computed anchor and input SHA-256s in every header, and the independently commissioned `SSB6`
  save-edit-restore discrimination record. The record meets T-3031 §5.4's per-defect thresholds and its SHA is
  carried by each reference header. The coverage auditor re-audits the population before E2's first grading.
  The loader refuses an old file, a header lacking those identities, or a population lacking the commissioned
  discrimination record. At E5, one 1.14.0 `SPX1` blob per geometry is pinned for 9.5.
  *The base and its re-diff (rev 1.6, D-SLM7988; rev 1.5 re-diffed only "before E5 relies on them", while E2–E4
  grade against them first):*
  - **E2, E3 and E4 build on `5d5ca58`**, and stay there until E5's merge of 1.13.0. *Rev 1.13:* that merge came
    early. The E2 build was merged with `v1.13.0` (`a5f3ad1`) at `6426a72` on 2026-10-15, under the paging-first
    ordering (paging UP-1), and E2–E4 now build on `v1.13.0` (branch `claude/precache-1p14-on-1p13`). The re-diff
    below ran at that merge: it is the re-anchor record `build/e2-reanchor-v1.13.0.md`, and its one §3 rule change
    (F5) is folded in rev 1.13 and re-gated (§13 item 13).
  - The re-diff trigger is **this plan's branch taking paged-KV code newer than `5d5ca58`**, at whatever step that
    happens, E5's merge included. Box timing runs moving the paged-KV branch's tip do not trigger it. Taking that
    code into this branch does. E0's re-anchor reading (§9 E0) is re-diffed against the new base before the next
    build step. A diff that changes a §3 or §4 rule is a fold plus a scoped plan review and coverage re-audit.
  - **Reference regeneration is triggered by every change to the recorded `git merge-base HEAD <line>`**, whether
    caused by newer paged-KV code, an unrelated commit, a line switch, or E5's tag merge. Before the next
    reference grading, record the new merge-base, regenerate and independently license the reference population
    on that anchor, and require a graded count above zero. This trigger is separate from the §3/§4 rule-change
    re-gate. An unchanged merge-base needs no regeneration even if the remote line tip moved.
  - **Structural half** (*rev 1.7*, D-SLM7994, refining D-SLM7988; T-3026 W-4, T-3027 F18). The reference file
    records its anchor SHA.
    - *Where it runs.* A ctest `FIXTURES_SETUP` step recomputes `git merge-base HEAD <line>` immediately before
      each run of the reference cells and writes the result, or a failure marker, to the build tree. Every
      reference cell has `FIXTURES_REQUIRED` on it; no configure-time value can be reused. `<line>` is a
      configure argument and its chosen ref is recorded in the build tree. Its default is
      `origin/claude/shared-prefix-kv-build-wj8q9c` until E5 switches it to the ref merged there (the
      1.13.0 tag). *Rev 1.13:* the tag was merged at E2 through the tag, not through `<line>`, which is the
      structural half's stated blind spot (*What it sees*, below): with `<line>` unchanged the merge-base still
      reads `5d5ca58`, and the loader would grade references cut on code that 1.13.0 has since changed (run-slot
      rotation, rule B, the softmax width cap). So the line switch moves from E5 to **E2, before E2's first
      reference grading**: `<line>` becomes `v1.13.0`, the recorded merge-base becomes `a5f3ad1`, and the
      population is regenerated and independently licensed there (§9 E2). *Rev 1.18 (PC-14):* `main` was re-rooted
      on 2026-10-12 into parentless snapshot commits, and the line lands there as one squash (M1). (*Rev 1.18.9, §15.28:* the "parentless" commits were a shallow clone's boundary, not roots.) (*Rev 1.18.2, the
      strike on rev 1.18.1, lesser plane 6:* rev 1.18 went on "so no ancestry joins the branch to `main`", which is
      false. `git merge-base 83a82c2 2e12a6e` is `a5f3ad1`, 1.13.0's release head. The branch took it in through its
      v1.13.0 merge `6426a72`, and `main` holds it through `0d1c378`. The squash is how PC-14 lands the line, not a
      consequence of missing ancestry. M1 is unaffected.) After M1, `<line>` is the `main` commit the squash lands on (`2e12a6e` at 2026-10-20, a commit on `d697927`; *rev 1.18.1*, plan review
      F13: rev 1.18 called it "1.13.1's snapshot", which it is not) and
      `HEAD` is `main`'s tip; the recorded merge-base changes, so the trigger above fires and the population is
      regenerated and independently licensed on that anchor before the next reference grading. A squash is one of
      the mechanisms the structural half does not see (*What it sees*, below), so M1's row names this regeneration
      as its own exit. *Rev 1.18.1 (plan review F2, audit G1; the conductor's ruling): the rule above fires once and
      never again* (that commit is an ancestor of every later `main` tip, so the merge-base stays on it while other
      plans' code lands underneath), and it is replaced. After M1, this plan's remaining work runs on a **post-M1
      branch** cut from `main` at the squash commit, and `<line>` is **`origin/main`**. Then `git merge-base HEAD
      origin/main` is the `main` commit the branch last took: it moves exactly when a `main` commit is merged into the
      branch, never on the branch's own commits, so every foreign commit taken in (fast-decode, GPU skip-ahead work)
      regenerates and licenses the population before the next grading, as D-SLM8013 intends. The branch lands on
      `main` as a second squash (M2, §9) before E5's tag. Commissioning, by the debunker: must-rejects (1) and (3)
      above run unchanged with `origin/main` as the line; **must-reject (5)**: after M1, a commit on `main` not from
      this line, touching a file the references depend on, merged into the post-M1 branch with the population
      unregenerated: the loader refuses. **Must-accept (5)**: a commit of this plan's own on the post-M1 branch, with
      `main` unmoved, grades. The reference loader, the one path
      every cell that grades against a pinned reference takes, compares it with the reference file's anchor and
      refuses to grade on a mismatch. So no ctest cell grades around it, though ctest binaries run no git.
    - *Refusal cases.* A missing line ref, a shallow history or any `git` error records no merge-base, and the
      loader refuses. It never grades. *Rev 1.18.9 (§15.28):* the fixture (`write_merge_base.py`) checks
      `git rev-parse --is-shallow-repository` first. On `true` it records no merge-base, and refuses with a message
      that names the shallow history and the remedy (`git fetch --unshallow`). Until rev 1.18.9 it refused only on a
      `git` error or malformed output, so it graded in shallow clones. A cloud run that grades references unshallows
      its clone first, as CI's full-history jobs already do.
    - *Where it can grade.* A CI job that runs the reference cells checks out with `fetch-depth: 0` and fetches the
      line. At `5d5ca58`, `tests.yml:1106-1108` and `:1481-1483` already use full history; other jobs use the
      default shallow checkout. Reference cells run only in named full-history jobs. In any run expected to
      grade references, a missing ref, shallow history, failed fixture or loader refusal is a **test failure**,
      never a skip; E2, E3 and E4 each record a graded-reference count above zero.
    - *What it sees.* It sees **ancestry through `<line>`**: a build that merged newer paged-KV code on that ref,
      which is how E5 takes 1.13.0. It does not see code taken by cherry-pick, squash, hand copy, or a merge
      through another ref. R2's trigger covers those cases as a discipline, whatever the mechanism.
    - *Why not content.* A content key was weighed and declined. No path set both covers what the references
      depend on and excludes this plan's own edits: `src/sslm_abi.cpp` holds paged KV's prefill, adopt and
      `begin_from` beside this plan's two verbs, and `src/kv_pages.*` gains this plan's code (§3).
    - *The stake, reasoned and not executed.* A stale reference should give a false red rather than a false green.
      The prefill twin, built by the new code, is itself checked against the pinned streams, so new numerics turn
      the twin's check red. A false green would need a restore wrong in exactly the way that reproduces the old
      code's tokens.
    - *Commissioning*, constructions by the debunker:
      - must-reject (1): a commit C whose parent is the anchor, `HEAD` merging C, and `<line>` set to C. The
        merge-base is C, so the loader refuses;
      - must-reject (2): `<line>` naming an absent ref. The loader refuses;
      - must-reject (3): configure at the anchor, merge C into `HEAD` with `<line>` at C, rebuild without
        reconfiguring, then run ctest. The fixture recomputes C and the loader refuses the old anchor;
      - must-reject (4): make a descendant tag T from the anchor with only an unrelated-file change; switch
        `<line>` to T and merge T into `HEAD`. The loader refuses the old anchor even though no paged-KV rule
        changed. Regenerate and license the references at T, then require the same cells to grade with a
        positive count;
      - must-accept: `HEAD` at the anchor plus this plan's own commits, with the real line. The loader grades.
- **The prefill twin** is built in the same process by the unchanged prefill path; it grades restore, and is itself
  checked against the pinned streams.
- **Page counts** use PKV §8's fill probe (resolves one page), reused unchanged. *Rev 1.5:* it resolves only counts of
  2 or more free pages (`tests/paged-kv/pkv_common.h:585-602`, `:651-688`). *Rev 1.6, the reading rule
  (D-SLM7987; rev 1.5 applied the floor to three cells):* every page-count reading in §7 states its form, and each
  form has a floor (S29):
  - **Two-sided** (`ProbeExactlyFree`), `k ≥ 2`, on a state the cell can rebuild. Engine cells: 1.1, 1.2, 2.1, 2.6,
    3.1, 5.1, 5.2, 5.10, 5.11, 7.4 (root and nested twin), and their GPU twins 1.8, 2.7, 3.4 ("nothing drawn",
    *rev 1.7*, T-3027 N6), 5.3 and 5.10/5.11's GPU halves, in the GPU form F0 records.
  - **One-state** (`ProbeExactlyFreeOneState`), `k ≥ 3`, on a live state it cannot rebuild: 1.4's exact end
    reading, 1.7's before-destroy reading, 2.10's after-teardown reading, and 5.5's before and after readings.
  - **Admission lower bound**: creates totalling `F_min ≥ 2` are all admitted and then released. This is for a
    one-sided claim ("at most the budget's pages"): the per-tick readings of 1.4, 1.9, 8.10 and 8.12.
  - **0 or 1 free is never read.** Exactly 0 is built on a fresh pool of known page count `N` by admitting creates
    totalling `N`, and is confirmed by a 2-page create refused `SSLM_KV_POOL_EXHAUSTED` (5.6, 5.7). Exactly
    `ceil(L/B) ≥ 2` free is read two-sided (7.4).

  Each cell fixes a shape or sizes its pool so the reading meets its form's floor, and the shape is written in the
  cell. *Rev 1.8 (T-3026 N-4, T-3027 N9, T-3033 F21):* a reading of a pool's **initial** free count
  requires a fixture with at least one block of `cap_pages ≥ 3` pages. The engine fixtures meet this;
  each plugin fixture that takes such a reading states its geometry and this floor (8.10 uses `B = 64`,
  `cap = 256`, `cap_pages = 4`; *rev 1.13*: rev 1.12 said `B = 16`, `cap = 64`). This is a fixture requirement, not a property of every plugin pool.
  A reading below the floor fails
  `ProbeSplit`'s check, which is a failure for the instrument's reason, never
  a pass. All engine geometries here have `cap_pages ≥ 3`, so `ProbeSplit`'s `cap_pages = 2` limit (S29) does not
  arise. Plugin readings on a live pool (1.4, 1.9, 5.5, 8.10, 8.12) are taken with the worker lanes parked at their
  dequeue and **no job executing on any lane** (an idle lane counts as parked; the lane-park seam below).
  1.7's and 2.10's are
  teardown readings, taken after every lane has stopped, when no lane is left to compete. So the probe's own
  creates never compete with a cache
  build or restore for pages. They could otherwise refuse one `SSLM_KV_POOL_EXHAUSTED` and send it down the
  evict-and-retry path the reading grades (T-3024 N5).
- **Byte addressing** uses PKV's page-index peek seam with the §3.1 formula written in the test.
- **Timing harness** (E4, F3, U4): one harness timing prefill-build vs restore per prefix, idle checks as PKV §8's
  harness. The 1,000× bar in 10.3 is a decision-bearing number, so its instrument is **commissioned before its
  verdict counts** by a seat independent of the builder: a must-reject construction (a restore mutant that runs
  prefill internally, which the bar must refuse) and a must-accept (the built restore). Readings before that are
  recorded, not acted on.
- **`check_version_header.py`** is registered as an instrument with its must-reject (7.7); its constructions are
  the debunker's (11.3).
- **GPU speed bar** (10.4): the same harness and the same commissioning as 10.3, with a must-reject (a restore
  mutant that prefills internally) and a must-accept, on each GPU.
- **Rev 1.3: the two-name artifact** (2.13) is PC-12's registered must-reject, built by a fixture writer from the
  strike's names. The strike's own probe (FNV arithmetic and first-match resolution) reproduces the collision
  without a build. **The certified-artifact audit** (E0) is a script that lists every schema name in each certified
  artifact and in the plugin's example assets and computes `Fnv1a64`; it fails on a duplicate or a 0. Its
  must-rejects are the two-name artifact and the zero-hash artifact using D-SLM8002's printable name. Both
  audit branches are **COMMISSIONED** (T-3030, D-SLM8001); 2.13's C++ hash-substitution seam pins the parser's
  separate zero branch and does not stand in for the Python audit. The audit's
  must-accept is the writer's renamed, non-colliding pair, whose names and distinct non-zero hashes are
  computed independently with the engine's `Fnv1a64` algorithm (`schema_masks.h:110-120`), not selected by
  the audit's own reading. *Executed in E0 (T-3030, D-SLM8001):* the test author built the audit at T-3022;
  the debunker independently authored its constructions and commissioned `T-3022-PC12-schema-hash-audit` in
  `Claude/Tooling/instrument-registry.json` at 2026-10-01T20:46:24Z. Its parsed, non-crash must-reject named
  both the collision and zero-hash findings. The commissioned audit reran over all eight certified artifacts
  and example assets; all passed, with no duplicate or zero name hash. T-3030 §6 records the instrument hash
  and byte-identical rerun. These eight readings stand; the earlier quarantined readings were not adopted as
  independent verdicts. **D-SLM8017 closes T-3046 W-1:** after the writer's independent commission, the audit
  was re-commissioned at 2026-10-01T21:48:44Z; both registrations verified COMMISSIONED and its post-writer
  eight-artifact batch was byte-identical to the earlier batch. E1 consumes the later receipt and batch as its
  audit-order evidence. The scratch registry at `D:/_scratch/T-3022/` decides nothing. Before pruning
  `D:/SuperSLM/.worktrees/T-3022`, move the audit script and reject wrapper to their merged, durable test/tool
  locations, re-register both commands from those paths in the project registry, and retain the T-3030 receipt
  and readings as provenance. Verify the new registration before its next decision-bearing run.
- **The independent `SCM1` writer** (rev 1.6, D-SLM7985; *rev 1.7*, D-SLM7990, which supersedes D-SLM7985's seat
  and step and keeps its design) is that fixture writer. It is a test-only python module under the test tree,
  authored by **the debunker inside E0**. The conductor's roster ran the
  debunker on Fable (D-SLM7990). The debunker is neither the test author, who built the audit (T-3022), nor the
  builder, who builds `_serialize_scm1`'s refusal and the parser's (E2). So the two-name artifact it emits is an
  independent construction for both instruments. Rev 1.6 named the test author at E1, which is the audit's own
  builder and one step after the audit's verdict is due (T-3026 S-1, T-3027 F11). It
  packs design §13.2's `SCM1` format with `struct` (24-byte header, `schema_count × 56`-byte descriptors, name
  blob, four per-schema blocks, as `t2132_build_g5_fixture.py:109-169` documents it) and never imports
  `_serialize_scm1`. It builds the two-name artifact and the zero-hash artifact using D-SLM8002's name (2.13),
  and nothing a shipped tool uses.
  *Commissioning:* must-accept, on two non-colliding lists (the renamed pair, and 6.7's three-schema list), its
  bytes equal `_serialize_scm1`'s byte for byte, which shows it writes the real format.
  *Executed in E0 (T-3043, D-SLM8010):* the test author, independent of the writer's debunker author,
  constructed its must-reject and invocation. Its variant swaps the equal-length names' blob order while
  leaving descriptors and compiled blocks in input order; byte equality with `_serialize_scm1` refuses the
  structurally valid variant. It ran cleanly and exited 1 for byte inequality; a missing-checkpoint control
  exited 0, so a broken construction cannot count as refusal. `T-3030-SCM1-writer` was registered and
  COMMISSIONED at 2026-10-01T20:58:11Z, after the audit's 20:46 commissioning. This differs from rev 1.8's
  prescribed writer-before-audit order (T-3042 M7). It does not make that order retroactively true: the audit's
  independent commissioning and eight readings stand under T-3030, and the writer's independent challenge now
  meets the intended second proof before E1 consumes either. The writer hash is pinned in T-3043 and the registry.
  Before pruning `D:/SuperSLM/.worktrees/T-3043`, move the writer, accept comparator and test-author reject
  construction to their merged, durable test locations; re-register both commands from there and verify before
  the next decision-bearing run. The two-name artifact it emits is the 2.13 parser must-reject (E2) and the
  audit's must-reject (E0). For the collision pair, compare the artifact with
  `_serialize_scm1`'s renamed-pair output at every offset outside the name blob and descriptors' name fields:
  all other bytes are equal. The audit's registered must-reject wrapper exits non-zero only when **both**
  artifacts' runs exit non-zero and each prints parsed `status: FAIL` with a duplicate or zero finding.
  A crash, parse error, `ERROR` or absent finding makes the wrapper exit zero, so commissioning rejects
  the audit rather than treating a broken construction as a successful refusal (T-3030 §4).
  This keeps both
  "producible by the real data path" (standards §5.4): only the names differ from what the production serializer
  would write if it did not refuse.
- **Seams and instruments new in rev 1.1**, each test-only and each with its must-reject:
  - `sslm_test_only_prefix_prefill_call_count`, a counter in `sslm_prefix_prefill` (10.1, 10.5, 7.13). *Rev 1.2:*
    not in `PrefillWholeTokens`, which sequence prefill shares (`sslm_abi.cpp:2712`), so a correct build whose
    sequences prefill prompts would read non-zero. Must-reject: a process that runs one prefix prefill reads 1.
  - *(Rev 1.2; rev 1.6 adds two positions and a second record, D-SLM7984.)* The **restore-site fault slot**, a
    thread-local single-shot flag shaped like `g_inject_tiled_gemm_alloc_failure`
    (`tests/support/bad_alloc_injection.h:159`), compiled only into the test-injection library. It is armed with
    one position:
    - **share**, with a page status (`kInvalidArgument` or `kRefcountUnderflow`): §3.4 step 2 returns that status
      in place of calling `HolderShareLeading`, before any count changes, as the real check does (S26);
    - **map**: the holder's reserve is returned to the pool (`HolderReturnReserve`, as
      `sslm_pkv_test_only_drain_reserve` does) just before step 3, so the real `MapPagesFor` refuses;
    - **allocation**: the handle and residual allocation of step 4 throws `std::bad_alloc`.

    When it fires it records the pool's free-page count and the holder's shared-entry count, and (*rev 1.7*, T-3027
    F13) the holder's mapped-entry count (`PageHolder::mapped`) and reserve size (`reserve.size()`; `kv_pages.h:93-103`).
    It also reports whether it fired. A thread-local single shot left armed by a case that never reached it would
    fire in the next case. So every case that should reach the slot asserts it fired, every case that should not
    asserts it did not (5.6), and the test disarms it before the next case. The general slot is
    not used for 5.2, 5.6 or 5.11, because `Sha256` consults it (`src/sha256.cpp:314`, `:365`, `:372`) at step 3,
    before any draw. Must-rejects, authored by a seat independent of the builder:
    - (1) armed at allocation, with a valid root blob and enough free pages, the restore returns
      `SSLM_ALLOCATION_FAILED` and the recorded free count is below the initial count. A slot placed before the
      draw fails this.
    - (2) *(rev 1.6, T-3024 F5)* armed at allocation, on a nested restore with a matching hint at `s ≥ 1`, the
      recorded shared-entry count equals `s`. A slot placed between the draw and the share records 0 and fails
      this, though it passes (1).
    - (3) armed at share, on a nested blob with a matching hint at `s ≥ 1` (the cohort shape; *rev 1.7*, T-3026
      N-3, T-3027 F13: on a root blob the share position is never consulted, and a correct slot would fail), the
      status returned is the armed one's mapping (`SSLM_INVALID_ARGUMENT`,
      `SSLM_ARTIFACT_REJECTED`). Armed at map, on a root blob, it is `SSLM_KV_POOL_EXHAUSTED`. A slot whose position is never
      consulted returns `SSLM_OK` and fails this.
    - (4) *(rev 1.7, T-3027 F13)* armed at allocation on a root blob at `L = 4B + 1`, the recorded mapped-entry
      count is 5 and the reserve 0. A slot that fires before the map records 0 mapped and a reserve of 5, and fails
      this though it passes (1) and (2). On such a slot a failure path that returns only the reserve passes 5.2,
      while the shipped code, which allocates after the map, would leak the `ceil(L/B) − s` mapped pages on every
      `bad_alloc`.
  - The off-game-thread assert (7.8). Must-reject: an instrumented call made on the game thread trips it.
  - *(Rev 1.2.)* The on-game-thread assert in the cache's single state accessor (3.9). Must-reject: a test call
    through the accessor from a pool thread trips it.
  - Plugin counting seams on `sslm_prefix_release` (per handle, 2.10) and on cache restores (7.13), and the
    spec-mismatch counter (2.12). Must-reject for each: one extra call is counted.
  - A test seam that skips the producer check, for 4.6(c) only. Must-reject: with it off, 4.6(c)'s blob is refused.
  - Fault seams between the disk write and the rename (5.8), and a lane delay (5.9). *Must-rejects (rev 1.2):* with
    the fault armed, no file exists under the key name and the temporary is present or removed as §5.7 specifies;
    with the delay armed and a zero slice, the hitch counter rises.
  - The no-page-to-spare pool of 5.6 is built with PKV's `AdmitPages` (the fill probe's own create split), not read
    by the probe (*rev 1.5*). *Rev 1.6:* it holds exactly 0 free, by admitting a fresh pool's whole page count
    (S29). 5.6's armed half uses the fault slot above at its allocation position only (*rev 1.7*, D-SLM7993).
  - *(Rev 1.6, T-3024 N5; rev 1.7 rewrites it, D-SLM7996, superseding D-SLM7987's "between Layer-1 calls";
    T-3026 W-6, T-3027 F14.)* The plugin's **lane-park seam**. Each lane is one persistent worker thread draining a
    FIFO of opaque `Execute` jobs. The game thread enqueues at any time, and the worker's only point outside a job
    is its dequeue (plugin `SuperSLMSubsystem.cpp:913-951`, `Run()`; `:965-973`, `Enqueue`; this worktree's tree).
    One job can make several Layer-1 calls, so the seam parks **between jobs**, never between two calls of one job.
    - *Park.* A test-only switch the worker checks at its dequeue, before taking the next job. While it is set the
      worker takes no job, and whatever its finished and unfinished cache work has drawn stays drawn.
    - *Barrier.* The seam exposes **no job executing on any lane**. A lane blocked in `WakeEvent->Wait()` with
      an empty queue counts as parked; a lane leaving the queue changes its executing state under `QueueCS`
      before `Execute`. An idle lane never has to wake just to report. The plugin-side fill-probe wrapper is
      the only entry to the probe in plugin cells; it refuses to read unless **the switch is set and the
      barrier holds**, or every lane has stopped. Every plugin reading of §8's rule goes through that wrapper.
    - *Teardown.* The park yields to the stop request. `Run()` drains its queue before honouring a stop, and
      teardown waits on it (`Kill(true)`, `:1624-1633`), so a park that held through a stop would hang
      `Configure()` and `Deinitialize()`. 1.7's and 2.10's readings are teardown readings, taken after the lanes
      stop and never under the park.
    - *Timing cells.* Cells that read the hitch counter (5.9, U4's timings) never run with the seam set, because a
      parked tick makes every due delivery late.
    - *Must-reject*, authored by a seat independent of the builder: lanes with work queued, one lane held inside a
      job by 5.9's delay seam when the switch is set. The observable is jobs **started by a worker**, a test-only
      counter at `Execute` entry, not jobs posted, which `Enqueue` keeps adding while lanes park. Beside it, the
      barrier's release time is read against the held job's finish. A seam that starts a queued job, or that
      releases the barrier before the held job returns, fails.
    - *Must-accept:* set the switch with all lanes idle: the barrier opens; clear it: queued cheap jobs run.
      *Stop-yield construction:* set the switch with a cheap job queued on every lane, then call
      `Deinitialize()`. It returns within a chosen 10-second watchdog bound and jobs started equal jobs
      posted. The watchdog expires red, never a hanging or skipped cell. *Wrapper must-reject:* attempt a
      probe reading with the switch clear and a lane still running; it refuses before any probe create.
      The test author authors these constructions independently of the seam builder.
    - Feasibility, read for rev 1.7: a gate at the dequeue fits the loop as written.
    - *Build hold* (*rev 1.18*, PC-15 Q7). A CPU cache build is one job (§5.3 item 4), so 8.10 (c)'s drawn-and-unfrozen
      reading needs a stop inside it. A test-only hold the build job checks once, after `sslm_prefix_begin_budgeted`
      returns and before its first prefill. While armed, the job blocks there, having drawn its reserve and making
      no Layer-1 call. The barrier counts a lane blocked at the hold as parked, exactly as an idle lane; it exposes
      the number of lanes at the hold. The hold yields to the stop request as the park does. Must-reject: a lane at
      the hold that makes a Layer-1 call while held (a test-side call counter on the lane) fails; a barrier that
      opens with a build job past the hold but not frozen fails. Must-accept: K builds at the hold with K lanes,
      the barrier open, the ledger's in-flight count K, zero frozen. 7.21 uses the same hold to keep a lease not
      Ready.
  - *(Rev 1.7, D-SLM7991, superseding D-SLM7989.)* The **`docs/api.md` ordinal check** (7.17) is registered as an
    instrument, built by the test author, with constructions authored by the debunker as 11.3's are. Must-rejects:
    `SSLM_KV_BUDGET_EXCEEDED` stated at 30; `SSLM_RESTORE_PRODUCER_MISMATCH` stated at 31; one
    `SSLM_KV_BUDGET_EXCEEDED` statement reworded past the locator, which drops below the per-name floor.
    Must-accept: the constructed anchor doc plus the correct status-30 statement in 7.17.
  - *(Rev 1.8, D-SLM7995; T-3033 F19.)* The plugin's **1.0.0-prefix counter** for 8.10's `P₁₀(t)` holds
    only the handles returned by successful `CreatePrefix` begin jobs. A release removes a handle only if it
    belongs to that set, and the set is read behind the lane-park barrier. The debunker authors its
    constructions independently of the counter builder. Must-accept: one live `CreatePrefix` handle is
    counted, a cache eviction leaves that count unchanged, and releasing the counted handle removes it.
    Must-rejects: an extra admitted `CreatePrefix` changes the count by one; a cache eviction that changes
    the count fails; a counted handle released without decrement fails.
  - *(Rev 1.18.)* **Plugin seams for the rev 1.18 cells**, each test-only and compiled out of shipping builds, each
    registered with a must-accept (the seam reads the unmutated path's expected count) and a must-reject (the
    counted event injected once by the test reads one more):
    - a **tokenize-call counter** over every plugin `sslm_tokenize` call, both sites (7.20; *rev 1.18.1*, audit G6:
      rev 1.18 named one site), read at runtime (must read 0) and at the editor build (must read > 0); its
      must-reject makes one call through each site;
    - a **game-thread prefill counter** in the editor build path (3.11), counting prefill calls made on the game
      thread;
    - a **pool-thread hold** on a key's write-back and on the trim, and a **file-delete refusal** seam (3.10, 5.12),
      with an exclusion counter over delete and trim;
    - a **lock-holder bound override** and per-job duration record on the GPU submission thread (5.13), and a
      **release-issue counter** beside the existing device-loss seam (5.14);
    - the drop diagnostics themselves (drops, deferred, marks applied, marks cleared) are cache diagnostics, read
      as bookkeeping; no verdict rests on them alone except in 7.18's marks-applied count, whose subject they are
      (*rev 1.18.2:* 3.13 reads them beside its sources and prefill count, which decide it).
    - *(Rev 1.18.1, audit N4.)* Two seams take their own must-rejects, not the generic one: the **exclusion counter**
      over delete and trim, whose must-reject is a forced overlap that must read 2; and the **lock-holder bound
      override**, whose must-reject is a job held past the lowered bound, which must mark the thread unresponsive.
    - *(Rev 1.18.1.)* A **hash-completion hold** per key (3.12), letting a test choose the order in which two ops'
      key hashes complete; must-reject: with the hold set, an op whose hash is held is never applied.
    - *(Rev 1.18.1, plan review F1.)* A **release-status substitution** seam (5.15) that runs the real
      `sslm_prefix_release` and then returns a chosen non-`OK` status, so the handle is consumed as the engine
      consumes it; must-reject: a seam that skips the real release (the rev 1.18 seam) is refused by the
      release-counting seam reading 0 releases.
    - *(Rev 1.18.2; the strike on rev 1.18.1, lesser planes 2 and 3.)* Three more, each with the generic
      must-accept and must-reject unless stated:
      - the pool-thread hold also holds a key's **disk read** (1.7, 3.10 (vii));
      - a **per-key overlap counter** over one key's disk read, write-back file write, delete and (*rev 1.18.5*) touch (3.10 (vii)); its
        must-reject is a forced overlap of a read and a delete of one key, which must read 2;
      - a GPU **busy-release substitution** seam (5.15 [U6]) that returns `SSLM_BUSY` for a chosen number of issues
        of `sslm_gpu_prefix_release` **without calling it**, then lets the real release run. Its must-reject is a
        seam that calls the real release before returning `SSLM_BUSY`: the release-counting seam then reads a
        release at the first issue and a second at the re-issue, and must be refused.
  - *(Rev 1.18.1, audit G3.)* The **GPU label check** (7.22) is registered as an instrument, built by the test author,
    its constructions authored by the debunker as 7.17's are.
- **SHA-256 on both dispatch arms** is already pinned to the FIPS vectors (S24). So a blob cooked on a SHA-NI machine
  and verified without SHA-NI is covered at engine level. *Rev 1.2:* the plugin's own build of the portable arm is
  **not covered**. The box's 3950X has SHA extensions and the plugin defines no forcing macro, so 10.5 runs the
  SHA-NI arm (rev 1.1 said wrongly that 10.5 covers it). A miscompile there refuses good blobs, which costs a
  recompute, not wrong output.
- **Executed figures in this plan** (§3.2 sizes, §3.5 cohort share, 10.3–10.4's speed arithmetic) came from a python
  computation in the planning session over the geometry constants cited; PKV §3.1 gives the same bytes per token.
  *Rev 1.6:* 5.1's and 7.4's shapes, the W-2 counterexample (origin 1,000, `L` 1,001: 1 free), S29's
  `R((r − 1)·B) = r` sweep and its `ProbeSplit` sweep (including the `cap_pages = 2` counterexample) came from a
  python computation in the fold session that transcribes `ReservationPages` and `ProbeSplit` from `5d5ca58`.
  Engine behaviour was read at source, not executed. *Rev 1.13:* §3.2's resident-page columns, §3.4's strike-cell
  figures at the rule B (3 vs 10; 5 vs 19), §3.5's rule-B cohort figures (7/3/896; 15/4/960), S30's rule-B values and
  8.10's `B = 64`, `cap = 256` fixture came from a python computation in the fold session that transcribes
  `PageRuleB` and `ReservationPagesIn` from `3b6631f`. They agree with the re-anchor record's measured and derived
  values (F2). The rule-B restore behaviour itself is the re-anchor's scratch probe (1 cell, 1,448 checks, 0
  failures), which 4.7, 8.13 and 9.8 turn into cells; no build or test ran in this fold.

---

## 9. Build steps

Roles: planner, plan reviewer, coverage auditor, adversary, test author, builder, code reviewer, debunker, and the
principal. In each build step the test author's red cells land first; the builder turns them green; the code
reviewer reviews. Nothing later gates anything earlier. Every commit carries `[skip ci]`; release tags are the
exception.

*Rev 1.18, where the steps stand* (git, 2026-10-20; `merge-path-2026-10-20.md`). E0 and E1 are done. E2's and E3's
code is built on `claude/project-thread-wt2dpx` (tip `83a82c2`), and their cloud legs ran, but **the 2026-10-17 box
queue never ran**, so none of E2's box receipts (1)–(6) and none of E3's box legs exist. E2's receipt (8) is widened
below to the four unreviewed changes. E4's cells still need the rule-B re-derivation (10.1; `e4_real.cpp` is
unchanged since 2026-10-02). E5's 7.17, 9.9, 9.10 and the 9.5 pins are not written. F0–F2 ran in the cloud on a
software GPU only; F2's box run on both certified GPUs has not run. U1–U3 and U6 are drafted on Wizard
`claude/project-thread-wt2dpx`, uncompiled and switched off (`SUPERSLM_WITH_PREFIX_CACHE`). **Long legs** (the 4.1
grid, ~7.5 h; 4.2/E3 on MSVC, over 90 min per 1.5B origin; the unpinned synthetic 1.5B legs of 4.7 and 9.8, ~5 h,
not checkpointed) run on the principal's **work machine**, never the home box (§1); checkpointed legs on the home
box only in resumed runs of at most 15 minutes.

| Step | Owner | Where | Depends on | Work | Cells green | Gate | Commit message |
|---|---|---|---|---|---|---|---|
| P0 | tracker | cloud | — | File this plan; mint PC-1..PC-12 in the decision log | — | records merged | `[skip ci] Precached prefixes: plan of record filed` |
| P1 | plan reviewer, coverage auditor, adversary | cloud | P0 | Temper, coverage audit, strike (§13) | — | all clear; folds re-gated | per seat |
| E0 | builder (reads), test author, debunker | cloud + box | P1; **PKV's built CPU slice** (`5d5ca58`) | The cloud re-anchor ran 2026-10-01 (`e0-reanchor-pkv-5d5ca58.md`). T-3022's reference capture is reproducibility evidence only (D-SLM7997). **Done:** the debunker built the independent `SCM1` writer; the audit was commissioned at 20:46 UTC (T-3030, D-SLM8001), and the test author's name-swap must-reject commissioned the writer at 20:58 UTC (T-3043, D-SLM8010). D-SLM8017 then re-commissioned the audit at 21:48 UTC after the writer, verified both registrations, and repeated the eight-artifact batch byte-identically; T-3046 W-1 is closed (§8). The audit's zero-hash branch is commissioned (D-SLM8002). Preserve their receipts and re-register the audit from a merged location before pruning `D:/SuperSLM/.worktrees/T-3022`, and the writer before pruning `D:/SuperSLM/.worktrees/T-3043` (§8); never promote the scratch registry. The test author regenerates the §8 reference population (T-3035); its discrimination instrument is independently commissioned and the coverage auditor re-audits it before E2's first grading. Re-diff when newer paged-KV code enters this branch; regenerate references whenever the recorded merge-base changes | — | **E1:** both commissioning receipts, the post-writer audit receipt and eight standing verdicts. **E2 grading:** regenerated references reproduced twice, discrimination commissioned and re-audited | `[skip ci] Precached prefixes: re-anchor on paged KV; pinned reference streams` |
| E1 | test author, coverage auditor (design recheck) | cloud | E0 | CPU red suite: the test author authors the complete fixtures and deciding assertions for **every [E2]–[E4] cell** against stubbed verbs (§7), including the T-3060 F1–F15 clauses the stubs prevent from executing. The coverage auditor rechecks those authored assertions against T-3060's full catalog before E1 closes; an ID registration or stub-red count alone does not clear a partial cell. Consume E0's commissioned writer and its two-name and zero-hash artifacts; no writer construction remains in E1. Add the `PC_RESTORE_PRODUCER_MISMATCH` test-side constant and `static_assert`. Keep the runtime red suite buildable; the public macro-vitality compile proof belongs to E2 after the builder defines it. Author 6.4's isolated nonnegative `current_token` mutant against `SPX1` blob equality; E2 executes it only after the unmutated pass. Do not use an adopter's `SSB6` sentinel as the `SPX1` token oracle. Include T-3024 F6 (2.3b's in-domain `m` and `e` equality, per S20) and F7 (7.12's blob equality across legacy and budgeted builds). The independently known bound, nonempty 7.12 fixture, five independent count operands, B=4/B=16 comparison outside identity/digest fields and symmetric writer/reader field-swap construction are authored here; their built-verb proof is E2. Author 5.7's constructed GPU-shaped header from a known-good CPU save, with independent field and digest checks and backend/producer pair assertions; a real GPU capture is an F2 obligation | — | **Before E2 product implementation starts:** every [E2]–[E4] deciding assertion authored and reviewable; coverage auditor clears T-3060 F1–F15's red design and the full §7 catalog; suite buildable and red for stated reasons. Executable reach and mutants remain green-step obligations | `[skip ci] Precached prefixes: CPU red suite` |
| E2 | builder, test author (execution), coverage auditor (discrimination re-audit) | cloud + box | E1's complete red-design clearance; T-3035 population re-audited before first grading | `sslm_version.h` and its check; `SPX1` writer/reader; `sslm_prefix_save`; `sslm_prefix_restore` private path; `SSLM_RESTORE_PRODUCER_MISMATCH` in every mirror; define `SSLM_HAS_PREFIX_SAVE_ABI` in `sslm_abi.h` and add it to `tools/ci/abi_header_inventory_allowlist.txt`; PC-12 parser and converter refusals with no opt-out; api-slot census; header text; §8 engine seams and releases on refusal. Run the E1-authored [E2] assertions against built verbs and record which assertions they reach. Prove 7.12's independent parser accepts the bound nonempty built blob, checks all five independent counts, rejects an isolated token edit and a symmetric writer/reader field swap, and compares B=4/B=16 canonical bytes outside independently checked identity/digest fields, with separate receipts from its standalone controls. Execute 5.7's validated constructed GPU-shaped header and its backend/producer pair on built CPU restore. After defining the macro, compile its vitality cell green and prove removal red. Build the ctest merge-base fixture and loader refusal; the debunker's constructions commission them before reference grading. Run 2.3b's edited-scale token equality. Record an unmutated 6.4 pass, then run its isolated `current_token` mutant and require only that mutation to turn 6.4 red; record both outcomes separately. Run the T-3060 F2–F15 applicable E2 mutants and have the coverage auditor re-audit discrimination against built verbs before E2 green; F1's parser proof is part of the same gate. E3/E4 portions retain their E1-authored assertions and execute at those gates. *Rev 1.13:* the build is merged onto `v1.13.0` (`6426a72`, then `3b6631f`; `build/e2-reanchor-v1.13.0.md`). Its pinned cells stay at B = 16 and the B = 4 view (§7 preamble). The test author writes rev 1.13's unpinned E2 cells against the built verbs: 4.7 (its E2 legs), 8.13, 9.8 (its E2 legs) and 7.12's pool-kind leg. *Rev 1.14:* each maps its own model instance, feeds the test-written rule B to every test-side helper, and asserts the engine's B equals it (§7 preamble); 9.8's contiguous pools on the synthetic 1.5B shape need about 1.3 GiB each, budgeted in E2's box time. Written after the build, they are not red-first; each counts only once its named mutant has turned it red, with a receipt. Switch `<line>` to `v1.13.0` before the first reference grading (§8) | every [E2] cell | cloud toolchains, TSan, ASan+UBSan and MSVC suites green; Gate C entries green; macro vitality green and removal red; regenerated references graded with count above zero on the pinned base; 7.12 parser and symmetric-swap receipts; 2.3b equality; 6.4 baseline and isolated-token-mutant receipts; T-3060 F2–F15 E2 assertions reached, applicable mutants killed, and coverage auditor's discrimination re-audit clear. *Rev 1.13, from the re-anchor record's §5, each its own receipt:* **(1)** the MSVC ASan container-overflow report in `7.12/E2:independent-parser` at `tests/precache/e2_format.cpp:160` (`swap_command.replace`; it was `:158`) triaged at the box by the test author: rebuild with `/MD`, or define `_DISABLE_STRING_ANNOTATION` for the whole link, or build `e2_format.cpp` alone without the annotation, and show the report gone with the cell still green. GNU ASan cannot see this class, and the leading hypothesis (annotated and unannotated `std::string` mixed under `/MT`) is unverified; a report that survives every remedy is a finding, not a pass. **(2)** 6.4's isolated `current_token` mutant run on MSVC at the box, recorded as the plan's receipt and audited (the cloud run: 110 checks unmutated; mutated, exactly 10 failures, `e2_root.cpp:109` and `:111` in each of the five non-empty shapes). **(3)** the T-3060 F2–F15 applicable E2 mutants executed by the test author, then the coverage auditor's discrimination re-audit against built verbs. **(4)** the coverage auditor's re-audit of the re-anchor's test changes: the F2 pins (+20 checks across the 44 cells) and the F3 `PageRunOffset` oracles keep each cell's discrimination, and 4.7, 8.13, 9.8 and 7.12's pool-kind leg close the rule-B, partial-page and contiguous-pool gap with their mutant receipts; *rev 1.14:* including the receipt that a 9.8 written through `GetFixture("pkv_def")` goes red on its B assertion, and 8.13's A22 receipt on a Release build. **(5)** references: `<line>` at `v1.13.0`, merge-base `a5f3ad1` recorded, the population regenerated and independently licensed there, and a graded count above zero. **(6)** the box legs: MSVC Release, clang-cl and MSVC ASan; the full `4.1/E2:synthetic-shapes` grid on the real pinned fixtures, **1.5B included** (the cloud ran only the 0.5B half and had no 1.5B stand-in; about 7.5 h of box time); cross-compiler `9.1/E2` with MSVC in the triple; the E2 real-artifact leg that needs box inputs (`e2_reference`, cell 6.2/E2); licensed `6.2/E2:reference` grading. *Rev 1.14 (re-gate review B1):* rev 1.13 also listed `real_two_process` and `e3_reference` here, copied from the re-anchor record's list of ctest failures, which did not filter by step. `real_two_process` runs 10.1/E4 and 10.2/E4 (`precache.cmake:59-61`, `run_real_two_process.py:34`, `:53`), which stay red until E4's re-derivation (10.1); it is an E4 receipt. `e3_reference` is 6.3/E3 (`precache.cmake:67`), which grades the hint that E3 builds (at `3b6631f` the restore ignores it, `sslm_abi.cpp:4108`); it is an E3 receipt. Their reds at E2 are expected and are not E2 findings. TSan runs on Linux in the cloud (not run at the re-anchor; the box's WSL has none). **(7)** `tests/t2807-api-slot/c_abi_verbs.txt` regenerated on Windows with `gen_consumed_symbols.ps1`, so its "read from" line names a real tree revision. **(8)** code review of `6426a72` and `3b6631f`, the merge resolution included; *rev 1.18:* the same independent round also reviews the shared `SPX1` codec `3a00abb` (it changes the CPU restore) and the GPU snapshot fixes `fdbb97b` and `83a82c2`. **(9)** rev 1.13 re-gated (§13 item 13), and rev 1.14's fold of that re-gate re-gated (§13 item 14), before E3 builds on it. *Rev 1.18:* receipt (6)'s 4.1 grid with 1.5B, and the unpinned synthetic 1.5B legs, run on the work machine (§1); receipt (5)'s reference line is `v1.13.0` until M1 and then `origin/main` on the post-M1 branch (§8; *rev 1.18.1*) | `[skip ci] ABI: save and restore frozen prefixes` |
| M1 | builder, code reviewer, test author (references), debunker | cloud + box (*rev 1.18.1*, plan review F12: `c_abi_verbs.txt` regenerates on Windows only, and the positive graded count needs box inputs) | *(Rev 1.18, PC-14; rev 1.18.1: the conductor's ruling on PC-14 (a).)* E2's receipt (8) (the code review, covering the four unreviewed changes) and E2's MSVC builds (receipt (6)'s MSVC Release, clang-cl and MSVC ASan builds and their bounded suites) green; before E2's reference grading (receipt (5)) | **The squash to `main`.** Merge `main` (1.13.1; `2e12a6e` at 2026-10-20) into the line, resolving the build lists (the three PowerShell list conflicts the merge path records); set `sslm_version.h` and CMake both to **1.13.1** (§3.8; fixed by approved PC-7; E5 moves both to 1.14.0); **relabel this plan's GPU code in every file** §4's preamble lists (and any the grep finds), to 1.16.0 and, in the public header and `docs/api.md`, "unreleased until 1.16.0" (plan review F5, audit G3), and run 7.22; regenerate `c_abi_verbs.txt` on Windows; land the result on `main` as **one squash commit** (no line history: the squash has one parent, and the line's tip is not an ancestor of `main`; *rev 1.18.9, §15.28:* rev 1.18 said "`main`'s 2026-10-12 parentless roots stay its only roots", a shallow clone's reading), recording the line's tip SHA in the commit message; pin the new `src/spx1_format.cpp` in `tools/ci/branch_coverage_floors.json` from the first CI run on that head (the merge path's step 4; plan review F12). Record the squash's diff against `main` file by file (§14). Then **cut the post-M1 branch** from `main` at the squash commit, set `<line>` to `origin/main` (§8; plan review F2, audit G1), and regenerate and license the reference population at that anchor before the next reference grading. E2's remaining box receipts, E3, E4 and E5 then run on the post-M1 branch, which lands by M2 | the cells already green on the line stay green on `main` (the bounded E2+E3 suites, cloud toolchains); 7.22 | code review of the squash's resolution and labels; *rev 1.18.1 (audit G2), receipts read as values:* `sslm_version.h` and CMake both read 1.13.1, `main`'s pre-squash CMake version; a blob saved on the M1 tree carries producer 1.13.1, read by 7.12's independent parser; *rev 1.18.9 (§15.28), taken in a full clone (`git rev-parse --is-shallow-repository` reads `false`, recorded):* the squash commit has exactly one parent, `main`'s pre-squash tip; `git rev-list <pre-squash tip>..<squash>` lists the squash alone; and `git merge-base --is-ancestor <line tip> <squash>` fails, so no line commit entered `main` (rev 1.18's "`git rev-list --max-parents=0 main` lists exactly the five 2026-10-12 roots" held only in a shallow clone) (mutant: the squash sets 1.14.0 → the value receipt red, while `check_version_header.py` stays green); 7.22 green; the debunker's must-reject (5) and must-accept (5) for the post-M1 line (§8); references regenerated at the new anchor with a positive graded count | `[skip ci] Precached prefixes: CPU save and restore (squash of claude/project-thread-wt2dpx at <tip>)` |
| E3 | builder, test author (execution), coverage auditor (discrimination re-audit) | cloud + work machine (*rev 1.18*: 4.2/E3's MSVC legs and 9.8/E3's 1.5B leg) | E2; E1-authored [E3] assertions | The hint: conditions (a)–(f), shared draw, scatter from `s·B`. *Rev 1.13:* builds on `v1.13.0`, after rev 1.13's re-gate (*rev 1.14:* and rev 1.14's, §13 item 14); the test author adds 4.7's nested leg at the rule B. *Rev 1.14 (re-gate audit B1):* and 9.8's nested leg in the contiguous pool, with its mutant (the hint's `s` or span from the model's rule B), added to E3's carried list beside T-3060's; the hint uses the restore pool's B in every operand (§3.5). Run the carried E3 portions of T-3060 F5–F9, F11 and F13 on built verbs, including F8's isolated hint conditions and physical page counts; execute their applicable mutants and have the coverage auditor re-audit discrimination | every [E3] cell | as E2; reference cells graded with count above zero, *rev 1.14:* including the box leg `e3_reference` (6.3/E3, moved here from E2's receipt (6)); 9.8's nested-leg mutant receipt; carried assertions reached, applicable mutants killed, and E3 discrimination re-audit clear before E3 green | `[skip ci] ABI: a restored child re-shares its live parent` |
| E4 | builder (runs), test author (execution), coverage auditor (discrimination re-audit), debunker (commissions) | box | E3; E1-authored [E4] assertions; *rev 1.18:* the test author's re-derivation of the E4 cells (10.1, 10.2), not yet done, and its coverage audit | Commission the timing instrument; run 4.3 (at the pin on pkv_def; *rev 1.18.4*: it has no real fixture, §7 *Geometry*) and 10.1–10.3 on real artifacts, unpinned at the rule B, with the `e4_real.cpp` geometry assertions re-derived to 128 (0.5B) and 64 (1.5B) per 10.1 (*rev 1.13*). *Rev 1.14:* the E4 cells take the rule B test-side (*rev 1.15:* each real fixture's `fx.pinned` set to the test-written rule B, never `PKV_FIXTURE_B_FROM_RULE` in `superslm_precache_e1`; 10.1) and assert `fx.B()` equal to the rule and the engine's B; pools are sized at the rule B (`:70-72`, `:272`, `:330-337`); 10.2's `:388` is re-derived to `s = 7` and its 3-page draw asserted by the fill probe (10.1, 10.2). Run 6.6's pinned leg as written and its rule-B leg on the real 0.5B artifact (6.6; rev 1.13 said all of 6.6 ran unpinned, which its pkv_def leg does not); record the per-MiB restore cost and worst-setting figure. Run the carried E4 portions of T-3060 F10–F13 on built verbs, including F11's separate-process and licensed-reference receipts and F12's resolving-power and real-workload receipts; execute applicable mutants or independent real-artifact controls and have the coverage auditor re-audit discrimination | 4.3, 6.6, 10.1–10.3 and every [E4] cell | regenerated real-artifact references graded with count above zero; *rev 1.14:* the box leg `real_two_process` (10.1/E4 and 10.2/E4, moved here from E2's receipt (6)) green after the re-derivation; 10.3's bar met or folded (§10 R1); carried assertions reached, applicable mutants or independent controls proved, and E4 discrimination re-audit clear before E4 green; ~45 min | `[skip ci] Precached prefixes: real-artifact oracle and timings` |
| E5 | builder, code reviewer, coverage auditor, adversary, test author, debunker | cloud + box (+ work machine: 9.10's 7900 XTX half) | E4; M1; *(rev 1.18.1, plan review F3: rev 1.17's "1.13.0 tagged" dependency and its `v1.13.0`/`a5f3ad1` checks are superseded and removed)* | *Rev 1.18.1, rewritten for the post-M1 line (plan review F3).* E5's work runs on the post-M1 branch, with `<line> = origin/main` (§8). Compare the recorded merge-base before grading: every change, including a `main` commit that added only unrelated files, regenerates and independently licenses references at the new anchor (§8, D-SLM8013). If a `main` commit taken in changes a §3 or §4 rule, re-diff, fold and re-gate that delta before the suite runs (§8, R2). Capture and pin 9.9's and 9.10's sequence saves on a `v1.13.1` build (`d697927`), then run 9.9 and 9.10 [box] on the release tree (PC-13). Before release edits, the test author writes 7.17 red and the debunker commissions it with the constructed must-accept in §8. Re-run the full CPU suite; release 1.14.0: `docs/api.md` states the new status at 30 and documents `SSLM_HAS_PREFIX_SAVE_ABI`; changelog, release note (also naming the macro, and naming the GPU twins as exported but unreleased until 1.16.0), version 1.14.0 in CMake and `sslm_version.h`; pin 9.5 blobs; final review, publish-gate audit (including 7.22's label check), strike on shipped contract sentences (§3.3–3.5). Then M2 lands the branch on `main`, and the release tree is `main` after M2 | 7.17, 9.9, 9.10, 7.22 | all clear; the release tree passes 7.17, 9.9, 9.10 and 7.22, and reference grading uses the population licensed at the current merge-base with a positive count; after M2, the principal tags `v1.14.0` on **M2's squash commit** (*rev 1.18.2*, the strike on rev 1.18.1, lesser plane 5: rev 1.18.1 said "from `main`", which does not name the commit), recorded in M2's receipt. *Rev 1.18.3 (the re-check's m2):* the tag step's receipt is `git rev-parse v1.14.0^{commit}` equal to that recorded SHA; it moved here from M2's exit, which runs before the tag exists (~20 min) | `[skip ci] Release 1.14.0: save and restore frozen prefixes` |
| M2 | builder, code reviewer, test author (references), debunker (*rev 1.18.2*, the strike on rev 1.18.1, lesser plane 5: rev 1.18.1 listed the first two only) | cloud + box (+ work machine: 9.10's 7900 XTX half) (*rev 1.18.2*: rev 1.18.1 said cloud only; the release tree's cells include 9.10 on both GPUs, and a regeneration needs box inputs, as M1's F12 note says) | *(Rev 1.18.1.)* E5's release work on the post-M1 branch, all clear | **The second squash to `main`**: merge `origin/main` into the post-M1 branch (any `main` commit taken here moves the merge-base: the test author regenerates the references and the debunker licenses them again before landing, §8), then land it on `main` as one squash commit recording the branch tip, as M1 did; M1's ancestry receipt, taken in a full clone, with the branch tip as the line tip (*rev 1.18.9, §15.28*: rev 1.18.2 said "`main`'s roots unchanged (`git rev-list --max-parents=0 main`)", a shallow clone's reading). The squash commit's SHA is recorded in M2's receipt, for the tag step (E5's exit) to tag and check. If later work of this plan (F, U) needs the engine tree, it repeats the M1/M2 pattern: a branch cut from `main`, `<line> = origin/main`, a squash to land | the release tree's cells, as E5: 7.17, 9.9, 9.10 [box, both GPUs] and 7.22, re-run on the squash commit (*rev 1.18.2*) | code review of the squash; roots unchanged; the version reads 1.14.0; references graded with a positive count at the current merge-base if it moved; the squash SHA recorded (*rev 1.18.3*, m2: the tag check moved to the tag step, E5's exit) | `[skip ci] Release 1.14.0: save and restore frozen prefixes (squash of the post-M1 branch at <tip>)` |
| F0 | builder | box | E5 | **Re-anchor on 1.13.0/1.14.0**: the GPU prefix handle type and fields, the host mirror gather of `SLM6`, `UploadPoolRegionsSync` and the latch rule, PKV §4.4's derived table and guard criterion, `SslmGpuStatus`'s tail, the GPU pool's draw and share primitives (§4 item 2; *rev 1.5*: the GPU schema name hash is settled, §4 item 4) | — | recorded in the build log | — (log only) |
| F1 | test author | cloud | F0 | GPU red suite: every [F2]–[F3] cell, stubbed | — | builds on the box, red for the stated reason; coverage audit | `[skip ci] Precached prefixes: GPU red suite` |
| F2 | builder | box | F1 | *Rev 1.18:* built in the cloud on a software GPU (lavapipe) at `83a82c2`, with the readings §4 now states (Q1, Q3) and cell 6.8; the box run on both certified GPUs remains, and runs soon after M1, before other GPU work edits `gpu_1p0.cpp`, so a later regression is attributable. `sslm_gpu_prefix_save`, `sslm_gpu_prefix_restore` (§4), the new GPU status in both mirrors, the derived-table rows, `EntrySubmitGuard`, site-census rows; capture a real GPU `SPX1` save from the same GPU model/adapter fixture, independently check its backend, vendor/device, model hash, precision, schema, domains, exact size and digest, then run 5.7's CPU backend/producer pair against it | every [F2] cell | green on both GPUs, debug layer clean (~60–90 min) | `[skip ci] GPU: save and restore frozen prefixes` |
| F3 | builder, debunker | box | F2 | 4.4, 4.6, 6.5, 10.4 on both GPUs; certification; commission the GPU speed instrument | 4.4, 6.5, 10.4 (4.6 records a finding either way) | tokens equal the GPU twin on both vendors; 10.4's bar met or R1b's fold taken and re-gated (~60–75 min) | `[skip ci] Precached prefixes: GPU certification on both vendors` |
| F4 | as E5 | box | F3; *rev 1.18:* 1.15.0 (the fast-decode release) | Release **1.16.0** (*rev 1.18*: rev 1.17 said 1.15.0); lift the unreleased label; run 9.5; re-run 9.9 and 9.10 against the pinned 1.14.0 and 1.13.1 saves (§5.11 (c)); if 4.6 passed, offer the principal PC-5's one-line relaxation | 9.5 | as E5 (~20 min) | `[skip ci] Release 1.16.0: GPU prefix save and restore` |
| U0 | builder | box | E5; *rev 1.13:* the paging release | Re-pin the plugin to 1.14.0 by the vendoring procedure (`VENDORED_VERSION.txt`). *Rev 1.13:* with paging first (§5.1), U0 starts from the paging release's tree, which vendors 1.13.0 (paging K0R), so the span is one release (1.13.0 → 1.14.0) and the trees named below are history; paging §3.9's seams supply the fill probe, which settles §14's reachability item. **Flag (*rev 1.5*, from E0):** the span depends on which tree U0 re-pins. The public plugin (SuperSLM-Unreal `main`@`7c58a92`, tag `v1.0.0`) vendors v1.9.0: five releases (1.9.0 → 1.14.0), including 1.11's `parallel_split.cpp`. The project's own copy (Wizard `Plugins/SuperSLMUnreal`, `develop`@`4dffa209`) already vendors **v1.11.0** (`8d85b7b`, re-pinned 2026-09-30): three releases (1.11.0 → 1.14.0), with `parallel_split.cpp` already in. U0 names its tree and reads that tree's `VENDORED_VERSION.txt` first; this plan does not choose. Either way PKV's `kv_pages.cpp` and every new source need their vendored wrapper under the completeness check; existing plugin suites green on legacy mode; read the pool sizing code and state the default budget and the extra blocks of §5.4 (PC-11); confirm the fill probe is reachable from the plugin's test module (§14) | — | plugin suites green (~90–120 min) | `[skip ci] Vendor SuperSLM 1.14.0` |
| U1 | test author | cloud (authoring) | U0 | Plugin red suite: every [U2]–[U4] cell, stubbed | — | builds on the box, red; coverage audit | `[skip ci] Prefix cache: red suite` |
| U2 | builder | box | U1 | *Rev 1.18:* also explicit residency control (`WarmPrefix`, `DropPrefix`, `GetPrefixOpResult`, their Blueprint nodes and diagnostics, §5.6), the leased-handle restore hint wired (Q5, 8.15), the one-job CPU build and its build hold (Q7, §8), `AdoptPrefix`'s not-Ready `Malformed` (Q4), the release-status rule (§5.4; *rev 1.18.1*, plan review F1: the U2 draft's V12 code, Wizard `e7811590e`, re-issues a consumed handle and needs the builder's fix here; cell 5.15), issue-ordered op application (§5.6, 3.12) and, *rev 1.18.2*, the mark-application order inside the apply pass (§5.6 *When a mark is applied*, 3.13), the disk part's in-flight-or-waiting coverage and its read serialization (3.10 (iv)–(vii)), and every op's result at re-`Configure()` (1.7), and the `'SLUW'` compatibility table with 1.13.1's row only (Q8; *rev 1.18.1*: the 1.13.0 row dropped). `SuperSLMPrefixCache` (key, tiers, budget, LRU, pins, coalescing, the game-thread ownership rule), budgeted builds with the schema bind (PC-11), the charge rule and `BudgetExhausted`, the pool's extra blocks, one handle per lease, *(rev 1.13)* §5.11: region C through `CacheRegionPages()` and the region-C seam in budget mode, refund after the release applies with *rev 1.14*'s deferred-refund admission (claimed and unclaimed pending refunds, *rev 1.15*; waiters, their coalescing and teardown), adopts and hints through the subsystem's one adopt entry point, and the `'SLUW'` 1.13.0 compatibility entry (PC-13), `AdoptPrefix`'s cache lookup, §5.10's dispositions and `PrefixLeaseEnded`, CPU subsystem wiring, disk wrapper and tier, config, API, Blueprint nodes, the §8 plugin seams | every [U2] cell; *rev 1.13:* paging's [cache] box legs (8.3, 8.4, 8.5 and the cache kinds of 1.2, 1.3, 5, 11) | plugin suites green (~30 min) | `[skip ci] Prefix cache: memory and disk tiers on the CPU backend` |
| U3 | builder | box | U2 | `USuperSLMPrefixSet`, the editor build action, the commandlet, cook validation; *rev 1.18:* `Text` entries, their build-time tokenize pass, `FindEntrySpec` and the `TextChanged` cook warning (§5.8; *rev 1.18.2:* `FindEntrySpec` takes the configured model (*rev 1.18.3*: one signature, §5.8), and it and `RegisterPrefixSet` refuse a set of another model, 2.18), and the build action as a background task with a progress dialog (Q9) | every [U3] cell | green; one cook of the example project (~30 min) | `[skip ci] Prefix cache: cooked prefix sets` |
| U4 | builder, debunker | box | U3 | 4.5, 8.9, 10.5, 10.6, *rev 1.18:* 10.7 and 5.9's one-job build leg, in a packaged build; calibrate the restore per-MiB cost. *Rev 1.18.8:* done at Wizard `15a617f01`; its build leg tripped the slicing trigger (§15.27), and the sliced leg re-runs at U5 | 4.5, 8.9, 10.5, 10.6, 10.7 | green (~60–90 min) | `[skip ci] Prefix cache: packaged-game oracle` |
| U5 | builder, code reviewer, coverage auditor, adversary | box | U4; *rev 1.18.7:* engine 1.15.0 released, and the fast-decode plan's P1 and P1-X merged; *rev 1.18.8:* the sliced CPU build (§5.3 item 4) built, and 5.9's re-graded build leg and 1.7's between-chunks leg green, in one capped review loop (§15.27) | Plugin **1.3.0** (*rev 1.18*: rev 1.17 said 1.1.0; paging took 1.2.0). *Rev 1.18.7 (the principal, 2026-10-21 22:20Z):* it carries the CPU prefix cache and the schema skip-ahead options (canonical and exact), and re-vendors 1.15.0 with `SUPERSLM_WITH_SCHEMA_JUMP_FORWARD` set to 1, after the fast-decode plan's re-vendor check (its §6 P1). The cache cells re-run on 1.15.0 before the tag: docs (`BLUEPRINT_API.md`, `ARCHITECTURE.md`, including the residency nodes and `Text` entries), changelog, version | — | all clear; principal tags (~20 min) | `[skip ci] SuperSLM-Unreal 1.3.0: prefix cache and schema skip-ahead` |
| U6 | builder | box | U5; F4; **the plugin paging plan released** | Re-pin to **1.16.0** (*rev 1.18*: rev 1.17 said 1.15.0); *rev 1.18:* GPU cache builds in token chunks under the lock-holder bound (Q7, 5.13), no re-issue after device loss (Q9, 5.14), region A for nested parents (Q6, 8.14's nested-A leg), and the GPU twins of the residency cells; the GPU subsystem's cache instance over GPU prefixes, keyed by adapter vendor and device; nested entries on both backends (lifting `NestedRequiresBudgetMode`) with ghost charges (§5.4); *rev 1.18.6 (box-fixes review, finding 3), an entry condition:* the GPU host's `SetHoldReleases` must send its held releases on when the gate opens, appending them to the lane jobs in hold order, as the CPU host now does (Wizard `afe45536d`). Today it only sets the flag (`SuperSLMGpuSubsystemPrefixCache.inl`), which would hang U6's gated-release cells as the CPU's hung the 2026-10-21 box run. The U6 red commit fixes it before any gated GPU cell runs | the GPU twins of 1.x–10.x's [U] cells, 1.9, 8.12, 2.12's parent-key case, and 1.x/6.x nested cases (authored in a U6 red commit first); *rev 1.16:* 8.14's nested legs (a waiter pins its parent; an ending ghost is claimable) on the CPU cache, with their mutants; *rev 1.17:* its sibling (D1's release gated behind D2's), chain, parent-after-child and own-parent legs | green on both GPUs | `[skip ci] Prefix cache: GPU backend and nested entries` |
| U7 | builder | box | U6 | GPU entries in cooked sets, built once per certified **device** (2080 SUPER, 7900 XTX) while PC-5's refusal stands; a player's GPU without an entry for its exact device recomputes | GPU 2.9, 9.7 | green | `[skip ci] Prefix cache: cooked GPU prefixes` |
| U8 | builder, debunker | box | U7 | GPU 10.5, 10.6 on both GPUs | — | green (~60–90 min) | `[skip ci] Prefix cache: GPU packaged-game oracle` |
| U9 | as U5 | box | U8 | Plugin **1.4.0** (*rev 1.18*: rev 1.17 said 1.2.0) | — | as U5 | `[skip ci] SuperSLM-Unreal 1.4.0: GPU prefix cache` |

**Why this slicing (PC-1).** The CPU verbs need only PKV's CPU slice, and their build is cloud work, so E0–E4 run
while the box is busy with PKV's GPU slice; only the tag waits for 1.13.0, because releases are numbered in order.
The GPU twins need PKV's GPU prefixes and pool upload, so they follow 1.13.0. Two small releases keep each review
small and let the plugin's CPU cache start on 1.14.0 without waiting for GPU work. If 1.13.0 slips, E5 waits at its
merge and nothing else moves. *Rev 1.18:* the principal renumbered the releases (1.14.0, then fast-decode's 1.15.0,
then the GPU twins' 1.16.0; plugin 1.3.0 and 1.4.0) and the line lands on `main` as one squash (M1), which also
unblocks other GPU work on `gpu_1p0.cpp` once the twins' code is on `main`.

---

## 10. Risks

| # | Risk | Mitigation / trigger |
|---|---|---|
| R1 | CPU: restore with the digest misses the 1,000× bar at the box (the digest runs at SHA-NI speed on the 3950X, not measured) | E4 measures; over the bar → a fold that moves the digest check to a separate verb the plugin runs on a pool thread, restore checking structure only (PC-8's named reversal) |
| R1b | *(Rev 1.1; rev 1.2 re-derived.)* GPU: restore misses 10.4's 100× bar on a certified GPU (most likely 1.5B on the 7900 XTX at 128 tokens, ~8.5× margin on assumed rates, so a miss means a measured rate far below the assumed ones) | F3 measures. Fold 1: PC-8's reversal, as R1 (predicted ~2,083× in that case without the digest; it does not remove the upload fence). Fold 2, if still missed: restate the GPU bar as an absolute latency (a 1,000-token restore ≤ 20 ms at 0.5B and ≤ 40 ms at 1.5B on each GPU), since the purpose is seconds to milliseconds. Each fold is re-gated before F4 |
| R2 | PKV's built code differs from its plan text in something §3 or §4 relies on (field names, header conventions, the fill probe's form, the GPU pool's draw and share), or a reference line moves without such a rule change | E0 and F0 re-anchor before any fixture; a change to a §3 or §4 rule is folded and re-gated before E1/F1 or the next build step. E2–E4 build on `5d5ca58`; *rev 1.13:* since E2's early merge, on `v1.13.0`, whose re-diff found one rule change (F5, folded in rev 1.13), and the reference line moves to `v1.13.0` at E2 because the tag was merged around `<line>` (§8). The reference loader compares its anchor with a ctest-time merge-base through the recorded `<line>` (§8). Every change in that merge-base triggers regeneration and independent licensing before grading, including E5's tag-only line movement; the §3/§4 rule-change re-gate applies only when those rules change. Cherry-pick, squash, hand copy, and a merge through another ref remain the re-diff trigger's discipline |
| R3 | 1.13.0 slips | E5 waits at its merge; U0 waits on E5. Nothing else moves |
| R4 | The plugin paging plan is not commissioned, so the GPU cache never starts | PC-6 surfaces it; §0 and §6 list it as the plan's one unowned dependency |
| R5 | A large restore on a worker lane delays other sequences on that lane | priced by the per-MiB cost (U4) and planned across ticks; late results counted as hitches (§5.5); worst setting recorded at E4 |
| R6 | Every engine release invalidates every blob (PC-7) | cooked sets rebuild at cook; disk entries are misses and are rewritten; the cost is a recompute per prefix once per engine upgrade |
| R7 | *(Rev 1.1; replaces rev 1's legacy-build reserve risk, removed by PC-11.)* F3's cells 4.6 show GPU K/V bytes differ across vendors or from the CPU | PC-5's refusals stay; cooked GPU entries stay per device; the finding goes to the principal against `docs/platform-support.md:311-315` |
| R9 | *(Rev 1.3.)* PC-12 refuses a model a consumer ships today, one with colliding or zero-hash schema names | E0's audit clears the certified artifacts. For others the release note names the refusal and the fix: rename one schema and re-convert. Any 64-bit collision has to be built on purpose; the chance of one arising by accident is about 2^-64 per pair |
| R10 | *(Rev 1.13.)* 1.14.0 has to change a sequence save format, or the forward path, after all, so PC-13's compatibility entry cannot cover it | 9.9 or 9.10 goes red at E5. That format leaves the entry, the release note names it, and its saves are refused `Layer1Mismatch` by name as before (paging R11); nothing is misread |
| R8 | *(Rev 1.1.)* E0, E2 and E4 need about 2 hours of box time while PKV's GPU slice holds the box | they are short runs placed between PKV's runs; if the box is not free, E5 waits, and nothing else moves |
| R11 | *(Rev 1.18, PC-14; rev 1.18.1.)* 1.14.0's tree, and 1.15.0's after it, carry the GPU twins exported but uncertified (F2 ran on a software GPU only), so a consumer can call verbs no release promises | they are labelled unreleased until 1.16.0 in every file (§4 preamble), checked by 7.22 at M1, at E5's publish gate and at the 1.15.0 release's publish gate (plan review F4, F5; the cross-plan note in `reviews/fold-rev1p18p1.md`); F2's box run follows M1 soon after |
| R12 | *(Rev 1.18.)* The home box crashes on long intense runs, and the long legs (4.1 grid ~7.5 h, 4.2/E3 on MSVC, the unpinned 1.5B legs ~5 h, not checkpointed) would lose their progress | they run on the work machine (§1, §9); checkpointed legs on the home box only in resumed runs of at most 15 minutes |
| R13 | *(Rev 1.18, PC-14; rev 1.18.1, plan review F2.)* The squash removes the line's history from `main`, and a reference line pinned to the squash commit would never move again, so other plans' commits on `main` would land unseen under E3–E5's grading | post-M1 work runs on a branch cut from `main` with `<line> = origin/main`, so every `main` commit taken in moves the merge-base and regenerates (§8, must-reject (5)); M1 and M2 record their tip SHAs in their commit messages |
| R14 | *(Rev 1.18.1, plan review F4.)* The fast-decode line pre-bumped its CMake to 1.15.0; once it takes `main`, this plan's `sslm_version.h` and its equality check force its header to 1.15.0 too, so its development builds write `SPX1` blobs with the 1.15.0 release's producer before 1.15.0 exists (PC-7's hazard, on another plan's line) | not this plan's to fix: the cross-plan note (`reviews/fold-rev1p18p1.md`) asks that plan to carry the last released version (1.13.1, then 1.14.0) until its own release commit, as PC-7 states |
| R15 | *(Rev 1.18.2; the strike on rev 1.18.1, lesser plane 4.)* A GPU prefix release given another context's handle returns `SSLM_SEQUENCE_KV_BUFFER_MISMATCH` without consuming it, the status a consumed underflow returns, so the plugin would count it applied and refund the pages of a handle that is still live | known limit, recorded in §5.4: the precondition is a plugin bug (the cache releases only handles made on its own context), which debug builds assert; nothing in the cache tries to separate the two cases |

---

## 11. Estimate

**Underived**: carried from the size of the work, not measured. Elapsed assumes the gate path clears in one round.

| Stage | Elapsed | Principal's time at the box |
|---|---|---|
| P1 gate path | 1–2 days | 15 min (decisions) |
| E0–E5 (CPU verbs, 1.14.0) | 3–4 days, in parallel with PKV's GPU slice (box time contends, R8) | E0 45 min, E2 20 min, E4 45 min, E5 20 min. *Rev 1.13:* E2 adds its box legs, including the full 4.1 grid (about 7.5 h of box time, unattended, per the re-anchor record) and the MSVC ASan triage; E5 adds 9.10's GPU capture and run on both cards (~20–30 min, underived) |
| M1 (squash to `main`, *rev 1.18*) | under a day, cloud and box | about 15 min at the box (*rev 1.18.1*: `c_abi_verbs.txt` on Windows and the positive graded count); the reference regeneration at the new anchor is E2's receipt (5) moved, not added |
| M2 (second squash, *rev 1.18.1*) | under a day, cloud and box (*rev 1.18.2*: rev 1.18.1 said cloud) | *rev 1.18.2:* 9.10's re-run on both cards, as E5's (~20–30 min, underived); plus receipt (5)'s regeneration if a `main` commit taken in forces one |
| F0–F4 (GPU twins, **1.16.0**; *rev 1.18*: rev 1.17 said 1.15.0) | 2–3 days | F0 20 min, F2 60–90 min, F3 60–75 min, F4 20 min |
| U0–U5 (plugin **1.3.0**; rev 1.17: 1.1.0) | 3–4 days; *rev 1.18:* plus about a day for residency control and `Text` entries (underived) | U0 90–120 min, U2 30 min, U3 30 min, U4 60–90 min, U5 20 min; *rev 1.18:* U2 +10 min, U3 +10 min, U4 +15 min (10.7) |
| U6–U9 (plugin **1.4.0**; rev 1.17: 1.2.0) | 2–3 days, after the plugin paging plan | 2–3 h |

Total principal's time, E0–U5: about 7–8.5 hours; with U6–U9, about 9–11 hours (*rev 1.1*: U0 and F3 grew). Whether both GPUs are in one
machine or need a swap is not known to this plan; a swap adds its time to F2, F3 and U8. *Rev 1.18:* the residency and
text cells add about 35 minutes through plugin 1.3.0 (so about 7.5–9 hours), and the long legs move to the work
machine: they are unattended machine time there, not the principal's time at the home box, though starting and
collecting them is. All figures stay underived.

---

## 12. Decisions

Each is the planner's conservative default where the ask forked, or the principal's stated default. Each can be
overruled in one line.

| # | Decision | Costs / buys / forecloses | Recommendation |
|---|---|---|---|
| PC-1 | *(Principal's default.)* Sequenced after PKV: CPU verbs built after 1.12.0 and released as **1.14.0** after 1.13.0; GPU twins **1.16.0**; plugin **1.3.0** (CPU cache; *rev 1.18.7, the principal 2026-10-21 22:20Z:* also the schema skip-ahead options, re-vendoring 1.15.0) and **1.4.0** (GPU cache). *Rev 1.18: renumbered by the principal 2026-10-20* (rev 1.17: GPU twins 1.15.0, plugin 1.1.0 and 1.2.0). 1.15.0 is the separate schema fast-decode release; plugin 1.2.0 is the paging release | Buys small releases and CPU work overlapping PKV's GPU slice. Costs one extra release cycle against a single combined 1.14.0 | Confirmed (numbering decided) |
| PC-2 | *(Principal's default.)* `sslm_prefix_save` / `sslm_prefix_restore` and GPU twins; restore builds a frozen handle with no prefill; blobs self-contained (rows `[0, L)`); a nested blob may re-share a live parent under the byte-compare rule | Self-contained nested blobs cost the parent's rows again (7.03 MiB against 1.17 MiB per persona at the cohort shape) and buy a blob that restores alone | Confirm |
| PC-3 | *(Principal's default.)* Plugin cache: memory tier with LRU under a byte budget, asynchronous disk tier, cooked sets; key over model hash, engine identity, backend, format, adapter (none), schema, tokens, parent; any mismatch or corruption recomputes | — | Confirm |
| PC-4 | *(Principal's default.)* Out of scope: automatic matching, a pre-paging stopgap | — | Confirm |
| PC-5 | *(Re-decided in rev 1.1.)* One `SPX1` format with `backend` and producer-adapter fields. A blob restores only on its own backend and, on GPU, only on the adapter vendor and device that wrote it, until F3's cells 4.6 pass | Premise, corrected: `docs/platform-support.md:311-315` claims CPU/GPU K/V identity, but at source it is backed only at position 0, for one token, in a run-only package (S21). No test pins the bytes over a span, so the conservative default stays and 4.6 adds the byte-identity cells either way. Costs a cook per backend and per GPU **device**: a cooked GPU entry hits only on the exact 2080 SUPER or 7900 XTX it was built on, and every other card, same vendor included, recomputes (*rev 1.2*). Reversible in one line each at F4 | Confirm |
| PC-6 | The plugin's move onto paged KV (budget-mode CPU sequences and prefixes, pooled GPU sequences, GPU prefix verbs) is its own plan, not folded here. The CPU cache ships on the plugin's current mode (adopters copy `O(L)`, no page sharing, no nested entries); the GPU cache and nested entries wait for it | Buys a plan of proportionate size and a CPU cache that does not wait. Costs: until that plan ships, no page sharing and no GPU cache. Folding it here instead adds an estimated 2–4 days and a larger gate path | Confirm, and commission the plugin paging plan as the next planning task |
| PC-7 | The blob records its producer (engine version; on GPU the adapter vendor and device); a different producer is refused with new `SSLM_RESTORE_PRODUCER_MISMATCH` (CPU ordinal after PKV's; GPU appended). The version comes from a checked-in header kept equal to CMake's by a CI check | Buys: a release that changes any K/V numerics cannot turn a stale blob into different output for any host, not only the plugin. Costs a recompute per prefix per engine upgrade. It separates **releases only**: between releases CMake carries the last released version, so two development builds on one line share a version. *Rev 1.3 (the strike's lesser plane):* the plan does not widen the producer; it scopes promise (i) to the same release version or the same development build (§3.4). The plugin separates development builds by the Layer-1 commit in its key and wrapper. Reversal, if the principal wants engine-level separation: add the build's commit to the producer, which needs a generated header in every build system, the plugin's included. Alternative: accept any version, as sequence blobs do, relying on the bit-exact law | Confirm |
| PC-8 | The blob carries a SHA-256 of itself, verified first on every restore | Buys detection of any accidental change (disk, packaging) in the engine, for every host; not authenticity (§3.4). Costs one SHA-256 pass per restore (SHA-NI on the box; not measured). Reversal if R1 or R1b fires: a separate verify verb the plugin runs off-lane | Confirm |
| PC-9 | Only a frozen prefix can be saved (`SSLM_INVALID_ARGUMENT` otherwise, as adopt) | Buys one kind of restored handle (frozen). Forecloses resuming a prefix's construction from a blob, which no consumer has asked for | Confirm |
| PC-10 | Plugin disk tier off by default (`bPrefixDiskCacheEnabled = false`); memory and cooked tiers on | Buys no unasked-for disk writes on a player's machine. Costs a recompute after restart for uncooked prefixes until a game turns it on | Confirm |
| PC-12 | *(Rev 1.3; from the strike.)* In 1.14.0 the schema-section parser refuses a model with two equal non-zero `Fnv1a64` name hashes (`DuplicateSchemaNameHash`) or a name hashing to 0 (`ZeroSchemaNameHash`), on both backends' model map. The converter refuses to emit such an artifact | Buys: one structural invariant makes the stored name hash an exact schema identity. It closes the planned `SPX1` restores on both backends, the plugin's cooked and disk tiers, and the shipped `SSB5` path, with no blob-layout change, and it protects every consumer of `ByNameHash`. Costs: a model with colliding or zero-hash names, which maps today, is refused from 1.14.0; its owner renames a schema and re-converts. E0 audits every certified artifact and the plugin's example assets first (§8). Reversal: store the exact schema name in `SPX1` (a variable-length field) and repair `SSB5` separately, which keeps colliding names legal at the cost of two format changes | Confirm |
| PC-11 | *(Rev 1.1; rev 1.2 adds the charge rule.)* The cache builds with `sslm_prefix_begin_budgeted(L)` in the plugin's current pool; cached entries do not count against `PrefixBlockCount`; the pool gains `ceil(PrefixCacheBudgetBytes / block bytes)` blocks at `Configure()`; every cache build or restore is charged at its peak draw before it draws, and one that cannot be charged resolves `BudgetExhausted` | Buys, exactly: no whole-context reserve per build, so rev 1's build-concurrency rule and its legacy-reserve risk go; one limit (bytes) on the cache instead of a count beside it; the cache's pages, resident and in flight, never exceed the extra blocks, so 1.0.0's `BlockCount` warm sequences and `PrefixBlockCount` live prefixes are admitted as in 1.0.0 with every entry leased and builds in flight (8.10), while adopters copy (to U5). Costs: a request beyond the budget with every entry pinned is refused rather than served. Costs the budget's memory added to the pool at `Configure()`. Alternative: legacy begin and a `PrefixBlockCount` build cap, as rev 1 | Confirm |
| PC-13 | *(Rev 1.13; from the plugin paging plan's cross-plan note, its R11.)* 1.14.0 keeps 1.13.0's sequence save formats (`SSB5`, `SSB6`, `SLM5`, `SLM6`) byte for byte, proved at E5 by cells 9.9 and 9.10, and the CPU-cache release's `'SLUW'` reader accepts a wrapper tagged 1.13.0 around one of those payloads (§5.11 (c)). **Accepted 2026-10-15.** *Rev 1.18 (principal, 2026-10-20):* the predecessor is **1.13.1** (`d697927`), which the paging release vendors; 9.9 and 9.10 capture on `v1.13.1`, and the table's 1.13.1 row is the one the paging release's saves meet | Buys: players' saves from the paging release load in the CPU-cache release, so the order the principal's delegation chose (paging first, UP-1) costs one save boundary (1.11.0 → 1.13.0), not two. Costs: one table entry and its cell in the plugin (9.11); two engine cells and a pinned set of 1.13.0 saves; ~20–30 box minutes at E5 for the GPU half; and a constraint on 1.14.0: if it must change a sequence format after all, that format keeps the boundary by name (R10). Forecloses nothing: every other tag is still refused `Layer1Mismatch`, and the cache's own `SPX1` blobs keep PC-7. Alternative: keep the second boundary (paging R11), refused by name and never misread, with no plugin or engine work | Accepted |
| PC-14 | *(Rev 1.18; **decided by the principal 2026-10-20**; rev 1.18.1 records the conductor's rulings on the points rev 1.18 left open.)* The line lands on `main` as **one squash commit** (`main` was re-rooted on 2026-10-12 into parentless snapshots), step M1 (§9). **(a) When:** early, inside E2, after E2's code review (receipt (8), covering the four unreviewed changes) and its MSVC builds, before E2's reference grading. This plan's later work runs on a post-M1 branch cut from `main` with `<line> = origin/main` (§8) and lands by a second squash, M2, before E5's tag. **(b) The version at the squash** is not a choice (plan review F11): approved PC-7 fixes it at 1.13.1, the last released version; E5 bumps both header and CMake. **(c) The GPU twins** are exported and labelled unreleased until 1.16.0 in every file, checked by 7.22 at M1, E5 and the 1.15.0 release | (a) buys `main` early, so other GPU work on `gpu_1p0.cpp` starts from it; costs a second squash (M2), a reference regeneration at each `main` commit taken in, F2's box run becoming a soft gate on other plans' `gpu_1p0.cpp` edits, and 1.15.0 tagging from a `main` that carries this plan's code (R11, R14). (c) exports verbs no release promises (R11); a macro gate would make 1.14.0's ABI inventory depend on configuration | Ruled (conductor, 2026-10-20); the principal may overrule in one line |
| PC-15 | *(Rev 1.18; **accepted by the principal 2026-10-17 14:51Z**, `build-2026-10-17/open-questions.md`.)* The nine build recommendations, each folded as recommended: **Q1** a GPU prefix stopped after landing tokens, on the budget or on a schema rejection, refreshes its prefill snapshot (§4 item 7, cell 6.8; `fdbb97b`, `83a82c2`). **Q2** the page-size qualifier is in promise (i)'s sentence (§0, §3.4, §13 item 2, 7.1). **Q3** F2's GPU rules as built: `current_token` on the handle, `kv_precision` on the model, the at-rest rule, the derived ready/snapshot/bind state, one status for a share refusal (§4 items 1–3; 5.11's GPU half settled). **Q4** `AdoptPrefix` on a lease not yet Ready is `Malformed` in plugin 1.3.0 (§5.10, 7.21). **Q5** the leased-handle restore hint is wired at U2 on the box (§5.11 (b), 8.15). **Q6** region A for nested parents is built at U6 (§5.11 (a), 8.14). **Q7** CPU builds are one lane job, GPU builds token chunks under the lock-holder bound (§5.3 item 4, 8.10 (c), 5.9, 5.13). **Q8** each compatibility row lists the release commit and the tag's peeled commit (§5.11 (c), 9.11). **Q9** no re-issue of a skipped release after GPU device loss; the editor build as a background task (§5.4, §5.8, 5.14, 3.11) | Q7's CPU half supersedes §5.3's rev 1.8 rule (begin as its own job), so 8.10 (c) reads its drawn-and-unfrozen window through a new in-job build hold (§8) and needs at least K lanes (§14). Q9's first half rested, in rev 1.18, on the U2 draft's failed-release rule (V12), which is false against the engine (both release verbs consume the handle on a failing status, the GPU's except on `SSLM_BUSY`); rev 1.18.1 withdraws it | Folded. *Rev 1.18.1:* the 1.13.0 row is **dropped** (conductor's ruling; no released plugin carried 1.13.0), with 9.11's foreign-commit case on the 1.13.1 row; Q9's premise is withdrawn and replaced by the release-status rule (§5.4, plan review F1) |
| PC-16 | *(Rev 1.18; **folded on the principal's word 2026-10-20 20:52Z**.)* Two additive plugin API additions: **(a)** explicit residency control, `WarmPrefix` (a request without a lease) and `DropPrefix` (memory, optionally disk), with `GetPrefixOpResult` and Blueprint nodes (§5.6); **(b)** Prefix Set entries defined by text, tokenized at build in the editor or commandlet against the asset's model, with the token ids stored and still the runtime key (§5.8; the §6 row removed) | Buys: a game can stage prefixes ahead of a level and free memory at will, and designers can author entries as text. Costs: about a day of plugin work and the cells 1.10, 2.16, 2.17, 3.10, 5.12, 7.18–7.20, 8.14's drop legs, 9.12, 10.7, 11.7. Changes no existing promise: a drop is an eviction decided outside a request's loop, so §5.11 (a)'s induction holds unchanged (executed, `drop_model.py`); a drop never ends or invalidates a lease, and never cancels engine work. Forecloses nothing; a synchronous residency query and an in-flight cancel stay left out (§6). *Rev 1.18.2 (the strike on rev 1.18.1):* a deferred mark waits behind earlier pending hashes, and the disk part covers in flight or waiting (§5.6). `RegisterPrefixSet` and `FindEntrySpec` refuse a set of another model (§5.8), which rev 1.18.1 accepted and served. Cells 1.7, 2.18, 3.10 (iv)–(vii), 3.13 | Folded |

---

## 13. Gate path

1. **Plan review** of this plan (temper), and a **coverage audit** of §7 against the catalog.
2. **Adversary strike** on the load-bearing promise, briefed with the promise and this plan only. The strike on
   rev 1.1 fractured (i) with colliding schema-name hashes. Rev 1.3 folds it (PC-12) and states the promise as:
   *"(i) for every blob this library's own `sslm_prefix_save` wrote, restored by a build of the same release version
   (or, between releases, the same build), against a model that maps (which requires distinct, non-zero schema-name
   hashes, PC-12), a restore that returns `SSLM_OK` yields a prefix whose every adopter emits exactly the tokens an
   adopter of the prefill-built prefix emits; (ii) any accidental change to such a blob is refused, barring
   a SHA-256 collision. Authenticity is not promised: in a blob edited with its digest recomputed, every field not
   fixed by steps 2–6 is attacker-chosen within the limits steps 7–8 enforce (§3.4)."* (*Rev 1.2:* rev 1.1's "lists
   every field" missed the schema binding, `L` and `origin`.) *Rev 1.18 (PC-15 Q2, approved 2026-10-17):* (i)'s
   sentence now reads "…yields a prefix whose every adopter, in a pool of the restore pool's page size, emits exactly
   the tokens an adopter of the prefill-built prefix emits in a pool of that page size", with an adopter in a pool of
   another page size refused `SSLM_INVALID_ARGUMENT` (X13). This moves rev 1.13's third scope statement into the
   sentence; it narrows nothing the strike was handed after rev 1.13.
3. **The loop ran and ended here (rev 1.4).**
   - Plan review and coverage audit, round 1 → rev 1.1.
   - Round 2 re-checks → rev 1.2.
   - The strike on rev 1.1 FRACTURED (i) with colliding schema-name hashes → folded in rev 1.3 as PC-12.
   - The re-strike on rev 1.3 **held (i) and (ii)**. It FRACTURED only §3.4's absolute admission claim for nested
     prefixes restored privately. That was folded in rev 1.4 as text and cells (§3.4 *Peak draw*, §3.5, §5.4,
     7.4, 5.10, 8.12), under a ruling delegated to the conductor, not a new PC.
   - No further gate round runs on rev 1.4.
4. The principal approves, confirming or reversing PC-1..PC-12. *(Done for rev 1.4, 2026-09-30: all twelve
   confirmed as recommended.)*
5. *(Rev 1.5.)* E0's folds (§15.5) were made after every gate above ran, so they re-enter those gates, scoped to the
   delta, before E1. The plan review and the coverage audit take every row §15.5 lists: §2, §3.1, §3.4 (*Draw and
   build*, step 8), §3.7, §3.8, §4 items 2–4, cells 2.13, 5.6, 5.7, 7.4 and 11.1, §8, §9's E0, E2, F0 and U0 rows, and
   §14. The delta changes neither the promise quoted in item 2
   nor any PC. Whether the adversary re-strikes is the conductor's call. The strike's promise text is unchanged, and
   the delta's one admission-relevant change is the order of draw and share. That change leaves the pages drawn at
   `ceil(L/B) − s`. *(Done: plan review T-3023, temper; coverage audit T-3024, gaps named; folded in rev 1.6, §15.6.)*
6. *(Rev 1.6.)* Rev 1.6's folds (§15.6) re-enter the plan review and the coverage audit, scoped to the delta,
   before E1. The rows: the header and §0 *Releases*; §2 S20, S26 and S29; §3.4 *Draw and build* steps 2–4, the
   step-8 `m` citation, and *So three refusals*; §3.6 (dispositions, X12); §3.7; §3.8 (*Status*, the converter's
   no-opt-out paragraph); §4 preamble and item 2; cells 1.4, 1.8, 1.9, 2.3b, 2.13, 5.1, 5.2, 5.6, 5.7, 5.11, 7.2,
   7.4, 7.12, 7.16, 7.17, 8.10 and 11.1; §8 (the reference base and re-diff, the reading rule, the independent
   `SCM1` writer, the fault slot, the lane-park seam, the docs check); §9's E0, E1, E2 and E5 rows; R2; §14.
   The delta changes neither the promise quoted in item 2 nor any PC, and the draw count stays `ceil(L/B) − s`.
   Whether the adversary strikes is the conductor's call. *(Done: plan review T-3026, temper; coverage audit T-3027,
   gaps named; folded in rev 1.7, §15.7.)*
7. *(Rev 1.7.)* Rev 1.7's folds (§15.7) re-enter the plan review and the coverage audit, scoped to the delta,
   before E1. The rows: the header; §2 S27; §3.8 (*Status*, the test-side constant and its macro, the converter
   paragraph's writer sentence); cells 1.4, 1.7, 1.9, 2.10, 5.2, 5.5, 5.6, 5.11, 7.17 and 8.10; §8 (the reference
   base's structural half, the reading rule's lists and its initial-count clause, the certified-artifact audit, the
   independent `SCM1` writer, the fault slot's records and must-rejects (3) and (4), the lane-park seam, the docs
   check, the 1.0.0-prefix counter); §9's E0, E1, E2 and E5 rows; R2; §14. Of these, E1 needs the rows its red
   suite is written from: 5.2, 5.6, 5.11, the fault slot, §3.8's constant, the E0 and E1 rows. The delta changes
   neither the promise quoted in item 2 nor any PC, and the draw count stays `ceil(L/B) − s`. Whether the
   adversary strikes is the conductor's call.
8. *(Rev 1.8.)* Rev 1.8's folds (§15.8) re-enter plan review and coverage audit, scoped to the delta,
   before their dependent steps. Rows: the header; §5.3; cells 7.1, 7.17, 8.10 and 10.1; §8's pinned
   references, structural half, initial-count rule, certified-artifact audit, independent writer, lane-park,
   docs check and 1.0.0-prefix counter; §9 E0–E5; R2; §15.8. E0 commissioning rows gate E1; regenerated
   reference rows gate E2's first grading; 8.10 and lane-park rows gate U1. The approved promise quoted in
   item 2 and all PC decisions remain byte-identical to rev 1.7. The scoped gate owns independent review of
   the whole coverage catalog against the mechanisms added here; the planner does not certify this fold.
9. *(Rev 1.9.)* Rev 1.9's folds (§15.9) re-enter plan review and coverage audit, scoped to the delta,
   before their dependent steps. Rows: header and §0; §3.8's writer wording; cells 2.13, 6.1, 6.4, 7.1 and 8.10;
   §8's reference trigger, loader commissioning, audit and writer facts; §9 E0–E2 and E5; R2; §14; §15.9.
   E0's commissioned audit and writer receipts and eight verdicts gate E1. The current-token oracle design and runnable
   runtime reds gate E1's suite; the public-macro vitality check gates E2 after the macro is defined. Reference
   regeneration gates E5 grading after any merge-base change; the 8.10 one-page must-reject and independent
   commissioning gate U2. The approved promise quoted in item 2 and all PC decisions remain byte-identical to
   `d460d3a9ae`. The scoped re-check reviews the full coverage catalog against this fold; the planner does not
   certify it.
10. *(Rev 1.10.)* Rev 1.10's F16 source resolution and E1/E2 gate sequencing (§15.10) re-enter plan review
    and coverage audit, scoped to S20; §3.4 step 8 and its attacker-chosen `m`/`e` interpretation; cells 2.3b,
    6.4 and 7.1; §9 E1/E2; §14; and this fold. E1 authors 6.4's isolated token mutant and the 2.3b scale
    equality cell. E2 runs them after built verbs reach their assertions, then runs the T-3060 F2–F15
    discrimination audit and its E2 mutants before declaring E2 green. The approved promise quoted in item 2,
    §0's *The promise, exactly*, §3.4's *What is promised, exactly*, §12 and all PC decisions remain
    byte-identical to `d460d3a9ae`. The planner does not certify this fold.
11. *(Rev 1.11.)* T-3062 S-1 and T-3063 F1–F3 (§15.11) re-enter plan review and coverage audit, scoped to
    the header; §7's red-first preamble and 7.12 independent oracle; §8's D-SLM8017 receipt; §9 E0–E4;
    §15.11; and this gate. The independent coverage pass crosses the full §7 catalog and the T-3060 F1–F15
    assertion design. E1's complete authored red suite and coverage recheck gate the **start** of E2 product
    implementation (D-SLM8022). E2's built-verb assertion reach, F1 format discrimination, 2.3b scale
    equality, 6.4 baseline-then-isolated-token mutant, and T-3060's applicable E2 mutants and independent
    discrimination re-audit gate E2 **green**. The E1-authored E3/E4 assertions execute and receive their
    independent discrimination re-audits before their respective green verdicts. Reference-population licensing
    remains a separate prerequisite before E2's first grading. The approved promise quoted in item 2,
    §0's *The promise, exactly*, §3.4's *What is promised, exactly*, §12 and all PC decisions remain
    byte-identical to `d460d3a9ae`. The planner does not certify this fold.

12. *(Rev 1.12.)* T-3070 A1/A2 (§15.12) re-enter plan review and coverage audit, scoped to the header,
    cells 5.7 and 7.12, §9 E1/E2/F2, this gate and §15.12. The independent pass checks the E1-authored
    count values, the B-cross-view comparison and the constructed backend/producer precedence pair against
    the full §7 catalog. E2 must execute those assertions on built CPU verbs; F2 must capture and check the
    real GPU operand before its GPU verdict. E1 clearance under D-SLM8022 still requires the test author's
    A1–A4 repairs and the coverage auditor's recheck. The approved promise and PC decisions remain as filed.

13. *(Rev 1.13.)* The E2 re-anchor on 1.13.0 and the paging plan's cross-plan notes (§15.13) re-enter plan review
    and coverage audit, scoped to: the header; §0's PC-13; §2 S26 and S30; §3.1; §3.2's table; §3.4 step 3, promise
    (i)'s third scope statement, *Peak draw* and *What adopters see*; §3.5's cohort; §3.6 (*keeps*, X1, X13);
    §3.8 (sequence formats, header text); §5.1, §5.4's three notes, §5.6, §5.9 and §5.11; the §7 preamble and cells
    4.7, 7.12, 8.10, 8.13, 8.14, 9.8–9.11, 10.1–10.2 and 11.1; §8's base, line switch and executed figures; §9 E2–E5,
    U0 and U2; R2, R10; §11; §12 PC-13; §15.13. **F5 is the one §3 rule change**, which is what makes this a re-gate
    and not a text note (§8, R2). Whether the adversary strikes the third scope statement is the conductor's call:
    the sentence of (i) and item 2's quoted promise are unchanged, and the scope statement narrows no case that
    existed at approval (§15.13). The coverage pass rules on whether 4.7, 8.13, 9.8 and 7.12's pool-kind leg close
    the re-anchor's coverage gap, and on E4's re-derivation. **PC-13 is the principal's**, with the recommendation
    *Accept*. This fold re-gates before E3 builds on it, and before E2's green. The planner does not certify this
    fold.

14. *(Rev 1.14.)* Both seats returned **HOLD** on rev 1.13 (plan review: B1, S1–S4, N1–N6; coverage audit: B1,
    S1–S7, N1–N9; §15.14). The fold re-enters plan review and coverage audit, scoped to: §3.1 and §3.4's page-size
    qualifier; §3.4's *Peak draw* and *What adopters see* citations; §3.5's contiguous-pool hint; X13; §5.4's charge
    rule, §5.5's delivery note, §5.9's row and §5.11 (a)'s **deferred-refund admission**; the §7 preamble's
    unpinned-cell harness rule; cells 4.7, 6.6, 7.12's pool-kind leg, 8.10, 8.13, 8.14, 9.8 (and its new nested leg
    [E3]), 9.9, 9.11, 10.1 and 10.2; §9 E2 (receipt (6)), E3, E4 and U2; §14; §15.14. **No §3 rule changes**:
    §3.5's contiguous-pool sentence states what "B is the restore pool's" (rev 1.13) already implied for the hint.
    §5.11 (a) is a new admission rule in the plugin, stated as one; it leaves PC-11's buys and costs unchanged
    (§5.11 (a), *Why the principal's decision is unchanged*), so the plan review rules on that claim and no strike is
    needed. Clearing this re-gate is what item 13 required before E3 builds and before E2's green. The planner does
    not certify this fold.

15. *(Rev 1.15.)* The re-check of rev 1.14 (`reviews/recheck-rev1p14.md`) returned **HOLD** on two items (R1, R2;
    §15.15). The fold returns to that re-check, scoped to: 10.1's E4 mechanism, the §7 preamble's note and §9 E4
    (R1); §5.11 (a)'s claim rule and its *Why the principal's decision is unchanged* argument, §5.4's charge rule,
    §5.9's row, 8.14's new legs and mutants and the waiters leg's stated condition (R2); §14's settled cap item; and
    §15.15. No §3 rule changes. The re-check rules on whether the induction in §5.11 (a) holds; if it does not, the
    added refusal goes to the principal as a PC-11 amendment. The planner does not certify this fold.

16. *(Rev 1.16.)* The blind strike on rev 1.15 (`reviews/strike-rev1p15.md`) returned **FRACTURED** on §5.11 (a) for
    nested entries (F1, F2; U6), with three lesser findings (L1–L3); promise (i)'s page-size scope, X13, PC-13's CPU
    half and the rule-B figures held on executed engine runs (§15.16). The fold re-enters the strike's scope for
    re-strike, and the plan review and coverage audit for: §5.4 *Pins*; §5.11 (a)'s waiter pin, generations,
    *Ghost charges under the deferred refund* and the extended induction with its executed model; 8.14's L1 clause
    and nested legs [U6]; §9 U6; §3.8 and 9.10 (L2); 10.2 (L3); §15.16. No §3 rule changes, and no PC: both fixes
    restore the approved rule's decisions for nested entries (§5.11 (a), *Executed*). The planner does not certify
    this fold.

17. *(Rev 1.17.)* The re-strike on rev 1.16 (`reviews/restrike-rev1p16.md`) returned **FRACTURED** on the ghost fix
    (R1) with one lesser finding (R2); F1 and L1 held (§15.17). Per the conductor, this fold's re-check is the
    re-strike's own model and generators run with the fix (`reviews/fold-rev1p17-probes/`), and no further strike
    round follows unless that run shows a failure. It shows none: 0 divergences and 0 physical overdraws on its three
    seeds with releases out of order. The changes are §5.4 *Pins* (the own-parent reading), §5.11 (a) (*Pending
    refunds wait on their family*, the ordering bullet, completion in *No double counting*, the executed record) and
    8.14's four new legs. The own-parent reading for builds is disclosed in §5.4 and §15.17; it does not change
    PC-11. The planner does not certify this fold.
18. *(Rev 1.18.)* Rev 1.18 folds the principal's own decisions (PC-14, PC-15, PC-16, PC-1 renumbered, PC-13
    retargeted; §15.18) and re-enters a **scoped re-check**: plan review and coverage audit, on the rows §15.18 lists,
    before the steps they gate. M1 and E2's remaining receipts need §3.8, §4's preamble, §8's line rule and §9's E2
    and M1 rows; U1 needs §5.3–§5.11's rev 1.18 text and every new [U] cell; F2's box run needs §4 items 1–3 and 7
    and cell 6.8. The promise's substance is unchanged (Q2 moves a scope statement into the sentence), and the one
    admission-relevant addition, the drop, is executed against §5.11 (a)'s induction (`reviews/fold-rev1p18-probes/`).
    Whether the adversary strikes the residency rules is the conductor's call; the drop model's mutants m1 and m2
    are the obvious constructions and both are caught. PC-14's three open points go to the principal. The planner
    does not certify this fold.
19. *(Rev 1.18.1.)* Item 18's re-check returned TEMPER (plan review, F1–F16) and gaps named (coverage audit, G1–G20,
    N1–N6); the conductor ruled the open points (§15.19). Rev 1.18.1's delta re-enters a scoped re-check before the
    steps it gates: M1 needs §8's post-M1 rule with must-reject (5) and must-accept (5), the M1 and M2 rows, 7.22;
    F2's box run needs 6.8's kind-0 legs, 8.16 and 11.8; U1 needs §5.4's release-status rule, §5.6's issue order and
    op results, §5.8's per-entry rule and every new or changed [U] cell. The promise is unchanged. The planner does
    not certify this fold.
20. *(Rev 1.18.2.)* The adversary struck rev 1.18.1's residency, text-entry, release-rule and squash planes and
    returned FRACTURED (`reviews/strike-rev1p18p1.md`: two fractures, eight lesser planes). The conductor ruled each
    one, and rev 1.18.2 folds them (§15.20). The delta re-enters a scoped re-check before the steps it gates:
    - U1 needs §5.6 *When a mark is applied*, the disk part, *Generations and teardown*, §5.4's two rev 1.18.2
      bullets, §5.8's `FindEntrySpec` and model check, and cells 1.7, 2.17, 2.18, 3.10 (iv)–(vii), 3.13, 5.15 and
      7.18;
    - E5 and M2 need §9's E5 and M2 rows.

    The pass bound in *When a mark is applied* goes beyond the ruling's words, and §15.20 says why. The re-check, or
    a re-strike of Fracture 1's plane if the conductor calls one, should read it. The promise is unchanged. The
    planner does not certify this fold.
21. *(Rev 1.18.3.)* The adversary's capped re-check of rev 1.18.2 HOLDS (`reviews/strike-rev1p18p1.md`, *Re-check*),
    with three minor findings, m1–m3, all folded as text (§15.21). The conductor keeps the pass bound. No rule or
    accepted outcome changes, so no further re-check is owed for this delta.
22. *(Rev 1.18.4.)* A text-only erratum on the approved rev 1.18.3, from the suite audit of the E4, E5 and U2 red
    suite and the red suite's readings (§15.22). No promise or PC changes. Acceptance forms change where the
    findings require it: 7.17's locator reads statements split across two lines, and 9.9's bound cases are graded
    in outcome form. 6.6's mutant and 8.16's twin construction are restated, and 6.8 gains the built legs A, B and
    D. §5.9 names the warm and drop rows' reason codes, adding `DropDeleteFailed`, and 5.12 grades them in 5.4's
    form, which is a third changed form. The carried T-3060 F10–F13 E4 portions are found and folded (§7). The test
    author aligns 6.6's comment, 8.16's twin and 5.12, and maps F10–F13. The debunker commissions 7.17's new
    must-rejects. The planner does not certify this fold.
23. *(Rev 1.18.5.)* The U2 code review (`code-review/review-1p14-u2-round1.md`, verdict FIX) found four places where
    an approved rule was unstated for the path the product takes. Rev 1.18.5 states them (§15.23). No promise, PC or
    design changes. The builder fixes findings 1, 3, 5, 6 and 7 in code, and the test author writes the new legs of
    3.10, 5.12 and 5.14 on the host's real teardown path (finding 2). The planner does not certify this fold.
24. *(Rev 1.18.6.)* The box-fixes code review (`code-review/review-box-fixes-2026-10-21.md`, verdict SHIP) noted that
    8.14's text says C is "filled exactly" with "free charge 0", which builds cannot reach. 8.14 now states the
    reachable construction, and U6 carries the GPU host's held-release defect (§15.24). Errata only: no promise, PC,
    reading or design changes.

---

## 14. Not verified at source

Each is re-anchored by the named step before any code depends on it.

*Rev 1.5:* E0 resolved five items at `5d5ca58`, and they leave this list. The schema-section writer is located
(S28, §3.8). The GPU model maps schemas through the same parser (S25) into the same table, whose entries carry the
name hash (§4 item 4). PKV's built CPU code is read (S1–S8, S13,
S26, S27). *Rev 1.10:* T-3060 F16 is settled by the `5d5ca58` adopter path (S20): copied `m`/`e` reach the
first norm call as an unused argument and cannot affect tokens. The next embed replaces the scale. The
attacker-chosen partition in §3.4 keeps its `m`/`e` domains and its token warning for other editable fields;
2.3b's `m`/`e` equality premise is source-grounded. No SuperSLM probe was needed because every route from adopt
to the first token and next embed is explicit in the anchored code.

- *(Rev 1.6.)* The GPU pool's fill-probe form and its floor (F0; 1.8, 2.7, 5.3, and the GPU halves of 5.10 and
  5.11). §8's rule assumes the CPU floors until F0 records the GPU's.
- *(Rev 1.6; rev 1.7 narrows it.)* The plugin's lane loop is read (`SuperSLMSubsystem.cpp:913-973`, this
  worktree's tree): a park gate fits at the worker's dequeue, between jobs, and rev 1.6's park "between Layer-1
  calls" does not, because one job can make several calls (§8's lane-park seam, D-SLM7996). Not read: whether the
  tree U0 re-pins (§9 U0) has the same loop. U1 confirms it before the seam is written. Rev 1.6's fallback (readings
  at tick boundaries with the lanes drained) is withdrawn, because 8.10 (c)'s own operand cannot carry the
  in-flight half (T-3027 F16).
- PKV's GPU code, which is not built at `5d5ca58` (S13): the GPU prefix handle type, its host-mirror gather,
  `UploadPoolRegionsSync`, the guard's derived table, and the GPU pool's draw and share primitives (F0; §4 item 2).
- How the plugin sizes its pool at `Configure()`, so the cache's extra blocks (PC-11) are added in the right
  place (U0; §5.4). *Rev 1.13:* paging specifies it (`S + P + H + A + C` in budget mode; the unchanged
  `sslm_kv_pool_create` in legacy mode; paging §3.2), and §5.11 (b) places C. What remains is U0 reading the
  paging release as built.
- *(Rev 1.1.)* That PKV's fill probe is reachable from the plugin's test module through the vendored engine (U0).
  *Rev 1.13:* paging ships `FSuperSLMPagedTestSeam::RunOnPoolThread` with the probe as test code (paging §3.9), so
  U0 confirms that seam rather than building one.
  The fallback, if it is not, is to register the probe as a plugin test seam over the vendored sources; 1.4's
  operand does not change.
- *(Rev 1.1.)* How the GPU context exposes its adapter's `VendorId` and `DeviceId` (F0). The D3D12 harness reads
  them from `DXGI_ADAPTER_DESC1` (S24); the engine's production accessor was not read.
- The restore cost on the box, with and without SHA-NI (E4). The cloud measured Python's SHA-256 at about
  400 MB/s without SHA-NI (this planning session); that is not the engine's implementation.
- Whether chunked prefill of the same tokens is byte-identical across chunk boundaries: not relied on (§5.2 keys by
  parent), so not verified.
- *(Rev 1.13.)* Whether the plugin's `'SLUW'` reader compares the Layer-1 tag alone or the tag and commit (S14 says
  it records both; paging S16 names the tag). §5.11 (c)'s entry is written to match either; U2 reads it before
  writing the entry.
- *(Rev 1.13.)* The paging release as built: §5.11 rests on paging rev 1.7's text (regions, the region-C seam,
  `CacheRegionPages()`, the one adopt entry point). U0 re-reads each against the released plugin. *Rev 1.14
  (review N5):* paging rev 1.7 is itself not yet approved: its status is "for the principal's approval of the delta
  since rev 1.3". If the principal changes that delta, §5.11 re-folds against the approved text before U1 authors
  8.14.
- *(Rev 1.14.)* The 0.5B cap the plugin ships. This plan reads the certified real 0.5B artifact at cap 32,768 (E0's
  receipt; `e4_real.cpp:59`, whose block-size check at `:60` asserts `32,768 · bytes_per_token`), so its rule B is
  128 (S30); paging works at "0.5B, cap 4,096" (B = 64), while its own 1.5B figures use that artifact's cap 32,768.
  On the CPU the plugin's cap is the model's (paging has no CPU `ContextCap`), so the CPU page size follows the
  artifact. For the certified artifact this plan's reading is the right one; paging's holds only for a 0.5B artifact
  converted at cap 4,096. U0 reads the artifact the plugin ships and its cap; §5.11 (b)'s `P_blk` and
  every 0.5B page figure follow it. The cross-plan note tells the paging planner. *Rev 1.15: settled.* The plugin
  ships the cap-4,096 0.5B artifact (`qwen2.5-0.5b-instruct-cap4096-aex-prompt-result.sslm`, named by the vendored
  `tests/t2130-g5-red-suite/t2922_real_model_manifest.json`), so paging's 0.5B CPU figures (`B = 64`, 384 KiB pages)
  are the plugin's and §5.11 (b)'s `P_blk` follows them. This plan's cap-32,768 figures (§3.2's first resident-page
  column, 10.x) belong to E4's certification artifact only. U0 still confirms the manifest in the tree it re-pins.
  The rev 1.14 cross-plan note's §2 is corrected to match.
- *(Rev 1.18; settled in rev 1.18.1, plan review F7, audit G15.)* The plugin's worker lane count is
  `Clamp(MaxConcurrentCalls, 1, BlockCount)`, so 8.10 (c)'s K + 1 lanes need `MaxConcurrentCalls ≥ K + 1` and
  `BlockCount ≥ K + 1` (8.10). U1 confirms it in the tree U0 re-pins.
- *(Rev 1.18; settled in rev 1.18.1, plan review F14.)* `SuperSLM::TokenizeText` and `USuperSLMSubsystem::Tokenize`
  make the same `sslm_tokenize` call with the same sizing and differ only in the mapping (§5.8). 7.20 executes it
  with the subsystem configured on the set's model.
- *(Rev 1.18.)* What the separate fast-decode release (1.15.0) changes in the sequence formats or the forward path.
  It is another plan's; F4 and U6 re-run 9.9 and 9.10 against its saves only if a plugin release vendors it.
- *(Rev 1.18.)* The squash's contents (M1): which files `main`'s 1.13.1 tree and the line differ in once the line is
  merged with 1.13.1. The four `v1.13.0` → `v1.13.1` files are all GPU (§3.8); M1 records the merge and its diff.

---

## 15. Fold map

Every finding from every round, and where it landed: 15.1–15.4 are the gate rounds (revs 1.1–1.4); 15.5 is E0's
re-anchor (rev 1.5); 15.6 is the re-gate of rev 1.5 (rev 1.6); 15.7–15.12 are the scoped re-check folds; 15.13 is
the E2 re-anchor on 1.13.0 and the paging plan's cross-plan notes (rev 1.13); 15.14 is the scoped re-gate of rev 1.13
(rev 1.14); 15.15 is the re-check of rev 1.14 (rev 1.15); 15.16 is the
strike on rev 1.15 (rev 1.16); 15.17 is the re-strike on rev 1.16 (rev 1.17); 15.18 is the principal's decisions of
2026-10-17 and 2026-10-20 (rev 1.18); 15.19 is the scoped re-check of rev 1.18 (rev 1.18.1); 15.20 is the strike on
rev 1.18.1 (rev 1.18.2); 15.21 is the adversary's re-check of rev 1.18.2 (rev 1.18.3); 15.22 is the erratum from the E4, E5 and U2 suite
audit (rev 1.18.4); 15.23 is the clarifications from the U2 code review (rev 1.18.5); 15.24 is the box-fixes review's
erratum (rev 1.18.6); 15.25 is the principal's plugin 1.3.0 ruling (rev 1.18.7).

### 15.1 Round 1 (rev 1.1)

*Rev 1.1.* Sources: the plan review of rev 1 (verdict **temper**: 3 structural, 5 weak, 6 notes) and the coverage
audit of rev 1 (verdict **gaps named**: B1–B5 blocking, G1–G11, N1–N8). Every item landed; none was declined.

**Plan review.**

| # | Finding | Landed |
|---|---|---|
| 1 | Strike promise false; "bound every field" false | §3.4 step 8 tightened (`current_token` lower bound; `m` within int32; `L` bounded before the size; checked arithmetic); §3.4 *What is promised*: (i) fidelity for this library's blobs, (ii) integrity against accidental change, authenticity disclaimed, every field accepted unchecked named, (iii) memory safety as a separate claim; `SSB5`'s unchecked `m`/`e` noted, not changed (S20); §0; §13 item 2; cells 2.3, 2.3b, 7.11 |
| 2 | GPU 1,000× bar underived | 10.4: a derived 100× bar from cited prefill figures (S22) with the rates marked assumed and the margin marked underived; 10.3's CPU bar derived too; R1b names two folds in order (the digest off-lane, then an absolute latency bar); §8 commissions the GPU instrument; F3 |
| 3 | U6 ledger loses an evicted parent's pages; 1.4's operand unnamed | §5.4 ghost charges; 1.4's operand named (the fill probe, minus the sequences' reservation, no adopter live); new 1.9 [U6] with its mutant; 7.10 |
| 4 | Recompute should use budgeted begin; `PrefixBlockCount` counting unstated | PC-11; §5.3 item 4; §5.4 (the build-concurrency rule removed; caches not counted; extra blocks at `Configure()`); rev 1's R7 (the legacy-reserve risk) removed, and R7 now names another risk; §14; cell 8.10 |
| 5 | No schema bind on recompute | §5.3 item 4 (`sslm_schema_lookup`, then `sslm_prefix_set_schema` before the first prefill); §5.6, §5.9 `UnknownSchema`; cell 2.11 |
| 6 | Cross-vendor GPU blobs; PC-5's premise versus `docs/platform-support.md` | Checked at source: the claim is backed at position 0 only, for one token, in a run-only package (S21). PC-5 re-decided: backend and GPU vendor/device refusal kept, the producer field added to the blob (§3.2) and the key (§5.2); §4 item 5 says what the promise means on an uncertified GPU; cell 4.6 (byte identity, three ways); U7 cooks per vendor; R7; §6 |
| 7 | Ownership of the cache's mutable state | §5.5 *Ownership*: the game thread owns it, workers get copies, results come back through a completion queue, eviction is decided on the game thread; cell 3.9 |
| 8 | GPU precedence is a set, not an order | §4 item 3 made a total order; cell 5.7 [F2] |
| 9 | Magic is refused by step 2, not the digest | 2.2 reworded |
| 10 | 2.1's `L` unnamed | 2.1: exhaustive at `L = 1`, strided at `L = 1,000` |
| 11 | Disk trim's access time | §5.7: the cache sets modification times itself; cell 7.14 |
| 12 | PC-7 separates releases only | PC-7's cost column |
| 13 | U0 underestimated; box contention | U0 90–120 min with its reasons; §11; R8 |
| 14 | Step 1 wording | §3.4 step 1 reworded |

**Coverage audit.**

| # | Finding | Landed |
|---|---|---|
| B1 | "draws nothing" cannot fail; precedence unmodeled | §3.4 draw paragraph; cells 5.6 (zero free pages and allocation seam; draw-then-return mutant) and 5.7 (pairwise precedence, CPU and GPU, `SSLM_BUSY` first); 7.2 remapped |
| B2 | Hostile values accepted; three claims overstated | As plan review #1. `m` is bounded, not left to the funnel. `e`, the counts, residual and rows are named as accepted unchecked. The "memory-safe" class is 2.3b under ASan and UBSan |
| B3 | Cache handles against the existing verbs | New §5.10; `PrefixLeaseEnded`; `ReturnPrefix` returns bool; the release-counting seam; cell 2.10 |
| B4 | Memory tier key unproven; no hit or LRU oracle | The memory tier compares the full spec on a hit (§5.2, §5.5); cells 2.12 (key-component mutants), 7.13 (memory-hit and LRU oracles), 11.4 extended |
| B5 | GPU blobs cross vendors | As plan review #6; cell 4.6 is in F1's red suite |
| G1 | Counts mutant routed to a blind cell; operand; fixtures | 7.1 re-routed to 6.1 and 6.4; 6.1's operand named; §7 preamble fixtures |
| G2 | Format never checked independently | 7.12 (a python `struct` parser, `hashlib` digest; B=4/B=16 comparison is refined by §15.12 to exclude independently checked identity/digest fields) |
| G3 | Link-red suites cannot fail for their reason | §7 preamble: stubbed suites; E1, F1, U1 |
| G4 | Cache × 1.0.0's caps | PC-11; 8.10 |
| G5 | Coalescing and reconfigure off the happy path | 3.5 extended; 1.7; §5.4 teardown drains the lanes |
| G6 | Disk trim and PIE sharing | 7.14; 3.8; §5.7's unique temporary names |
| G7 | No GPU lifetime cells | 1.8 |
| G8 | Restored handle × the prefix verbs; hint parents' modes | 8.11; 2.6 extended |
| G9 | Corruption and version cells too coarse | 2.2 exhaustive at `L = 1`; 2.3's major, minor and patch cases; 7.6's second mutant |
| G10 | Prefill counter and off-thread assert do not exist | §8: `sslm_test_only_prefill_call_count` (moved and renamed in rev 1.2) and the off-game-thread assert, each with a must-reject; 7.8, 10.1, 10.5 |
| G11 | Unknown `SchemaName` | As plan review #5; 2.11 |
| N1 | Cite the SHA FIPS tests | S24; §8 |
| N2 | Who authors the version check's constructions | 11.3 (the debunker; an unparsable-line must-reject); §8 |
| N3 | 1.3 on restored pages; dirty mutant | 1.3 |
| N4 | 1.4 on the fill probe; both entry kinds | 1.4 |
| N5 | Fault between write and rename | 5.8; §8 |
| N6 | Hitch claim unmapped | 5.9; §8 |
| N7 | Canary after the save buffer | 2.4 |
| N8 | Size-formula wrap | §3.4 step 8 (bound `L` first, checked arithmetic); 2.3's wrap case |

### 15.2 Round 2 (rev 1.2)

Sources: the plan review of rev 1.1 (**temper**: B1–B2 blocking, N1–N8) and the coverage audit of rev 1.1 (**gaps
named**: R2-B1 to R2-B4 blocking, N-R2-1 to N-R2-14). Every item landed; none was declined. The strike on rev 1.1
folds separately.

**Plan review.**

| # | Finding | Landed |
|---|---|---|
| B1 | The unchecked-field list missed the schema binding, `L` and `origin` | §3.4 restated as a partition of §3.2's table: fields fixed by steps 2–6, and every other field attacker-chosen within steps 7–8's limits, each named with its limit; §0, §3.8, §13 item 2 |
| B2 | "A full cache leaves 1.0.0's blocks" false under pinned leases and in-flight builds | §5.4 *charge rule*: every build or restore is charged at its peak draw (`R(L)` or `ceil(L/B) − s`) before it draws; if it cannot be charged, `BudgetExhausted`. PC-11's buy restated exactly; §5.9, §5.6; 8.10 made dynamic (a)–(c); 1.4 and 7.10 aligned (no pinned exception); 5.5 |
| N1 | 0.5B GPU prefill mis-inferred | S22 corrected; 10.4 re-derived (0.5B ~919× / ~1,468×; binding case 1.5B on the 7900 XTX, ~846×); R1b's trigger restated; the 100× bar kept |
| N2 | Ghost charge keyed to children; tail overcount | §5.4 "last cached descendant", overcount ≤ 1 page stated; 1.9 three levels |
| N3 | 1.4 against 7.10 | Resolved by the charge rule; both assert no exception |
| N4 | Per device, not per vendor; the "45,845-token" claim | §4 item 5, PC-5's cost, U7 say device; S21 notes the figure is a token id |
| N5 | PKV's legacy-admission scope | §5.4 *Legacy admission*; 8.6's premise fixed |
| N6 | Checked arithmetic unreachable; lower bounds untested | §3.4 step 8 marks it defence in depth; 11.1 exempts it; 2.3 adds `L = −1`, `origin = −1` |
| N7 | Wrong citation; id counter; `AdoptPrefix` lookup | §5.10 cites `SuperSLMSubsystem.cpp:84-87`; §5.6 one handle per lease from `GNextPrefixId`; `AdoptPrefix`'s cache lookup in §5.6 and U2 |
| N8 | §0 said "by its digest" | §0 says size, magic or digest |

**Coverage audit.**

| # | Finding | Landed |
|---|---|---|
| R2-B1 | Precedence unpinned across equal-status runs | 5.7's pairing rule and full CPU and GPU pair lists; 8.7 names both statuses; mutant: steps 4 and 5 swapped |
| R2-B2 | The general allocation slot fires in SHA-256, before the draw | §8 restore-site slot with its must-reject (fires after the draw, records the free count); 5.2 and 5.6 use it; 5.2's two return mutants; §3.4 |
| R2-B3 | In-flight builds outside the budget | As plan review B2; 8.10 (c) with K ≥ 8 concurrent misses, `CreatePrefix` to its cap and a slot reset, plus the uncharged-build mutant |
| R2-B4 | 7.14 serves no hit | 7.14 writes A, B, C, hits A, then trims; the no-refresh mutant |
| N-R2-1 | 2.3b needs the sanitizer build | E2's gate adds the ASan+UBSan build |
| N-R2-2 | §5.10 citation | Fixed (with N7) |
| N-R2-3 | Lease identity | §5.6 one handle per lease; 2.10's two-lease case |
| N-R2-4 | 2.12 parent key at U2; length-prefix example | Parent-key case moved to U6 (and U6's cell list); the pair is derived from the real field order, or the mutant is recorded equivalent |
| N-R2-5 | LRU against FIFO | 7.13 touch order and FIFO mutant |
| N-R2-6 | 4.6's `L` and artifact | Real 0.5B and 1.5B at `L = 1,000`, every layer, head and position |
| N-R2-7 | 7.12's operand list | Every field's operand named |
| N-R2-8 | Absence-only cells pass on stubs | §7 preamble positive clauses (3.7, 3.9, 7.8, 8.9); 3.9's assert in the single accessor, registered in §8 |
| N-R2-9 | 1.7's operand | Pool destroy status, release seam, fill probe |
| N-R2-10 | 1.8's landing | Exactly `ceil(L/B)` free pages, confirmed by page index; rows differ on every page |
| N-R2-11 | Prefill counter false-fails | Moved to `sslm_prefix_prefill` and renamed |
| N-R2-12 | Plugin portable SHA arm | §8 names it not covered, with the cost of a miss |
| N-R2-13 | 1.4 pins and in-flight | 1.4 includes in-flight pages; no pinned exception (charge rule) |
| N-R2-14 | Seams without must-rejects | §8 must-rejects for the write-rename fault and the lane delay |

### 15.3 Round 3 (rev 1.3): the adversary strike on rev 1.1

Verdict **FRACTURED** on promise (i). Schema identity in `SPX1` is `Fnv1a64(name)`; the parser refuses duplicate
names but not duplicate hashes; `ByNameHash` returns the first match. So an untouched, library-written blob bound
to the second of two colliding names restores `SSLM_OK` bound to the first, on both backends and through the
plugin's cooked and disk tiers. The shipped `SSB5` path has the same hole.

| Finding | Landed |
|---|---|
| The fracture: colliding schema-name hashes | PC-12 (refused at model map on both backends, with its registered must-reject the two-name artifact; the converter refuses to emit one; zero-hash names refused too, a hole of the same class the strike did not name); S25; §3.1 (the hash is lossy, and unique only under PC-12); §3.4 step 7 and promise (i) scoped to PC-12; §3.8; §4 item 4; §5.2's key claim scoped; X1–X4 note, new X11; cells 2.13 (must-reject), 2.14 (the `SSB5` path), 2.15 (plugin `Configure()`, editor build, cooked and disk tiers), 6.7 (per-schema adopter dispositions on a many-schema model), 7.1 re-scoped, 7.15, 11.6; §8 the certified-artifact audit; E0, E2; R9; §13 promise |
| Blast radius: the shipped `SSB5` path | Closed by PC-12 with no format change (2.14). The plan changes no `SSB5` code; the fix is in the shared parser |
| Blast radius: cells 6.1–6.5, 7.1, 8.4, 9.1, 10.1–10.2 never built two distinct equal hashes | They stand as proofs within PC-12. 2.13 pins that the colliding case cannot occur, and 6.7 pins non-first-schema fidelity |
| Lesser plane: the engine version does not separate development builds | Promise (i) is scoped to the same release version, or the same build between releases (§3.4, §0, §13); PC-7's cost column says what the plan does and names the reversal (the build's commit in the producer); the plugin already separates development builds by the Layer-1 commit |
| Lesser plane: the GPU cache and nested entries depend on the plugin paging plan | Already PC-6 and R4; no change |

### 15.4 Round 4 (rev 1.4): the re-strike on rev 1.3

Verdict: promise (i) (fidelity, within PC-12 and the version scope) and (ii) (integrity) **held**. **FRACTURED:** §3.4's
claim "a restore is never refused where the build it replaces would have been admitted" does not hold for nested
prefixes restored privately. At the constructed cell (parent 1,000 resident, child `begin_from(parent, 200)` to
1,200, 14 free pages) the build draws `R(200) = 14` and is admitted. Every private restore (null, unfrozen,
another-model, another-pool, wrong-length, or unequal-rows hint; `s = 0`) needs `ceil(1200/16) = 75` and is
refused. The error was "`begin_from` shares the same `s`", which holds only for a matching hint.

**Ruling** (delegated to the conductor; recorded, not a PC for the principal): the parent hint stays optional. A
private nested restore needs `ceil(L/B)` free pages, and build-equivalent admission is promised only with a matching
parent.

| Finding | Landed |
|---|---|
| The admission claim is false for private nested restores | §3.4 *Peak draw* split into three cases (root; nested with a matching parent; nested private, with the strike's figures); §3.5's "never a requirement" narrowed to correctness; §3.8 header text |
| No cell composes private restore with a pool sized for the build | New 5.10 [E3, F2]: the strike's construction, six private hints refused `SSLM_KV_POOL_EXHAUSTED` with nothing drawn, and the matching-hint control admitted |
| 7.4 claims the universal statement | 7.4 narrowed to root and matching-parent restores; the private case maps to 5.10; 7.5 reworded |
| The plugin cache's nested path | §5.4: a nested restore is charged `s = 0` (`ceil(L/B)`) before the draw, with the shared pages refunded when the result is applied, so a failed hint cannot overdraw the ledger. A charge that cannot fit resolves by name per §5.9, with no retry as a build. New 8.12 [U6] with its mutant |
| X4 | Unchanged: it states disposition and bytes, not admission; X8 covers capacity |

### 15.5 Round 5 (rev 1.5): E0's re-anchor on paged KV at `5d5ca58`

Source: E0's cloud half, 2026-10-01 (`e0-reanchor-pkv-5d5ca58.md`, beside this file), run at paged KV's build-ahead
tip `5d5ca58` on the principal's build-ahead call, in place of the 1.12.0 tag. Per §9 E0 and R2, a finding that
changes a §3 rule is a fold before E1. The planner re-read every finding at `5d5ca58` before folding it. Every
finding held at source and was folded; none was declined. The note's evidence was corrected in three places
(marked *corrected*). No PC changes. §3.4 *What is promised* and §13 item 2's quoted promise are untouched. The
promise's citation `checked_chain_funnel.cpp:197` reads `:199` at `5d5ca58` and was left as is, because it sits
inside the promise text. Record: `Claude/Vitruvius/t3021-precache-e0-fold-2026-10-01.md`.

**Rule changes.**

| # | Finding | Verified at source | Landed |
|---|---|---|---|
| R1 | Draw and share are two critical sections, so a refusal can follow the draw; rev 1.4 said one section, with allocation failure the only post-draw refusal | Holds (S26) | §3.4 *Draw and build* as four ordered steps (draw, share, map, allocate), with the share refusal's statuses and "two refusals follow a draw"; §3.7's two bullets; 11.1 exempts the share-refusal release as defence in depth (no ABI seam reaches it; no PKV cell drives that branch either, S26); §4 item 2 takes the order, and F0 re-anchors the GPU primitives (§14). `SSLM_KV_POOL_EXHAUSTED` still draws nothing, and admission is unchanged, because the pages drawn stay `ceil(L/B) − s` |
| R2 | §3.8 named the two G35 ordinal cells as `sslm_status` mirrors; they pin `SslmGpuStatus` | Holds (S27). A tree-wide grep for the status tail finds no other hand-written mirror. Each of the three has a compile-time sentinel check that needs no edit | §3.8 *Status*: ordinal 30, the X-macro, the three hand-written mirrors and `docs/api.md`; the G35 cells move to F2 (§4 item 3, step 6) |
| R3 | No converter writes the schema section; PC-12's converter refusal goes into `_serialize_scm1` | Holds (S28). *Corrected:* five in-repository importers, not four (`tests/t2899-schema-deadend-red-suite/make_t2909_string_schema_fixture.py:69` added). A second in-repository `_serialize_scm1` exists, a frozen and hash-pinned test input, which must not be edited | §3.8's new *converter refusal* bullet (live serializer refuses; frozen copy and the out-of-repository copy unchanged, both caught at map); cell 2.13's converter must-reject; E2's row. PC-12's own text ("the converter refuses to emit one") stands: E0 was charged with locating the writer (§14, rev 1.3), and this is where it is |

**Text corrections.**

| # | Finding | Verified at source | Landed |
|---|---|---|---|
| T1 | §3.4 step 8 and §14: an adopter's first step does not pass the prefix's scale through the funnel | Holds (S20). Also checked: the embed routes through the same checked funnel, so every stored scale passed the int32 check | §3.4 step 8 rationale reworded (out of the funnel's domain; never produced by save; an edited `m` or `e` changes no token); the §14 item removed; S20 extended |
| T2 | §3.1 said `SPX1` carries no mode "as `SSB6` is"; `SSB6` carries `kv_mode` and `budget` | Holds (S7; `sslm_abi.cpp:3638-3646`) | §3.1: `SPX1` carries neither, with the four reasons neither is observable on a frozen prefix |
| T3 | §3.7 said restore reads the hint's rows under the pool mutex; the built comparison reads them unlocked | Holds (`SharingHandleMatches`, `sslm_abi.cpp:3685-3707`) | §3.7's first bullet |
| T4 | E0's "confirmed early for F0": the GPU model maps schemas through the same parser | Holds (`gpu_1p0.cpp:1247`); the planner also read that the GPU model holds the same `SchemaMasksTable`, whose entries carry `name_hash` (`gpu_1p0.cpp:351`) | S25; §3.8 *Model map*; §4 item 4 (the name-hash question closed, its fallback dropped); §14; F0's row |

**Builder notes.** Kept as notes for E2 and E3 and not folded into plan text: reuse `SharingHandleMatches` for
(b)–(f); a holder-taking `GatherCanonicalRows` for the save, with `ScatterRows` reused; `SPX1`'s `current_token`
written verbatim (not `sslm_seq_save`'s pending-embed rule); `kv_budget` set to any in-domain value on a restored
prefix; map with `HolderMapNext`; the restore-site slot is still needed, because `Sha256` consults the general
slot. **One note is reclassified as a plan correction:** the fill probe resolves only `k ≥ 2` free pages
(`pkv_common.h:585-602`, `:651-688`). Rev 1.4's §8, 5.6, 5.7 and 7.4 asked it to read 0 free. Landed in §8's two
fill-probe sentences, 5.6 (built by admission, blobs of 2 or more pages), 5.7's CPU domain pair, and 7.4 (a refused
2-page create replaces the "0 free" reading).

**Anchor and ground truth.** The header, §0's step list, §2 (S1–S14, S20–S22 and S25 re-read, new rows S26–S28),
§3's and §8's engine line citations, §8's reference anchor (pin at `5d5ca58`, re-pin at the tag if PKV code
changes before it), §9's E0 row, and §14 are at `5d5ca58`. *Corrected:* the note cites `ready_for_logits` at
`:2683`, which is a comment line; the assignment is `:2673` (S3). The E0 note's "flag for U0" (the project's vendored
copy is at v1.11.0, not v1.9.0) is in U0's row and S14. U0 names the tree it re-pins. Rev 1.4's header named copies
`plan-rev1.md` to `plan-rev1.3.md` beside this file, which do not exist. Earlier revisions are in git history.

### 15.6 Round 6 (rev 1.6): the re-gate of rev 1.5

Sources: the plan review T-3023 (`Claude/Charpy/t3023-precache-rev1p5-review-2026-10-01.md`, verdict **temper**:
S-1, W-1 to W-3, N-1 to N-8) and the coverage audit T-3024 (`Claude/Mendeleev/t3024-precache-rev1p5-audit-2026-10-01.md`,
verdict **gaps named**: F1 to F9, N1 to N5), both of rev 1.5 at `4b469b6e7f`. The planner re-read every finding at
SuperSLM `5d5ca58` before folding it (read-only, `git show`). Every finding held at source and landed, with three
exceptions: N-7 and N-8 ask for no change, and F6 and F7 are routed to the test author at E1 (§9 E1) as the audit
directs. No PC changes; §3.4 *What is promised* and §13 item 2 are untouched. Decisions D-SLM7984 to D-SLM7989.
Record: `Claude/Vitruvius/t3025-precache-rev1p6-fold-2026-10-01.md`.

**Plan review T-3023.**

| # | Finding | Verified at source | Landed |
|---|---|---|---|
| S-1 | The refusal in `_serialize_scm1`, the only live `SCM1` writer, leaves no sanctioned writer for PC-12's must-reject artifacts | Holds: five importers (S28); the decode-threading fixtures are generated in every CI job through it (`tests.yml:32-33`); the frozen copy is hash-pinned. T-3022's brief leaves the must-reject to an independent seat | D-SLM7985: an independent test-only `SCM1` writer, commissioned by byte equality with `_serialize_scm1`; no opt-out on `_serialize_scm1`, so PC-12's text stands unchanged (the reviewer's option (c)). §3.8, 2.13, §8, E1, E2 |
| W-1 | 5.6's "every refusal case of 2.1–2.5" against its ≥ 2-page blob rule; exactly 0 free is buildable by admission | Holds: `HolderCreate` draws exactly the reserve and refuses before drawing; `R((r−1)·B) = r` (executed); a fresh pool is `block_count · cap_pages` pages (S29) | D-SLM7986: 5.6's pool has exactly 0 free by admission of `N`; the `L = 1` cases stay and now discriminate; the 2-page blob rule withdrawn. 5.6, 5.7, §8 |
| W-2 | The `k ≥ 2` limit not applied to 5.1, 7.4's nested twin, 1.8 | Holds: `ProbeSplit` `:592-593`; the W-2 example gives 1 (executed) | 5.1 shapes (4 and 12 free), 7.4 nested twin (13 at `B = 16`, 50 at `B = 4`), 1.8 (`ceil(L/B) ≥ 2`, F0's GPU form); §8's reading rule (D-SLM7987) |
| W-3 | Moving anchor: E2–E4's base unnamed; a rule-changing re-diff at E5 triggers nothing | Holds: §8 and R2 as written in rev 1.5. The paged-KV branch tip is still `5d5ca58` (local ref, 2026-10-01) | D-SLM7988: E2–E4 build on `5d5ca58`; the trigger is this branch taking newer paged-KV code at any step; fold plus scoped re-gate before the next build step; the harness refuses references whose anchor is not the build's merge-base. §8, R2, E0, E2, E5 |
| N-1 | §3.7's "twice" is "once, or twice when `s > 0`" | Holds (`kv_pages.cpp:148-181`, `:198-217`) | §3.7 |
| N-2 | t2138's mirror is positional | Holds (t2138 `sslm_abi.h:168-177`, no `=`) | §3.8 *Status* |
| N-3 | "Fails the build" is "fails ctest's Gate C entries" | Holds (`CMakeLists.txt` targets `EXCLUDE_FROM_ALL`; entries `:794-858`) | §3.8 *Status*; 7.16; E2's exit |
| N-4 | 11.1's share-refusal exemption: seam or exemption; §3.6's "refused" does not describe the refusal | Holds | As T-3024 F4 and N2 below |
| N-5 | §4 preamble's `f4a1b6a` | Holds | §4 preamble |
| N-6 | §0 *Releases* reads as gated on the tag | Holds | §0 *Releases* gains the "after 1.12.0's code" clause; PC-1 untouched |
| N-7 | PC-12's authoring-time refusal lands in a test-fixture tool | No change asked. D-SLM7985 keeps the refusal total in that tool | — |
| N-8 | The promise's `checked_chain_funnel.cpp:197` reads `:199` | No change asked; the promise is out of scope | — (owner: the principal, if the promise text reopens) |

**Coverage audit T-3024.**

| # | Finding | Verified at source | Landed |
|---|---|---|---|
| F1 | The probe floor class open: 5.1, 7.4 nested, 1.4, 1.9, 8.10, 8.12 | Holds (S29; the one-state form is "for `k >= 3`", `pkv_common.h:690-695`); 8.10 (a)'s slack is the last block's rounding, from 0 | D-SLM7987: §8's reading rule, with three forms, their floors, and every reading cell named; 1.4 (admission lower bound at `F_min ≥ 2`), 1.9, 8.10 (one spare block), 8.12 via 1.4; plus W-2's cells |
| F2 | 5.6's population includes one-page blobs; draw-first survives | Holds | As W-1 (D-SLM7986). The population is the restore refusals of 2.1, 2.2, 2.3 and 2.5; the mutants are stated; `L = −1` recorded as non-discriminating |
| F3 | No construction path for PC-12's must-reject after E2 | Holds (as S-1) | As S-1 (D-SLM7985). The audit's option (b) |
| F4 | The share-refusal release exempted where a seam is cheap | Holds: PKV pins its own unreachable short-reserve branch through `sslm_pkv_test_only_drain_reserve` (`sslm_abi.cpp:4626-4632`); this plan's 11.2 precedent | D-SLM7984, the ruling: **seam, not exemption.** The restore-site fault slot gains share and map positions; cell 5.11; 11.1's exemption withdrawn; §3.4 step 2 |
| F5 | The restore-site slot's must-reject passes a slot placed between draw and share | Holds | §8: the slot records the shared-entry count; must-reject (2); 5.2's nested case at `s ≥ 1` [E3] |
| F6 | "An edited `m` or `e` changes no token" maps to no cell | Holds (S20) | **Routed to the test author at E1** (§9 E1; pointer at 2.3b). No cell authored here |
| F7 | §3.1's "no mode or budget" half unmapped | Holds (S7, §3.1) | **Routed to the test author at E1** (§9 E1; pointer at 7.12). No cell authored here |
| F8 | "Two refusals follow a draw" omits `MapPagesFor`'s branches | Holds (`sslm_abi.cpp:1641-1653`; PKV's restore releases on either, `:4032-4037`) | §3.4 step 3: disposition named; short reserve pinned (5.11, map position); past the table exempt under 11.1 (unreachable: `B` divides the cap, S26); "three refusals can follow a draw"; X12 |
| F9 | The GPU share-refusal position has no receiving cell; R2 routes only §3 | Holds | §4 item 2; 5.11's GPU half pending F0; R2 reads "§3 or §4" |
| N1 | Ordinal-30 pins not named; E1's constant; `docs/api.md` unchecked | Holds (`CMakeLists.txt:794-858`; `pkv_abi.h:37-39`; `docs/api.md:553`, `:787`) | 7.16 names the Gate C entries; E2's exit; §3.8's test-side constant; D-SLM7989: 7.17, the `docs/api.md` ordinal check at E5, registered in §8 |
| N2 | §3.6's "no third disposition" contradicted | Holds | §3.6 gains *refused after a draw*; X12 |
| N3 | References consumed at E2–E4, re-pinned "before E5" | Holds | As W-3 (D-SLM7988), including the audit's structural form (the anchor recorded, the harness refuses) |
| N4 | "Save never writes an `m` outside int32" cites no cell | Holds: the funnel's refusal is pinned at `tests/test_main.cpp:9065`, `:9125`, `:13848` | S20; §3.4 step 8 |
| N5 | The one-state probe's admissions compete with in-flight draws on the live plugin pool | Holds as reasoning over §5.4's evict-and-retry rule. Plugin lane code not read | §8: plugin readings with the lanes parked (the lane-park seam, with its must-reject); §14 names the seam's feasibility unverified and its fallback |

**Found while verifying, outside this plan.** PKV's `ProbeSplit` splits an odd `k` with a 1-page part when
`cap_pages = 2` (executed: `k = 3` gives `[1, 2]`), and the 1-page create, at budget 0, is refused
`SSLM_INVALID_ARGUMENT`. The probe fails loudly there; it does not pass silently. No geometry this plan uses has
`cap_pages < 3` (S29). It is reported to the conductor for PKV's owner and is not folded here.

### 15.7 Round 7 (rev 1.7): the re-check of rev 1.6

Sources: the plan review T-3026 (`Claude/Charpy/t3026-precache-rev1p6-recheck-2026-10-01.md`, verdict **temper**:
S-1, W-1 to W-6, N-1 to N-7) and the coverage audit T-3027 (`Claude/Mendeleev/t3027-precache-rev1p6-reaudit-2026-10-01.md`,
verdict **gaps named**: F10 to F18, N6 to N9), both of rev 1.6 at `f482adcfe8`. S-1 and F11 were ruled by the
conductor before this fold (D-SLM7990). The planner re-read every finding at source before folding it: SuperSLM
`5d5ca58` read-only through `git show`; the plugin read-only in this worktree. Every finding held and landed. Four
pairs name one defect from two sides (S-1/F11, W-1/F15, W-2/F12, W-5/F10), and W-4/F18 prescribed different fixes,
ruled in D-SLM7994. No PC changes; §3.4 *What is promised* and §13 item 2 are untouched. Decisions D-SLM7991 to
D-SLM7996. Record: `Claude/Vitruvius/t3029-precache-rev1p7-fold-2026-10-01.md`.

**Plan review T-3026.**

| # | Finding | Verified at source | Landed |
|---|---|---|---|
| S-1 | The `SCM1` writer's author is the audit's builder, and it lands after the audit's verdict is due | Holds: the T-3022 record names the test author as the audit's builder and the audit `QUARANTINED`; `instrument-commission-schema.md` requires `constructed_by ≠ built_by`; E1 depends on E0 | D-SLM7990 (conductor): the debunker authors the writer and its artifacts and commissions the audit, inside E0, after T-3022. §3.8, §8 (audit and writer), E0 (owner, work, gate), E1 |
| W-1 | 7.17 checks every stated ordinal against `NEXT_FREE − 1`; the doc states `SSLM_KV_BUDGET_EXCEEDED`'s own | Holds: `docs/api.md:553`, `:787` both state 29 for that name | D-SLM7991 supersedes D-SLM7989: per-name check, the new status required at 30, a per-name floor. S27, §3.8, 7.17, §8 |
| W-2 | 5.11's root map case cannot kill the holder-leak mutant | Holds: `HolderReturnReserve` returns the whole reserve (`kv_pages.cpp:379-411`); `live_refs` drops only in `ReleaseHolder` (`sslm_abi.cpp:1622-1627`); the destroy refuses on it (`:1226-1231`) | D-SLM7992: every 5.11 case asserts the pool destroy returns `SSLM_OK`; the full drain stays. 5.11 |
| W-3 | 5.6's share-armed runs never reach a share, and its statuses collide with validation's | Holds: §3.4 step 2 runs only when `s > 0`; 5.6's population is root | D-SLM7993: 5.6 arms only the allocation position and asserts the slot did not fire. 5.6, §8 |
| W-4 | The merge-base refusal sees ancestry, not code; CI is shallow | Holds: `tests.yml:25` and the rest have no `fetch-depth`; one job has `fetch-depth: 0` (`:1106-1108`) | D-SLM7994: ancestry kept, and its limit stated; the line is a configure argument; a missing ref refuses; CI needs full history to grade; "never a false green" marked reasoned; must-accept named. A content key was declined: no path set separates the references' dependencies from this plan's own edits. §8, R2, E2 |
| W-5 | 8.10's admission bound has no defined subtrahend for 1.0.0's prefixes | Holds: legacy begin draws `cap_pages` (`sslm_abi.cpp:2155-2170`); first freeze leaves `ceil(L/B)` (`:2301-2306`) | D-SLM7995: `P₁₀(t)` subtracted, exact by construction and a counter. 8.10, §8 |
| W-6 | The lane-park must-reject does not test quiescence | Holds: plugin `Run()` drains opaque jobs from a FIFO; one job can make several calls (`SuperSLMSubsystem.cpp:913-951`) | D-SLM7996 supersedes D-SLM7987's clause: park at the dequeue, a barrier, yields to stop; must-reject with a held job. §8, 1.4, 1.7, 1.9, 2.10, 5.5, 8.10, §14 |
| N-1 | The E1 constant's `static_assert` has no macro | Holds: PKV's assert is at `pkv_common.h:41-42` under `SSLM_HAS_PAGED_KV_ABI` (`sslm_abi.h:282`) | §3.8 names `SSLM_HAS_PREFIX_SAVE_ABI`; E1, E2 |
| N-2 | 5.6's non-discriminating cases unrecorded; another-model pool unsized | Holds: `L`'s field ends at byte 120 (§3.2) | 5.6 records them; the other model's pool is held at 0 free |
| N-3 | Must-reject (3) at share needs a nested blob | Holds (§3.4 step 2) | §8 must-reject (3) |
| N-4 | Several cells write no pool size | Holds; harmless | §8's initial-count clause (with T-3027 N9) |
| N-5 | The writer's zero-hash artifact is conditional | Holds (2.13) | §3.8, §8 |
| N-6 | 8.10's slack is `[0, cap_pages − 1]`, not "0 or 1" | Holds (§5.4's sizing) | 8.10 |
| N-7 | No row says who writes 7.17 red or authors its constructions | Holds | E5 row: the test author writes it red; the debunker authors and commissions; 7.17 |

**Coverage audit T-3027.**

| # | Finding | Verified at source | Landed |
|---|---|---|---|
| F10 | 8.10's `F_min` omits 1.0.0 prefixes' pages | As W-5; also `R_seq` constant (plugin `:2095-2101`, reset `:3261-3263`) | As W-5 (D-SLM7995), with the audit's commissioning pair |
| F11 | The writer's author and timing | As S-1 | As S-1 (D-SLM7990) |
| F12 | 5.11 root map leaks no page | As W-2 | D-SLM7992 takes the audit's occupant (b). Occupant (a), a one-page drain, was declined: it needs a new `kv_pages` primitive and still passes a reserve-only release, which (b) kills |
| F13 | No must-reject places the allocation position after the map; (3)'s fixture | Holds: `PageHolder::mapped` and `reserve` (`kv_pages.h:93-103`) | §8: mapped and reserve records, must-reject (4), (3)'s fixture, the fired record and disarm rule; 5.2's reserve-only mutant |
| F14 | Lane-park: no barrier, no job boundary, posts counted, teardown hang | Holds: teardown `Kill(true)` waits on the drain (`:1624-1633`) | As W-6 (D-SLM7996) |
| F15 | 7.17 checks the wrong quantity | As W-1 | As W-1 (D-SLM7991), with the audit's three must-rejects; its floor is per name, because a pair count of 2 would pass a reworded statement once the new status adds a third |
| F16 | "The spare block hides no mutant" false for the 1.0.0 operand | Holds: K = 8 overdrawn pages fit inside a spare block of `cap_pages ≥ 8` pages (one comparison, not executed) | 8.10: the claim narrowed to the probe reading; "also caught by its own operand" withdrawn; the ledger held at the budget during (c) |
| F17 | 5.5's one-state reading has no fixture size | Holds | 5.5: the spare block, no 1.0.0 prefix live, `k ≥ cap_pages ≥ 3` |
| F18 | The merge-base must-reject cannot be built | Holds: the line's tip is the anchor itself | As W-4 (D-SLM7994), with the audit's constructions |
| N6 | 3.4 [F2] missing from §8's list | Holds | §8 two-sided list |
| N7 | 5.6's share and map legs cannot fire at E2 | Holds | As W-3 (D-SLM7993) |
| N8 | 5.6's another-model pool must be at 0 free | Holds | As N-2 |
| N9 | Fresh-pool cells write no shape | Holds | As N-4 |

### 15.8 Round 8 (rev 1.8): the scoped re-check of rev 1.7 and E0 references

Sources: T-3032 (`Claude/Charpy/t3032-precache-rev1p7-recheck-2026-10-01.md`, **temper**), T-3033
(`Claude/Mendeleev/t3033-precache-rev1p7-reaudit-2026-10-01.md`, **gaps named**), and T-3031
(`Claude/Mendeleev/t3031-precache-e0-reference-population-2026-10-01.md`, **inadequate** real-model
population). Their rev 1.7 base is `a29f114f77`. Every item below held on the pinned engine or the
plugin source, or by the cited T-3022 artifact record and decision log. This fold is plan text only;
T-3030 commissions the audit and T-3035 regenerates the references in their own seats.

| Source finding | Disposition in rev 1.8 |
|---|---|
| T-3032 W-1 | Held. §8 uses a writer-produced renamed pair with independent FNV hashes as audit must-accept; E0 re-runs after commissioning. |
| T-3032 W-2 | Held. 7.17 and §8 commission on the constructed anchor doc with the correct status at 30; E5 grades the edited release tree. |
| T-3032 W-3 | Held. §8 computes the merge-base in a required ctest setup at reference-test time; must-reject (3) rebuilds without reconfigure. E5 switches the line to its merged tag. |
| T-3032 W-4 | Held. 8.10 and §8 count releases only for handles returned by 1.0.0 `CreatePrefix`; a cache eviction is an explicit counter construction. |
| T-3032 W-5 | Held. 8.10 pins `B = 16`, `cap = 64`, `cap_pages = 4` and `L = 32` for phase (c); §8 restricts the initial-count floor to stated fixtures. The `ReservationPages` formula was executed at divisible caps 16, 32, 64, 256, 4,096 and 32,768: overdraw occurs at aligned `L ≤ cap − B` and stops at `L = cap`; caps 16, 32, 40 and 100 yield fewer than three pages (the latter two take `B = cap`). |
| T-3032 W-6 | Held. §5.3 splits cache begin from freeze; 8.10 requires a parked reading with K begun, none frozen, witnessed by test state. |
| T-3032 N-1 | Held. §8 names both full-history CI jobs (`tests.yml:1106-1108`, `:1481-1483`) and fail-not-skip refusal. |
| T-3032 N-2 | Held. §8 marks the audit's zero-hash branch uncommissioned if no natural name is found; 2.13's parser seam stays separate. |
| T-3032 N-3 | Held. §8 and E0 locate the T-3022 registration in scratch, direct its preservation under `D:/_artifacts/SuperSLM/t3022-precache-audit/`, and require a post-commissioning rerun. |
| T-3032 N-4 | Held. §8's barrier uses “no job executing”; an idle lane counts as parked. |
| T-3032 N-5 | Held. §8 names the lane-park must-accept, counter's failing constructions, and independent construction authors. |
| T-3032 N-6 | Held. 7.17 takes the integer immediately after “ordinal”. |
| T-3032 N-7 | Held. E5 documents `SSLM_HAS_PREFIX_SAVE_ABI` in `docs/api.md` and the release note. |
| T-3033 F19 | Held; same remedy as W-4, with independent counter constructions in §8. |
| T-3033 F20 | Held; same remedy as W-6, with required in-flight witness in 8.10(c). |
| T-3033 F21 | Held; same remedy as W-5, with distinct full-cap lengths for 1.0.0 prefixes. |
| T-3033 F22 | Held. 8.10's one-page must-reject undercharges a real-path restore via a test seam; the probe refuses its one-page overdraw. |
| T-3033 F23 | Held; same remedy as W-3. §8 states what ancestry cannot see, records the line, and fails on refusal; E2–E4 require a positive graded count. |
| T-3033 F24 | Held. D-SLM8000 places the writer's must-reject with the test author; §8 checks the collision artifact's non-name bytes and requires the audit's collision reason. A crash cannot count as refusal. |
| T-3033 F25 | Held. §8's wrapper guards every plugin fill-probe reading, and its stop-yield construction has a chosen watchdog bound. The code reviewer checks the dequeue-to-Execute state transition. |
| T-3033 N10 | Held. E1 has the macro-vitality `#error`; E2 adds the public macro to the ABI-header inventory allowlist. |
| T-3033 N11 | Held; same conditional disposition as T-3032 N-2. |
| T-3031 F5 / O2 | Held at source (`RmsNormSite` ignores incoming scale; next embed overwrites it). 7.1 names the hidden-scale-only mutant and grades it through 6.1/6.4, alongside the current-token and saturation-count fields (D-SLM7998). |
| T-3031 O1 | Held as an ambiguity in rev 1.7. Cell 10.1 now says both **adopter** modes and also both decode modes on the DGC1 artifact. |
| T-3031 reference ruling | D-SLM7997. §8 quarantines T-3022's streams as reproducibility evidence and requires T-3035's regenerated, discrimination-licensed population before E2 grading. |

No new PC decision is taken. D-SLM7997, D-SLM7998 and D-SLM8000 are existing rulings. The plan review and
coverage audit re-check the rows listed in §13 item 8 before this revision is ratified.

### 15.9 Round 9 (rev 1.9): the scoped re-check of rev 1.8 and executed E0 commissioning

Sources: T-3041 (`Claude/Charpy/t3041-precache-rev1p8-recheck-2026-10-01.md`, **temper**) and T-3042
(`Claude/Mendeleev/t3042-precache-rev1p8-reaudit-2026-10-01.md`, **gaps named**), both against rev 1.8 at
`f3c01d3354`; T-3030 (`Claude/Popper/t3030-precache-audit-commission-2026-10-01.md`), the audit's registry
entry and receipt; T-3043 (`Claude/Curie/t3043-scm1-writer-mustreject-2026-10-01.md`), the writer's entry and
receipt. Engine claims were re-read at `5d5ca58`; plugin claims were re-read in this worktree. The numeric
one-page construction is derived from source and §5.4, not executed on a built cache. No build or test ran in
this fold.

| Source row | Rev 1.9 disposition | Dependency |
|---|---|---|
| T-3041 W-1; T-3042 F1/M4 | §7 8.10 fixes a one-page LRU victim, `B = 16`, `L = 17` restore, post-eviction ledger at budget and physical holdings at budget + 1; a two-page victim is a non-refusing control. The debunker independently commissions the production-path overdraw. | U2 8.10 commissioning |
| T-3041 W-2; T-3042 F3/M7 | §8 and E0 record the actual audit-first (20:46 UTC), writer-second (20:58 UTC) commissions. The independent author and must-reject now exist; E1 consumes the result. Chronology did not meet rev 1.8's prescribed order, though both independent proofs now meet its intent before E1. | E1 opens on receipts and eight standing audit verdicts |
| T-3041 W-3; T-3042 F2/M6 | §8, E5 and R2 trigger reference regeneration on every recorded merge-base change, even unrelated tag-only movement; loader commissioning includes a tag-only refusal and positive grading after regeneration. §3/§4 rule changes still take the separate scoped re-gate. D-SLM8013. | E5 grading |
| T-3041 W-4; T-3042 F4/M10 | E1's stubbed runtime red suite stays buildable. E2 compiles the public-macro cell after defining `SSLM_HAS_PREFIX_SAVE_ABI`, then removes the macro to prove the cell goes red. | E1 runtime reds; E2 macro gate |
| T-3042 N1/M13 | 6.1's adopter `SSB6` header writes a sentinel for nonnegative post-adoption `current_token`; 6.4's `SPX1` blob equality is its oracle. Rev 1.9 placed the isolated mutant run in E1; rev 1.10 moves that run after the unmutated E2 pass (§15.10). | E1 test design; E2 mutant proof |
| T-3030, D-SLM8001/D-SLM8002 | Audit `T-3022-PC12-schema-hash-audit` is COMMISSIONED, including its natural zero-hash branch; its eight certified-artifact and example-asset readings stand. The scratch registry has no authority. Move and re-register its absolute worktree paths before pruning `T-3022`. | E1 audit evidence; next audit run after relocation |
| T-3043, D-SLM8010 | Writer `T-3030-SCM1-writer` is COMMISSIONED with the test author's independent name-swap must-reject. Move and re-register its absolute worktree paths before pruning `T-3043`. | E1 writer evidence; next writer run after relocation |

The scoped plan review and coverage re-audit take §13 item 9's rows, including the full catalog cross, before
rev 1.9 is ratified. No PC decision changes. D-SLM8013 records the reference-anchor rule; the executed
commissioning facts remain D-SLM8001, D-SLM8002 and D-SLM8010.

### 15.10 Round 10 (rev 1.10): F16 source resolution and E1/E2 gate sequencing

Sources: T-3060 F16 (`Claude/Mendeleev/t3060-precache-e1-suite-audit-2026-10-01.md`); the test author's
T-3051 repair F16 (`Claude/Curie/t3051-precache-e1-cpu-red-2026-10-01.md`); T-3046 W-2
(`Claude/Charpy/t3046-precache-rev1p9-recheck-2026-10-01.md`); and SuperSLM `5d5ca58`. The F16 answer is
source-derived, not an executed token comparison. The anchored code provides the full path, so no probe was
needed. No SuperSLM code was changed.

| Source row | Rev 1.10 disposition | Dependent gate |
|---|---|---|
| T-3060 F16; T-3051 repair F16 | Adoption copies `hidden_scale` (`sslm_abi.cpp:2655-2673`), but the first decode skips embed and layers (`:3012-3019`), `RmsNormSite` ignores its incoming scale and passes an empty incoming span to the funnel (`forward_sites.cpp:837-896`), and logits consume only the norm codes (`sslm_abi.cpp:3098-3110`). The next embed replaces the scale (`:3052-3063`); `begin_from` prefill embeds too (`:1943-1987`, `:2228-2238`). S20, §3.4 step 8, 2.3b and §14 state the resulting equality premise. The attacker-chosen `m`/`e` domains remain as written. D-SLM7998 stands; the test author's assertion that `RmsNormSite` consumes the copied scale is refuted. | E2 runs 2.3b's in-domain `m`/`e` equality cases against built verbs. |
| T-3046 W-2; T-3051 repair F2 | E1 authors 6.4's isolated nonnegative `current_token` mutant and its blob assertion. E2 records an unmutated 6.4 pass, then requires only that mutation to turn 6.4 red; both outcomes are recorded separately. An E1 stub refusal is construction evidence only. | E2 mutant proof before E2 green. |
| T-3060 F2–F15; T-3051 repair handoff | Most E1 prefix cells stop at stubbed save/restore before their deciding assertions. E2 repairs the E2 portions of F2–F15, runs their named mutants against built verbs, and has the coverage auditor re-audit discrimination before E2 is green. E3/E4 portions remain at their named gates. | E2 discrimination audit and mutant receipts. |

The scoped plan review and coverage re-audit take §13 item 10's rows and the full Coverage Model cross before
rev 1.10 is ratified. The approved promise and PC decisions remain as filed. No decision ID is minted: the
anchored source supports D-SLM7998, and this fold corrects the test author's contrary inference.

### 15.11 Round 11 (rev 1.11): complete E1 red design before the E2 build

Sources: T-3062 (`Claude/Charpy/t3062-precache-rev1p10-recheck-2026-10-01.md`), T-3063
(`Claude/Mendeleev/t3063-precache-rev1p10-reaudit-2026-10-01.md`), T-3060
(`Claude/Mendeleev/t3060-precache-e1-suite-audit-2026-10-01.md`), the T-3051 repair record, T-3046,
D-SLM8017 and SuperSLM `5d5ca58`. This fold is a plan amendment; no product build, test or mutant was run.
The F16 scale and token premises and the `current_token` sentinel were re-read at the anchored engine source.

| Source row | Rev 1.11 disposition | Dependent gate |
|---|---|---|
| T-3062 S-1; T-3063 F1; T-3060 F2–F15 | D-SLM8022: the test author completes every [E2]–[E4] fixture and deciding assertion in E1, including assertions held behind stubs; the coverage auditor rechecks that authored design against all T-3060 rows and the full catalog. This supersedes §15.10's placement of assertion repair after the E2 build starts. Stub reds establish construction, not discrimination. | E1 exit and E2 product-build start. |
| T-3063 F2; T-3060 F1 | Cell 7.12's `SPX1` token operand is the fixture's known last prefill token, not a fresh adopter's `SSB6` sentinel. Its E1-authored independent parser checks a bound nonempty fixture and fixed fields, digest and canonical rows; E2 records acceptance of a built-verb blob and separate failures for an isolated token edit and a symmetric writer/reader field swap. Standalone parser controls remain separate evidence. | E1 design recheck; built-verb format proof before E2 green. |
| T-3062 S-1; T-3063 F1/F3; T-3060 F2–F15 | E2 runs its preauthored cells and applicable mutants on built verbs, with a coverage-auditor discrimination re-audit before green. E3 runs the carried F5–F9/F11/F13 assertions and applicable mutants, including F8's hint and physical-page evidence; E4 runs F10–F13's real-artifact, separate-process/reference and timing evidence and applicable controls. The test author executes and the coverage auditor re-audits each step; the E3/E4 assertion design was cleared in E1. | E2, E3 and E4 green, respectively; no later assertion authoring is a substitute for E1. |
| T-3046 W-1; D-SLM8017 | The audit was re-commissioned after the writer at 21:48:44Z; both registrations verified COMMISSIONED and the post-writer eight-artifact batch matched the earlier batch byte-for-byte. §8 and E0 cite the executed closure rather than treating the original audit-first order as an open E1 gate. | E1 consumes the standing post-writer audit receipt and verdicts. |
| T-3046 W-2; T-3062 §2; T-3063 C1/C2 | The F16 source resolution of rev 1.10 stands. E1 authors the 2.3b scale-equality and 6.4 isolated-token assertions; E2 executes the scale comparison, records unmutated 6.4 green, then requires only the isolated nonnegative token mutant to turn it red. The fresh adopter's `SSB6` sentinel is not a token-value oracle. | Separate 2.3b, 6.4 baseline and 6.4 mutant receipts before E2 green. |
| T-3063 C6; T-3031 | T-3047 M1–M20 and T-3031 F5/O1 retain their recorded design dispositions; M13's execution is the E2 isolated-token proof. The T-3035 replacement references remain quarantined until independent discrimination commissioning and coverage re-audit license the actual population. | License before E2's first reference grading; positive graded count at E2–E4. |

T-3063's eleven catalog dimensions remain applicable. The T-3060 baseline found 59 registered IDs, 11 fully
targeted and 48 partial; those counts describe the audited stubbed suite, not rev 1.11's executed coverage.
The E1 recheck must disposition every partial ID on its authored assertions before any E2 product implementation.
Rev 1.11 re-enters the scoped gates in §13 item 11. The approved promise and PC decisions remain as filed.

### 15.12 Round 12 (rev 1.12): T-3070 A1/A2 fixture rulings

Source: T-3070 A1/A2 (`Claude/Mendeleev/t3070-precache-e1-suite-recheck-2026-10-01.md`);
SuperSLM `5d5ca58` (`src/sslm_abi.cpp:669-702`, `src/model.cpp:774`,
`tests/paged-kv/pkv_legacy_a_helpers.h:97-128`, `include/superslm/artifact.h:222-226`);
the `SPX1` table in §3.2 and the E1/E2/F2 step order in §9. This is a plan and fixture ruling;
no product build or mutant was run.

| Audit row | Rev 1.12 ruling | Proof boundary |
|---|---|---|
| A1, 7.12 counts | The test author fixes five nonzero, pairwise-distinct expected count values before either blob is serialized and supplies them to the built-blob parser; `SSB6` corroborates but does not define them. An isolated symmetric count swap must fail. A standalone parser control is separate evidence. | E1 authors and the coverage auditor checks the assertion design; E2 runs it on a built save. |
| A1, 7.12 B views | `ComputePageGeometry` takes B from CFG1 `kv_block_size`. Patching that field to form B=4 changes the artifact integrity hash that the `SPX1` model-hash field must carry. Compare bytes outside digest [28,60) and model hash [60,92) only after independently confirming equal pre-save semantic fields and rows, distinct expected model hashes and valid per-blob digests. The comparison covers every other byte, including canonical K/V rows. Whole-blob equality between these different model identities is withdrawn. | E1 authors both views and their equality/must-differ checks; E2 executes on built saves and a B-dependent row/layout mutant must turn the comparison red. |
| A2, 5.7 CPU precedence | E2 has no real GPU save verb: §9 builds it at F2 after the CPU release. The E1 fixture derives both a backend-only control and a GPU-shaped blob from an E2-good CPU save, checks all preserved fields independently, and recomputes each digest. The GPU-shaped current-version operand and its version-edited twin must both return `SSLM_INVALID_ARGUMENT` from CPU restore; swapping CPU steps 4 and 5 must change the paired case. This is evidence of CPU precedence over a constructed, valid later-field population. | E1 authors the construction and checks; E2 runs the controls, pair and step-swap mutant on built CPU restore. F2 captures a real GPU save, validates its fields and repeats the pair before the real-GPU part of 5.7 is credited. |

T-3070 A3/A4 remain test-author repairs under D-SLM8025. E1 clearance still requires all four repairs and
the coverage auditor's independent recheck. Rev 1.12 re-enters the scoped gates in §13 item 12.

### 15.13 Round 13 (rev 1.13): the E2 re-anchor on 1.13.0 and the paging plan's cross-plan notes

2026-10-15. Sources: the E2 re-anchor record (`build/e2-reanchor-v1.13.0.md`, beside this file: §2 F1–F11, §3's
scratch probe, §5's remaining work), which merged the E2 build with `v1.13.0` (`a5f3ad1`) at `6426a72` and re-anchored
at `3b6631f`; the plugin paging plan, rev 1.7 (`/mnt/project-files/plans/superslm-unreal-paged-kv/plan.md`), §4's
cross-plan notes, §3.2, §3.5, §6, §13 items 8, 9 and 15, R9 and R11; and SuperSLM `3b6631f` (branch
`claude/precache-1p14-on-1p13`), read-only. The planner re-read every cited site at `3b6631f`: `PageRuleB`,
`ComputePageGeometry`, `ReservationPagesIn`, the three pool creates, adopt's page-size check (`sslm_abi.cpp:2854`), the
header and `docs/api.md` sentences, the `e4_real.cpp` geometry assertions, `PinFixtureB` and the pkv fixture caps. It
also read the `a5f3ad1..3b6631f` diff of `src/` and `include/` (five files, additive, no GPU file). Every finding held
at source. Rev 1.12 is kept beside this file as `plan-rev1.12.md`. No build or test ran in this fold; the rule-B
figures are executed arithmetic (§8).

**Rule change.**

| Source row | Rev 1.13 disposition | Dependent gate |
|---|---|---|
| F5: adopt is refused between pools of different page sizes, even for a whole_reserve (copy) adopter; restore takes B from its pool, and one `SPX1` restores into either pool kind | §3.4 *What adopters see* and promise (i) gain the qualifier (a third scope statement beside rev 1.3's two; the sentence of (i), §0's promise and §13 item 2's quote are unchanged); §3.1 says B is the restore pool's; §3.6's *keeps* reads B as the restore pool's, and new row X13: a restore into a pool of another page size keeps, and an adopter across page sizes is refused as for a prefill-built prefix; §3.8 header text; S30. New cells 8.13 (equal cross-geometry dispositions, restored against built), 9.8 (one blob through both pool kinds, with a single-page count where `ProbeSplit` cannot read) and 7.12's pool-kind leg (whole-blob equality across pool kinds, one model). | Scoped re-gate (§13 item 13) before E3; 8.13, 9.8 and 7.12's leg before E2 green. |

**Does F5 change anything the principal approved? No.** Four reasons, each checked:
1. At approval (rev 1.4, read at `f4a1b6a`) and through rev 1.12 (`5d5ca58`), B came from CFG1, so every pool of a
   model had one page size. The case the qualifier excludes did not exist, so it narrows no case the approval
   covered. 1.13.0 introduced it (the contiguous option, PKV §21.2, D-SLM8398) and refused adopt across it (PKV A22,
   D-SLM8434); both are paged KV's decisions, made under that plan.
2. The refusal treats a restored prefix and a prefill-built one identically, by name (`SSLM_INVALID_ARGUMENT`). An
   adopter is never given other tokens; promise (i)'s comparison still holds wherever an adopter is admitted.
3. The plugin, the plan's one consumer, keeps one page size per backend. Checked against the paging plan: the CPU
   subsystem has one pool, `sslm_kv_pool_create` in legacy mode or `sslm_kv_page_pool_create` in budget mode (paging
   §3.2, §4), and both take the model's rule B (S30 (c)). The GPU subsystem has its own pool, and cross-backend adopt
   is refused anyway (paging §6). Paging never calls the contiguous option. The cache restores into and adopts
   within its subsystem's own pool, and a mode change re-`Configure()`s and releases every cached handle (§5.4). So
   no behaviour of PC-3 or PC-11 changes, and cooked or disk blobs, carrying no page size, restore under either mode.
4. 1.13.0's own contract names save and restore as the way to move state across pool kinds
   (`docs/api.md:733-736` at `a5f3ad1`; *rev 1.14:* rev 1.13 cited `sslm_abi_functions.inc:172-174` and
   `docs/api.md:1036-1039` at `3b6631f`, which are this plan's header text, written at E2), and the qualifier is the reading §3.1's
   "carries no page size" already implies.

So F5 needs no principal decision. It is a §3 rule change, so it re-gates (§8, R2), and whether the adversary
strikes the new scope statement is the conductor's call.

**Text and numbers.**

| Source row | Rev 1.13 disposition | Dependent gate |
|---|---|---|
| F2: B is the rule's per model, CFG1 no longer selects it; the E1/E2 cells are pinned at B = 16 and B = 4 | §3.2's resident-page column redone at the rule B (128 at the real 0.5B artifact; 64 at cap 4,096); §3.5's cohort at the rule B (7 shared, 3 private, scatter from 896 at `B = 128`; 15, 4, 960 at `B = 64`); §3.4 *Peak draw*'s strike cell (3 vs 10; 5 vs 19), with B = 16 kept as cell 5.10's pinned view; S30 (a), (f); the §7 preamble names the pin; 7.12's B = 4 view now comes from the pin; 8.10's plugin fixture re-derived (`B = 64`, `cap = 256`, `cap_pages = 4`, since the rule cannot give rev 1.12's `B = 16` at cap 64 and plugin fixtures have no pin) | Scoped re-gate |
| F2: E4's real cells assert `page_size == 16·bytes_per_token` (`e4_real.cpp:60`, `:181`, `:269`, `:322`) and go red on real artifacts | **Re-derive, not pin** (10.1): B computed in the test from the rule (128 at 0.5B, 64 at 1.5B). 10.x is the feature oracle and 10.3 times the shipped page size; the pin is an engine test seam no game can call, and the pinned shapes are already covered. §9 E4 | E4 |
| F2/F4: coverage gap, no E2 cell restores at a rule B or at a partial last page | New 4.7 [E2, E3]: six models unpinned at their rule B, pkv_odd's partial page (cap 4,100, B 256, last page 4 positions), `L` through the partial page, the draw read by the fill probe, rows peeked with the rotation, a nested leg at E3, and three mutants. 8.13, 9.8 and 7.12's leg cover the contiguous option | E2 green (4.7's E2 legs); E3 (its nested leg) |
| F4: §3.4 step 3 said "`B` divides the cap, and the table has `context_cap / B` entries" | §3.4 step 3, S26 and 11.1: the table has `ceil(cap/B)` entries and `ceil(L/B) ≤ ceil(cap/B)` for `L ≤ cap`; the branch stays unreachable and 11.1's exemption stands; 4.7 executes the partial-page premise | Scoped re-gate |
| F6: `ReservationPages(model, budget)` became `ReservationPagesIn(pool, budget)` | §3.4 *Peak draw*: every comparison is in the restore pool, with that pool's `R`; in the contiguous option `R = ceil(L/B) = 1`; S30 (d) | Scoped re-gate |
| F1, F3, F7–F11 | No fold, as the record finds. F3's rotation is used by the new cells' peeks (`pkv::PageRunOffset`). F8's version bump stays E5's | — |
| §5 item 5's "a fold note that R2's re-diff ran here" | §8's base: E2 merged `v1.13.0` early (`6426a72`), E2–E4 now build on it, and the re-anchor record is R2's re-diff; its one rule change is F5. The tag was merged around `<line>`, the structural half's stated blind spot, so the line switch moves from E5 to E2, before E2's first grading; R2 | E2's first reference grading |
| §5 items 1–4, 6–9: what remains for E2 green | §9 E2's gate lists them as receipts (1)–(9): the MSVC ASan triage at `e2_format.cpp:160`; 6.4's mutant receipt at the box; T-3060 F2–F15 mutants and the discrimination re-audit; the coverage re-audit of the pins and `PageRunOffset`; references at `v1.13.0`; the box legs, the 1.5B grid included; `c_abi_verbs.txt` on Windows; code review of `6426a72` and `3b6631f`; this fold's re-gate | E2 green |

**Cross-plan notes from the paging plan (§4).**

| Note | Rev 1.13 disposition | Dependent gate |
|---|---|---|
| (a) "Holds while adopters copy" | §5.1 (paging ships first, so budget mode exists from U2); §5.4's three notes (the argument stands in legacy mode); §5.11 (a): in budget mode, C bounds the cache and paging's region A holds what adopters keep after an eviction. One ordering is this plan's (paging §13 item 15, R9): a cache charge is refunded only when the freeing release has applied, a request waits only on decided evictions' refunds, and the eviction loop counts pending refunds. That is §5.4's "counts every page the cache holds" read exactly, not a PC change (*rev 1.14:* overstated; it is a new admission rule, specified in §5.11 (a), still not a PC change, §15.14). Cell 8.14 | U2 |
| (b) Region C's wiring | §5.11 (b): `CacheRegionPages()` and the region-C seam in budget mode, the unchanged `sslm_kv_pool_create` in legacy mode, adopts and hints through the one adopt entry point, `PrefixLeaseEnded` after paging's values; paging's [cache] box legs at U2's gate; §9 U0, U2; §14 | U0, U2 |
| (c) The second save boundary | **PC-13, for the principal, recommended *Accept*.** §3.8: 1.14.0 changes no sequence writer, reader or GPU source (diff read at `3b6631f`); §5.11 (c): the engine half (9.9, 9.10 at E5) and the plugin half (the `'SLUW'` 1.13.0 entry, 9.11 at U2); R10 names the failure path, which keeps the boundary by name. F4 re-asks the question at 1.15.0 | E5 (9.9, 9.10); U2 (9.11); the principal (PC-13) |
| Enum order; `NestedRequiresBudgetMode`; numbers; the moved reads | Enum order and `NestedRequiresBudgetMode` noted in §5.11 (b) and §5.6/§5.9. The plugin release numbers and paging's moved §13 reads were not in this fold's brief; §0's plugin numbers ("1.1.0", "1.2.0") still read as before, and paging §5.1 records that they move | Conductor |

**New cells for the test author:** 4.7 [E2, E3], 8.13 [E2], 9.8 [E2], 7.12's pool-kind leg [E2]; the 10.1–10.3
geometry re-derivation [E4]; 9.9 [E5], 9.10 [E5, box]; 8.14 [U2], 9.11 [U2]. The E2 ones are written after the build,
so each counts only with its named mutant's red receipt (§9 E2).

No decision ID is minted here; PC-13 is a plan decision for the principal and gets its ID when the tracker files it.
The approved promise sentence and PC-1..PC-12 remain as filed. Rev 1.13 re-enters the scoped gates in §13 item 13.

### 15.14 Round 14 (rev 1.14): the scoped re-gate of rev 1.13

2026-10-15. Sources: the plan review of rev 1.13 (`reviews/plan-review-rev1p13.md`: **HOLD**; B1, S1–S4, N1–N6) and the
coverage audit of rev 1.13 (`reviews/coverage-audit-rev1p13.md`: **HOLD**; B1, S1–S7, N1–N9), both scoped to §13 item
13. The planner re-read the cited test sites at `3b6631f` (read-only): `precache.cmake:32`, `:59-61`, `:65-69`;
`pkv_common.h:236-242`, `:296-310`; `precache_common.h:247`, `:274`; `paged_kv.cmake:246-252`; `e4_real.cpp:59-72`,
`:175-185`, `:265-275`, `:318-340`, `:388`; `e2_root.cpp:26`, `:72`. Every finding held at source and landed. Rev 1.13 is
kept beside this file as `plan-rev1.13.md`. No build or test ran.

**Blocking.**

| Finding | Rev 1.14 disposition | Dependent gate |
|---|---|---|
| Review B1: E2's receipt (6) names `real_two_process` (10.1/E4, 10.2/E4) and `e3_reference` (6.3/E3), which cannot be green at E2 | Moved to their stages: `real_two_process` is an E4 receipt, after 10.1's re-derivation; `e3_reference` an E3 receipt. Receipt (6) keeps `e2_reference` (6.2/E2) and states why the other two are red at E2 and are not findings there. §9 E2, E3, E4 | E2 green; E3; E4 |
| Audit B1: X13's nested case has no cell; a hint computed at the model's B passes every listed cell | New nested leg of 9.8 [E3]: a nested blob, `origin ∈ {B_r, B_r + 1}`, restored into a contiguous pool (i) with a matching contiguous parent (`s = 0`, one page drawn by the single-page count, re-saves byte-identical, the parent's included, adopters equal the contiguous twin's), (ii) with a page-pool parent ((d) fails), (iii) `origin = L = cap` (`s = 1`, 0 drawn). Mutant: the hint's `s` or span from the model's B. §3.5 states the contiguous-pool hint; X13's cells and E3's carried list name the leg | E3; this re-gate |

**Should-fix.**

| Finding | Rev 1.14 disposition | Dependent gate |
|---|---|---|
| Review S4, audit S1: the unpinned cells live in a binary that pins; three would pass at B = 16 | §7 preamble: each unpinned cell maps its own model instance, feeds the test-written rule B to every helper (a `PKV_FIXTURE_B_FROM_RULE` target without the pin define, or the fixture's test-side `pinned`), and asserts the engine's B and `fx.B()` equal the rule before anything else. 8.13 and 9.8 assert rule B < cap. Receipt: a 9.8 through `GetFixture("pkv_def")` goes red on that assertion (§9 E2 receipt (4)) | E2 green |
| Review S3, audit S2: E4's re-derivation misses `Fixture::B()`, the pool sizing and `:388` | 10.1: E4 cells compile with `PKV_FIXTURE_B_FROM_RULE` (pinned fixtures keep their pin, `pinned > 0` first) and assert `fx.B()` (*rev 1.15:* withdrawn as false at source, re-check R1; each real fixture sets `fx.pinned` instead, §15.15); pools sized at the rule B (`:70-72`, `:272`, `:330-337`); `:388` re-derived to `s = 7`; 10.2 asserts the 3-page draw by the fill probe; the *Geometry* note applies to 4.3 and 10.1–10.3 (`:181` is 4.3's). §9 E4 | E4 |
| Audit S3: 6.6 has no rule-B leg | 6.6 gains a rule-B leg on the real 0.5B artifact (`L = 300`, 32 tokens against serial decode, with its mutant); the `Parallel66` leg is recorded as pinned; §9 E4's wording corrected | E4 |
| Review S2, audit S4–S6 (eviction loop): 8.14's mutant survives on lane order and at `L ≡ 0 (mod B)`; C graded by its own function; the eviction count unasserted | 8.14 rewritten: E's release held behind a batch gate; E at `L = 2B + 1` (`k = 3`, `f = 2`); a required reading between the eviction decision and the release's apply, absent → fail; region C computed in the test at a non-multiple budget (mutant `floor`); exactly one eviction and F still a `Memory` hit (mutant: the loop ignores pending refunds); the `AdoptPrefix` bypass mutant; "nothing evictable and no eviction pending"; waiter legs. 8.10's must-reject states the wait for the victim's release apply | U1 (red design); U2 |
| Audit S7: 7.12's pool-kind leg does not state `L` | Bound to 7.12's `L = 65` fixture on the synthetic 0.5B shape at rule `B = 64` (`L ≥ B + 1`), with the B assertion | E2 green |
| Review S1: §5.11 (a) is a new rule, not §5.4 read exactly, and is under-specified | (*Rev 1.15:* the earmark rule below left unclaimed refunds no loop could count; corrected, §15.15.) §5.11 (a) now says it is a **new admission rule** (deferred-refund admission) and specifies it: pending refunds earmarked to the request whose loop decided them; a waiter takes the free charge at once, and each earmarked refund passes to it at its apply, so the ledger never double-counts or exceeds the budget; admission at the last earmarked apply, two in one pass in arrival order; a waiter is never short at its apply, so it never re-runs the loop or resolves `BudgetExhausted` after waiting; coalescing attaches to the waiter; at re-`Configure()` and teardown it resolves by name as 1.7's queued request does and draws nothing. §5.4's charge rule, §5.5's delivery note and §5.9's row amended | U2; this re-gate |
| Audit S6 (commit rule) and N6: 9.11 lacks the tag-and-commit case and enumerates nothing outside the four | 9.11: a 1.13.0 tag with a foreign commit refused `Layer1Mismatch`, conditional on U2's read of the commit field (§14), mutant "commit comparison dropped"; cases for `SSB4`, `SSB3`, `SSB2` and `SPX1` | U2 |

**Does the deferred refund touch anything the principal approved? No.** PC-11 says a request "that cannot be charged
resolves `BudgetExhausted` by name". The bound (the ledger never exceeds the budget; nothing overdraws) and the cost
(a named refusal) are unchanged; "cannot be charged" now means not chargeable even after every evictable entry's
refund, which is what PC-11 bought. The one difference a caller sees is a later tick for a request that needed
evictions, which is §5.5's existing delivery rule. The plan review reached the same view. So no PC is raised; PC-13
remains the only open principal item.

**Nits.**

| Finding | Rev 1.14 disposition |
|---|---|
| Review N1 | §3.4 *Peak draw* and §3.5 say "executed arithmetic" for the rule-B figures |
| Review N2 | §3.4 *What adopters see* and §15.13 reason 4 cite 1.13.0's own sentence, `docs/api.md:733-736` at `a5f3ad1`; the `3b6631f` lines are named as this plan's header text |
| Review N3 | 8.10: "for the fixture's `D = 64`, int8" (true for `D·elem ≤ 128`) |
| Review N4, audit N1 | 8.13 states `L = B + 1`, records the budget legs (A16) and the reverse leg (mapped count) as non-discriminating for A22, and takes the A22 receipt in Release |
| Review N5 | §14: paging rev 1.7 is unapproved; §5.11 re-folds if its delta changes |
| Review N6 | §14 item and the cross-plan note below |
| Audit N2 | 9.8 names its deciding operand (status or single-page count, at `L ≥ B + 1`) and runs one model at a time |
| Audit N3 | 4.7 decodes `min(64, cap − L)` tokens, none at `L = cap`, as 4.1's code does |
| Audit N4 | Recorded in the §7 preamble: 5.10's rule-B figures, 8.1's share count and 8.3's `restore_shared` run only pinned |
| Audit N5 | 9.9's size-rule mutant is a tightened rule |
| Audit N7 | §3.1 and §3.4: "two page sizes" only where the rule B is below the cap |
| Audit N8 | Recorded, not celled: `PrefixLeaseEnded`'s ordinal after paging's three values has no assertion (U1 may add one beside the enum's append rule); the §3.8 header sentence is the code reviewer's at E5; R10's "only a release that re-ran 9.9/9.10 adds a row" is a process rule with no check |
| Audit N9 | 9.8's contiguous pools on the synthetic 1.5B shape (448 MiB blocks, `N ≥ 3`) budgeted in §9 E2 |

**Cross-plan note to the paging planner:** `/mnt/project-files/plans/superslm-unreal-paged-kv/reviews/xplan-note-from-precache-rev1p14.md`.
It says that paging §3.2's and §4's "PC's ledger refunds C at eviction" is stale under §5.11 (a), with paging's
safety argument intact, and that the two plans read the 0.5B cap differently (paging 4,096 / B = 64; this plan
32,768 / B = 128 for the certified artifact), which one is right, and what paging would change.

No decision ID is minted here. The approved promise sentence and PC-1..PC-12 remain as filed; PC-13 is still the
principal's. Rev 1.14 re-enters the scoped gates in §13 item 14.

### 15.15 Round 15 (rev 1.15): the re-check of rev 1.14

2026-10-15. Source: the re-check of rev 1.14 (`reviews/recheck-rev1p14.md`: **HOLD**; R1 and R2 blocking, one
non-blocking condition on 8.14, one cross-plan correction). It found both rev 1.13 B1s fixed and every should-fix
landed. The planner re-read `pkv_common.h:296-310` (the loader's pin leaves `pinned` at 0), `precache.cmake:3-5`,
`:59-61`, `e4_real.cpp:57-59`, and the vendored `t2922_real_model_manifest.json` in the plugin tree. Every finding held
at source and landed. Rev 1.14 is kept beside this file as `plan-rev1.14.md`. No build or test ran.

| Finding | Rev 1.15 disposition | Dependent gate |
|---|---|---|
| R1: `PKV_FIXTURE_B_FROM_RULE` on the E4 cells turns `GetFixture`-pinned cells red target-wide (the loader never sets `pinned`), and per file is an ODR violation; rev 1.14's "pinned cells unchanged" is false | 10.1: each real E4 fixture (`Real101`, `Real102`, `Timing103`, 4.3's, 6.6's rule-B leg; `wf` for 1.5B) sets its test-side `pinned` to the test-written rule B and asserts it equals the engine's B. Chosen over a separate target because only the hand-built real fixtures change, `real_two_process` keeps its one binary, and the pinned cells are untouched by construction. The macro is never defined in `superslm_precache_e1`; the §7 preamble says so; §9 E4; §15.14's row marked withdrawn | E4 |
| R2: rev 1.14's earmark rule leaves a waiter's surplus and a refused request's evictions unclaimable, so a request within the budget can be refused where the approved rule admits it (the E/W/X example) | §5.11 (a): pending refunds are claimed or unclaimed; a request covers its charge from free charge, then unclaimed pending refunds, then new evictions, each claim earmarked and transferred at apply under the same rule; a refused request's claims return to unclaimed. The "PC-11 unchanged" argument is now an induction: given equal requests and job completions, this rule's free charge plus the unclaimed refunds equals the approved rule's free charge at every decision, so both admit, refuse and evict identically, and a waiter only draws later. The residual is lane timing (a waiter's freeze or failure applies later), which PC-11 never fixed. §5.4, §5.9, §9 U2 amended. 8.14 gains the E/W/X leg and an orphaned-refund leg, with the mutant "loops cannot claim unclaimed refunds" (X, Z refused) | U1 (red design); U2; the re-check |
| Re-check §4: 8.14's non-earmarked mutant survives when the first victim is smaller than the second request's need | The waiters leg states free charge 0, two victims of equal charge, each need equal to it; the mutant's condition is written beside it | U2 |
| Re-check §7: the rev 1.14 cross-plan note's §2 steers paging wrong | The note's §2 is corrected: the plugin ships the cap-4,096 0.5B artifact (`qwen2.5-0.5b-instruct-cap4096`, per the vendored `t2922_real_model_manifest.json`), so paging's 0.5B figures at `B = 64` stand; this plan's cap-32,768 figures are E4's certification artifact. "The outlier" and the proposal to redo paging's figures are removed; item 1 is kept. §14's item is settled, and 10.1's note on 10.3's page size names the certification artifact | — |

**Does anything now need the principal? No, if the re-check accepts §5.11 (a)'s induction.** With claiming, the
deferred refund admits and refuses exactly what the approved rule does for equal completions, so PC-11 is unchanged
and raises no PC. The plan says plainly that the comparison is for equal job completions: a waiter's build starts
later, so its freeze is applied later, and a later request can see the ledger before that drop. Two runs of the
approved rule at different lane speeds differ the same way. If the re-check rules that this residual is a change
to PC-11, the added refusal goes to the principal as a PC-11 amendment; the plan does not decide that for the
principal. PC-13 remains the one open principal item, recommended *Accept*. Rev 1.15 returns to the re-check (§13 item 15).

### 15.16 Round 16 (rev 1.16): the strike on rev 1.15

2026-10-15. Source: the blind strike on rev 1.15 (`reviews/strike-rev1p15.md`: **FRACTURED** on target 2, §5.11 (a),
for nested entries; probes in `reviews/strike-rev1p15-probes/`). Targets 1 (promise (i)'s page-size scope and X13),
3 (PC-13's CPU half, six 1.13.0 saves restored across builds) and 4 (every rule-B figure, with engine draws at
B = 128, 64 and 256) held on executed runs. The planner re-read `src/gpu/gpu_1p0.cpp:66-71` at `3b6631f` and the
strike's `refund_model.py`, which it extended and re-ran. Rev 1.15 is kept beside this file as `plan-rev1.15.md`.

| Finding | Rev 1.16 disposition | Dependent gate |
|---|---|---|
| F1 (fracture, U6): a nested waiter does not pin its parent, so the rule evicts and refuses differently from the approved rule, and the waiter's job can start against a released parent | §5.4 *Pins* and §5.11 (a): a waiter pins its spec's parent from its decision, passing the pin to its job at admission; a nested request's own loop never evicts its parent. The induction adds equal pin sets. 8.14 gains the strike's construction [U6] (C = 12; V, P, Q, L): X refused as under the approved rule; mutant "no pin while waiting" | U6 (red commit); re-strike |
| F2 (fracture, U6): the ghost's end under the deferred refund is unspecified; refunded at the decision it overdraws physically, held unclaimable it refuses what the approved rule admits | §5.11 (a) *Ghost charges under the deferred refund*: a ghost that ends at a descendant's eviction joins that eviction's pending refund, held to its apply and claimable under the claim rule. The induction places it in `e_w` or `U`. 8.14 gains the strike's construction [U6] (C = 10; P ghost 4, D 2, E 4 leased) with both readings as mutants | U6 (red commit); re-strike |
| L1: a stale release completion after re-`Configure()` could refund into the rebuilt ledger | §5.11 (a): pending records and completions carry the configuration generation; a stale completion is dropped and refunds nothing. 8.14's re-`Configure()` leg reads the ledger after the next apply, with a mutant | U2 |
| L2: "no GPU source" hides that PC-12's refusal changes GPU map acceptance | §3.8 and 9.10 name it: `schema_masks.h` is included by `src/gpu/gpu_1p0.cpp:70`; 9.10 is a regression proof for the save formats, not for GPU map acceptance | E5 |
| L3: 10.2's `s = 7` and draw 3 are literals | 10.2 computes `floor(|world|/B)` and `ceil((|world| + |persona|)/B) − s` from the files it reads | E4 |

**The executed model.** `reviews/fold-rev1p16-probes/refund_model.py` is the strike's model with the two fixes
(`fixes=True`), configuration generations, transitive descendants for ghosts, and a part 5 that runs random nested
sequences. Run without `--fixed` it reproduces the strike's output (`refund_model.strike.out`; only two printed
records gain a field). With `--fixed` (`refund_model.fixed.out`):
- U2 roots: 3,000 runs, no divergence (unchanged).
- F1 construction: approved rule `now`/`refused`, evictions V, Q; this rule `wait`/`refused`, evictions V, Q; no
  fault.
- F2 construction, both readings: this rule waits where the approved rule admits at once, and the cache's physical
  pages stay at C (10) where the approved rule's reach 16.
- L1: after `Configure()` and a stale apply, free charge 0 and ledger 8 of C = 8.
- Random nested runs, three seeds of 20,000: 297,115 requests, 125,042 nested, 51,015 waits and 11,128 ghost endings,
  with **0 divergences**, ledger never above C, physical never above C, no job against an evicted parent. The same
  generator without the fixes: 3,410 of 20,000 runs diverge (the first on pin sets).

**Does either fix change PC-11? No.** Both restore, for nested entries, the decision-for-decision equivalence §5.11
(a) claims for roots: the waiter pin reproduces the pin the approved rule's queued job holds from the same decision,
and the claimable ghost is exactly the charge the approved rule frees at the same decision, held until its pages are
free. Neither adds a refusal or a cost, so nothing goes to the principal. (*Rev 1.17:* the ghost fix did not hold
with releases applied out of order, and the own-parent clause added a refusal for nested restores; both are
corrected in §15.17.) PC-13 remains the one open principal
item, recommended *Accept*. Rev 1.16 re-enters the gates in §13 item 16.

### 15.17 Round 17 (rev 1.17): the re-strike on rev 1.16

2026-10-15. Source: the re-strike on rev 1.16 (`reviews/restrike-rev1p16.md`: **FRACTURED** on R1; R2 lesser; F1
pin-on-wait, L1 generations and coalescing onto waiters held across 60,000 adversarial runs), and its probes. Rev 1.16
is kept beside this file as `plan-rev1.16.md`. No engine build or test ran; the model runs below are this fold's
re-check.

| Finding | Rev 1.17 disposition | Dependent gate |
|---|---|---|
| R1 (fracture, U6): a ghost refunded at the apply of the eviction that ended it, while a sibling's or the ghost parent's own handle still maps its pages, lets the cache's handles hold more than C when releases apply out of order | The re-strike's fix (a), not lane serialisation: §5.11 (a) *Pending refunds wait on their family*. Every pending refund, an evicted entry's own charge and an ending ghost's alike, completes only when every pending release among that entry's own handle and its descendants' handles has applied, and stays claimable until then. The ordering bullet names the last release as the job that frees shared pages; *No double counting* admits a waiter when its last claimed refund completes. The rule covers non-ghost parents too: with it applied to ghosts only, the re-strike's generators still found 23, 7 and 23 overdraws, from a parent evicted after its child's eviction but before the child's release applied. 8.14 gains the sibling leg (D1's release gated behind D2's), the chain leg and the parent-after-child leg, each with its mutant. The model's page counter is now refcount-accurate | U6 (red commit) |
| R2 (lesser): "a nested request's own eviction loop never evicts its parent" refuses a nested restore that §5.4 as written admits | §5.4 *Pins*: a nested restore's loop may evict its own parent and then restores privately with a null hint, which adds no refusal; a nested build's loop never evicts its parent. For builds this is an added refusal against the literal text, whose admission would start `begin_from` against a released handle (caller UB) and overdraw; the only sound reading refuses too. It does not change PC-11 (the named refusal is PC-11's cost; nothing is overdrawn). 8.14 gains the own-parent leg for both kinds | U6 |

**The re-check run** (`reviews/fold-rev1p17-probes/`). `restrike_model.py` is the re-strike's file, byte for byte; it
imports `fold_refund_model.py`, the planner's model with the rev 1.17 rules (family sets on every pending refund, a
refcount-accurate page counter, request kinds for R2). `restrike_model.out`:
- Sibling construction, releases out of order (D2, then D1): at D2's apply nobody is admitted, physical 9; at D1's
  apply X and Y are, physical 10 = C.
- The re-strike's own-parent construction (a restore): both rules evict P; this rule waits where the approved rule
  admits at once.
- Random runs with its generators, releases out of order, seeds 7, 11 and 23, 20,000 each: 405,247 requests, 154,004
  nested, 77,457 waits, 20,817 coalesced onto waiters, 10,760 re-`Configure()`s mid-wait, 18,628 ghost endings.
  **Divergences 0, ledger over C 0, physical over C 0, parent faults 0.**
- The same with releases in decision order: 0 on every count.

`fold_runs.py` (`fold_runs.out`) adds the 8.14 constructions and the ablations on the same generators:
- the rev 1.16 ghost rule: 608, 612 and 777 physical overdraws;
- family sets on ghosts only: 23, 7 and 23;
- rule A reading §5.4 literally for builds (a build may evict its own parent): 2,922, 2,898 and 2,787 diverging
  runs. These are exactly the runs in which that reading's build evicted its own parent, which is UB in the engine.
  This is the R2 refusal for builds, counted.

**Does anything need the principal? No PC is raised.** The family rule only delays completions, so it adds no
refusal: the induction's `e_w`/`U` bookkeeping is unchanged and the run shows 0 divergences. R2's build refusal is
the one difference from §5.4 read literally. It is disclosed above and in §5.4, and the literal admission it replaces
is unsound (caller UB and an overdraw). The plan judges that this leaves PC-11's text, buys and costs unchanged. The
conductor may still show it to the principal as a disclosure. PC-13 remains the one open principal item,
recommended *Accept*.

### 15.18 Round 18 (rev 1.18): the principal's build answers, release numbering and API additions

Rev 1.18 folds three sets of the principal's decisions and nothing else: the nine recommendations of
`build-2026-10-17/open-questions.md`, accepted 2026-10-17 14:51Z (PC-15); the release numbering and the squash to
`main`, 2026-10-20 (PC-1 renumbered, PC-13 retargeted, PC-14); and two plugin API additions, folded on the
principal's word 2026-10-20 20:52Z (PC-16). Repository state is from git on 2026-10-20 and
`merge-path-2026-10-20.md`. The fold's note is `reviews/fold-rev1p18.md`; its executed run is
`reviews/fold-rev1p18-probes/`.

| Item | Rev 1.18 disposition | Dependent gate |
|---|---|---|
| Q1: a GPU prefix's schema-rejected prefill after landing tokens (and the budget stop, V5) leaves its snapshot invalid while a restored copy's is valid | §4 item 7 (refresh, not refuse; prefix layer only); cell 6.8 [F2] (`pcg_cell_budget_snapshot`, both legs, mutant), from `fdbb97b` and `83a82c2` | F2 box run; code review of `fdbb97b`, `83a82c2` (E2 receipt (8)) |
| Q2: the page-size qualifier belongs in promise (i) | §0 promise; §3.4 (i) sentence and its third scope statement; §13 item 2; 7.1 | scoped re-check |
| Q3: GPU rules F2 built that the plan did not name (R1–R5, V4) | §4 item 1 (`current_token`, `kv_precision`, the at-rest rule), item 2 (derived ready/snapshot/bind; the share refusal settled), item 3 step 11a; 5.11's GPU half no longer pending | F2 box run |
| Q4: `AdoptPrefix` on a lease not yet Ready | §5.10 row (`Malformed`, plugin 1.3.0); cell 7.21 [U2] | U2 |
| Q5: the leased-handle restore hint is not wired (V9) | §5.11 (b) *Adopt and hint*; cell 8.15 [U2, box] | U2 |
| Q6: region A for nested parents not built (R36) | §5.11 (a) last bullet; 8.14's nested-A leg [U6] | U6 |
| Q7: CPU build one job; GPU build in chunks under the 10 s bound (R13, R40, W4) | §5.3 item 4 (supersedes rev 1.8's job split); 8.10 (c) through the new build hold (§8), K lanes (§14); 5.9's one-job leg; cell 5.13 [U6] | U1 (8.10's design), U2, U6 |
| Q8: compatibility rows list both commits | §5.11 (c) (1.13.1: `d697927`; 1.13.0: `a5f3ad1`, `9e235965`; `002094a6` leaves); 9.11. Open: keep the 1.13.0 row (PC-15) | U2; the principal |
| Q9: device-loss re-issue spin (W3); editor build on the game thread (W5) | §5.4 *After a terminal GPU device loss*, with the U2 draft's failed-release rule (V12) recorded as premise; §5.8 background build; cells 5.14 [U6], 3.11 [U3]; §5.9 row | U3, U6 |
| Numbering (PC-1) | §0 *Releases* and *Steps*; §4 heading and preamble; §5.11 (c); §6; §9 F4, U5, U6, U9; §11; §12 PC-1; index | scoped re-check |
| 1.13.1 retarget (PC-13) | §0 PC-13; §3.8; §5.11 (c); 9.9, 9.10, 9.11; §9 E5 | E5; U2 |
| The squash to `main` (PC-14) | §0; §3.8 (version at the squash; "no GPU source" withdrawn); §4 preamble (exported, labelled unreleased); §8 `<line>` after M1; §9 M1 row, E5; R11, R13; §14. Open: when M1 runs, the version it carries, labelled or gated GPU verbs (PC-14 (a)–(c)) | the principal; M1 |
| Where it stands; long legs to the work machine | §0 *Where it stands*, *Principal's time*; §1 constraint; §9 preamble, E2 receipt (6), E3 and E5 rows; E2 receipt (8) widened to `3a00abb`, `fdbb97b`, `83a82c2`; E4 depends on the re-derivation; R12 | E2, E3, E4 |
| Residency control (PC-16 (a)) | §5.5 rows; §5.6 *Explicit residency control*; §5.9 rows; §6 two new left-outs; cells 1.10, 2.16, 3.10, 5.12, 7.18, 7.19, 8.14's drop legs, 10.7 [U4], 11.7; §8 seams; executed `drop_model.py` | U1, U2, U4 |
| Text entries (PC-16 (b)) | §5.8 *Entries defined by text*; §6 row removed; cells 2.17, 7.20, 9.12 [U3]; §8 tokenize counter; §14 (`TokenizeText` equals `Tokenize`) | U1, U3 |

**The executed run** (`reviews/fold-rev1p18-probes/`). `fold_refund_model.py` and `restrike_model.py` are byte
copies of `reviews/fold-rev1p17-probes/`; `drop_model.py` adds drops and drop marks under §5.6's rule and compares
the rev 1.17 rule (with drops) against the approved rule at every step. `drop_model.out`: seeds 7, 11 and 23, 20,000
runs each, about 136,000 requests, 51,000 nested, 27,000 waits and 38,000 drops (29,000 deferred) per seed, releases
in random and in decision order: **divergences 0, ledger over C 0, physical over C 0, parent faults 0, pinned
entries released 0**. Mutant m1 (a drop releases a pinned entry): about 7,700 pinned releases and divergences per
seed. Mutant m2 (a drop refunded at its decision): about 10,000 physical overdraws per seed. One model limit, found by
the run and named in its docstring: a request that re-creates a key whose earlier instance is still a ghost overwrites
that ghost's charge in the model, so such requests (11–20 per seed) are skipped. The same keying is in the rev 1.17
model, whose runs showed 0 divergences; whether its runs reached that shape is not checked here.

**Does anything need the principal?** PC-14's three open points, and PC-15's 1.13.0 row, each with a recommendation
(§12). Nothing else is raised: Q1–Q9 are folded as accepted, and the API additions change no existing promise.

### 15.19 Round 19 (rev 1.18.1): the scoped re-check of rev 1.18 and the conductor's rulings

Inputs: `reviews/plan-review-rev1p18.md` (TEMPER, F1–F16) and `reviews/coverage-audit-rev1p18.md` (gaps named, G1–G20,
N1–N6), and the conductor's rulings of 2026-10-20 on rev 1.18's six open points and on the readings the seats left
open. The fold note is `reviews/fold-rev1p18p1.md`; the executed re-run is `reviews/fold-rev1p18p1-probes/`. Every
finding is folded; none is rejected.

| Finding | Rev 1.18.1 disposition | Dependent gate |
|---|---|---|
| F1, G13 (a): the failed-release premise is false (both release verbs consume the handle on a failing status, the GPU's except on `SSLM_BUSY`); 5.14's seam certifies the defect; the rule ships at U2 with its only cell at U6 | §5.4 *What a release's status means* (applied unless GPU `SSLM_BUSY`; only `SSLM_BUSY` re-issued); the device-loss paragraph kept for skipped jobs; 5.14's beside-leg withdrawn; new 5.15 [U2, U6 GPU legs] with a seam that consumes the handle; §8 seam; §9 U2 (the draft's V12 code, Wizard `e7811590e`, needs the builder's fix); PC-15 | U1, U2 |
| F2, G1: the post-M1 reference trigger fires once; post-M1 commits have no stated home | §8: post-M1 branch cut from `main`, `<line> = origin/main`, must-reject (5) and must-accept (5); §9 M1 (cut the branch), new M2 row, E5; R13 | M1 |
| F3: E5's row keeps rev 1.13's `v1.13.0`/`a5f3ad1` sentences | §9 E5 rewritten for the post-M1 line and 1.13.1; M2 before the tag | E5 |
| F4, N6 (a): 1.15.0 carries this plan's code; the label check runs only at E5; the fast-decode line's pre-bumped version | §0 *Releases* 1.15.0 bullet; R11 (7.22 at the 1.15.0 release too); R14; cross-plan note in the fold record | M1; the 1.15.0 release |
| F5, G3: relabel in two files only; no instrument; today's labels read "(1.15.0)" | §4 preamble lists all files; §9 M1 relabels every file; new 7.22 [M1, E5, 1.15.0] with §8 registration | M1 |
| F6, G7: per-entry or whole-set failure | §5.8 per-entry rule (the conductor's ruling); GPU-empty entries "not built"; 2.17 and 3.11 aligned | U3 |
| F7, G15: K lanes leave phase (c)'s 1.0.0 calls no lane | 8.10 (c): K + 1 lanes, `MaxConcurrentCalls ≥ K + 1`, `BlockCount ≥ K + 1`, margins re-derived; §14 settled | U1 |
| F8, G10: "later" undefined across hash completions | §5.6 *What "later" means*: issue order, a game-thread sequence number, applied in issue order; new 3.12; §8 hash-completion hold | U1, U2 |
| F9: an attach to a suppressed in-flight write-back | §5.6: an attaching later request clears the mark and restores write-back; 3.10 leg (i) | U2 |
| F10, G20: a re-created key while the earlier instance is pending has no cell | new 8.17 [U2; U6 twin]; §5.6's model-limit sentence points to it | U2, U6 |
| F11: PC-14 (b) is not open | §3.8, §0, §12 PC-14: fixed by PC-7 | — |
| F12: M1 is not cloud-only; the coverage-floor pin is missing | §9 M1 (cloud + box; `branch_coverage_floors.json` step); §11 | M1 |
| F13: "four files"; "1.13.1's snapshot" | §3.8 (21 files, four library sources); §8 (`2e12a6e`, a commit on `d697927`) | — |
| F14, G6: `TokenizeText` read at source; two `sslm_tokenize` sites | §5.8; §14 settled; 7.20 and §8 counter cover both sites, fixture on the set's model | U3 |
| F15: the drop extends the shared residency core | §5.6 *The shared core*; cross-plan note in the fold record | U2 |
| F16, G11: `Pending` never expires; `Cancelled` unreadable | §5.6 reading (b): resolved ops keep their result across re-`Configure()` until the ring drops them; `Pending` never expires; 2.16, 3.10 leg (iii) | U2 |
| G2: M1 has no value observable or root check | §9 M1 receipts: version values, producer 1.13.1 through 7.12's parser, root count, the 1.14.0 mutant | M1 |
| G4: 6.8 lacks its kind-0 legs | 6.8 (test author adds two legs and the all-kinds mutant) | F2 box run |
| G5: GPU at-rest rule and derived state uncovered | new 11.8 and 8.16 [F2]; §4 items 1–2 | F2 box run |
| G8: text-entry staleness and nested legs | §5.8 (model hash takes the `TextChanged` path); 2.17 legs (construction, model-hash leg and mutant, nested fixture property, the no-tokenizer artifact) | U3 |
| G9: warm tier order, waiting, coalescing, `PoolExhausted` | 7.19 legs; 5.12 | U2 |
| G12: small drop and op-result legs | 2.16 (ring pair), 3.10 (`Pending`/`Deleted`), 5.12 (`NotRequested`, cold key with a file), 7.18 (waiting key), 8.14 legacy drop leg, 7.8 extended | U2 |
| G13 (b), (c): U6 unpinned-parent drop; nested warm pin | 8.14 U6 legs | U6 |
| G14: 9.11's foreign-commit case under the one-row table; 9.9's wording | 9.11 rewritten (1.13.1 row; foreign commit `9e235965`; 1.13.0 refused); 9.9 rewritten on `d697927`; (b) moot with the row dropped | E5, U2 |
| G16: 10.7 under-asserts | 10.7 (cooked warms with prefill 0; resident bytes fall; dropped specs re-requested `Cooked`) | U4 |
| G17: no Blueprint cell | new 7.23 [U2] | U2 |
| G18: unnamed mutants | 1.10, 2.16, 7.19 (LRU), 8.15 (A pre-charge) | U1 |
| G19: drop-model evidence reads stronger than it is | §5.6 reworded; m1 re-run past divergence (537, 642, 612 parent faults; 77, 75, 102 overdraws), sections 1–3 and m2 reproduced exactly | — |
| N1: §5.9 rows by step | 7.9 | — |
| N2: 1.4's in-flight clause vacuous for CPU builds | 1.4 note | — |
| N3: 7.21's refused-lease leg | 7.21 | U2 |
| N4: smaller edges | 2.17 [U6] `FindEntrySpec` parent; 3.11 counts prefill, freeze and save; 5.13 coalesced leg; §8 seam must-rejects | U1 |
| N5: mark-clear at `Configure()` is an equivalent mutant | §5.6, recorded | — |
| Conductor's rulings | PC-14 (a) early with the post-M1 branch; (b) settled by PC-7; (c) labelled, every file, 7.22 at E5 and 1.15.0; PC-15 Q8 row dropped; Q5 build hold with K + 1 lanes; Q6 the release-status rule; G7 per entry; G10 issue order | §12 |

**Does anything need the principal?** No PC is raised. The conductor's rulings settle PC-14 (a) and (c) and PC-15
Q8; the principal may overrule any of them in one line (§12).

### 15.20 Round 20 (rev 1.18.2): the strike on rev 1.18.1

Inputs: `reviews/strike-rev1p18p1.md` (FRACTURED: Fracture 1, Fracture 2 and lesser planes 1–8, with probes in
`reviews/strike-rev1p18p1-probes/`) and the conductor's rulings on each. The fold note is `reviews/fold-rev1p18p2.md`.
The executed re-run is `reviews/fold-rev1p18p2-probes/`: the strike's two probes, copied, with the rev 1.18.2 rule
added, and their original output reproduced exactly. Every finding is folded, and none is rejected. One fold goes
beyond its ruling's words, the pass bound in Fracture 1's rule, and its row says why.

| Finding | Rev 1.18.2 disposition | Dependent gate |
|---|---|---|
| Fracture 1: a deferred mark is applied while a request issued after the drop has a pending hash, so that request rebuilds (18 of 24 model cells; every cell with the hash held; all six triggers) | §5.6 *When a mark is applied*: no mark is applied while an op issued after its drop, and before the pass that finds the entry resident and unpinned, has a pending hash. The pass order is stated: completions, then mark applications queued in issue order, then the queue drained. §5.6's mark and clear bullets point to it. New 3.13 [U2; T4 at U6] covers all six triggers, with the hash held and released, plus an after-the-pass leg and a steady-caller leg. 7.18 gains the same-frame return. Executed: 12 of 12 hold. Marks as a stage before the ops hold 0 of 12. Each per-trigger mutant is red on its own trigger only, which is why all six are covered. *Beyond the ruling:* the words "any op issued after that drop" never apply a mark under a caller that issues a request every frame (40 of 40 frames), so the wait is bounded by the pass. The bound changes no casebook cell. It decides one case differently: a request for the key issued after that pass, while the mark still waits on an earlier op's hash, is rebuilt rather than served from memory (probe section 3b). The ruling's scenario does not contain that case. One line restores the unbounded words | U1, U2 |
| Fracture 2: a `MemoryAndDisk` drop leaves a file in s3 (a waiter), s5 (a save not yet written) and s8 (a `PoolExhausted` retry wait) | §5.6 disk part: covers "in flight or waiting", the same as the memory part; write-back is the save plus the file write; a retry wait counts as in flight; F9's attach covers waiting too. 3.10 (iv) s3 (the strike's keeper), (v) s5, (vi) s8, each with its mutant. Census: 0 of 7 states leave a file | U2 |
| Lesser plane 1: ops can read `Pending` forever across a re-`Configure()` | §5.6 *Generations and teardown*: each `Pending` state has a stated result. A warm in any state is `Cancelled`. A drop with its hash pending is `Cancelled` with `NotRequested`. A pending delete is drained and resolves with its result. §5.4 teardown: deletes are drained, never cancelled. 1.7 gains a leg for each state | U2 |
| Lesser plane 2: the delete is not serialized with an in-flight disk read (on Windows it fails) | §5.6 disk part and §5.5 row: reads are on the key's existing serialization, so the read finishes and then the delete runs. 3.10 (vii). §8: the pool-thread hold covers reads, plus a per-key overlap counter | U2 |
| Lesser plane 3: GPU `SSLM_BUSY` on a prefix release is unreachable without a seam | Read at source (`gpu_1p0.cpp:2272-2275`, `:4497-4510`): `SSLM_BUSY` needs the prefix's own sequence `Submitted`, which the cache never meets, so the branch is defensive (§5.4). The seam is named: §8's busy-release substitution, which returns `SSLM_BUSY` without calling the release, with its own must-reject. 5.15's [U6] leg uses it, and "driven by a Submitted sequence" is withdrawn | U6 |
| Lesser plane 4: a cross-context GPU release and a consumed underflow share one status | Recorded as a known limit: §5.4 bullet and R15 | — |
| Lesser plane 5: M2 under-scoped; the tag not pinned to a commit | §9 M2: test author and debunker added; cloud + box + work machine; re-runs 7.17, 9.9, 9.10 on both GPUs and 7.22; the tag's commit equals the recorded squash SHA. E5's exit says the tag goes on M2's squash commit. §11 M2 row; §0 | E5, M2 |
| Lesser plane 6: §8's "no ancestry joins the branch to `main`" is false | §8 corrected: the merge-base is `a5f3ad1`, through `6426a72` on the branch and `0d1c378` on `main` (read in git). M1 unaffected | — |
| Lesser plane 7: `FindEntrySpec` cannot return the parent's spec | §5.8: at U6 it returns `OutParentEntry`, the parent entry's name, and leaves `Out.Parent` invalid. 2.17's [U6] leg replaced | U6 |
| Lesser plane 8: the set's model is checked against the asset only | §5.8 *The set's model against the subsystem's*: refuse, the simpler option (re-tokenizing at runtime would break "the runtime never tokenizes"). `RegisterPrefixSet` and `FindEntrySpec` (which now takes the subsystem) refuse with `PrefixSetModelMismatch`, per set, for both input kinds. A re-`Configure()` to another model unregisters the set. §5.9 row; new 2.18 [U3]; §9 U3; PC-16 | U3 |

**Does an acceptance form change?** Yes, in four places, all in the plugin's residency and cooked-set API. No engine
form, no `SPX1` form and no promise sentence changes.
- A request issued after a drop and before the pass that frees the entry is now served from memory whatever its
  hash timing. Rev 1.18.1 rebuilt it when its hash was late.
- A `MemoryAndDisk` drop now leaves no file behind a waiter, a pending save or a retry wait. Rev 1.18.1 wrote one.
- No op reads `Pending` after a re-`Configure()`.
- A set whose model is not the subsystem's is refused, where rev 1.18.1 registered and served it.

The fold note lists the cells whose accepted outcomes move.

**Does anything need the principal?** No PC is raised. PC-16's row records the refusal. The conductor may strike
the pass bound in one line, and the cost is §5.6's steady-caller case.

### 15.21 Round 21 (rev 1.18.3): the re-check of rev 1.18.2

Input: the *Re-check* section of `reviews/strike-rev1p18p1.md` (HOLDS, minor findings m1–m3; the adversary's own
implementation is `strike-rev1p18p1-probes/recheck_queued.py`, 12 of 12), and the conductor's message keeping the
pass bound. Every finding is folded as text, and none is rejected. No rule, cell outcome or acceptance form changes.

| Finding | Rev 1.18.3 disposition | Dependent gate |
|---|---|---|
| m1: §5.8 gives `FindEntrySpec`'s signature in three pieces | §5.8 states one signature: `FindEntrySpec(const USuperSLMModel* ConfiguredModel, FName Entry, FSuperSLMPrefixSpec& Out, FName& OutParentEntry) const`. The caller passes its subsystem's `GetConfiguredModel()`, which both the CPU and GPU subsystems have (read in the plugin's public headers), so one signature serves both backends. `OutParentEntry` is `NAME_None` before U6. The model-check bullet and 2.18 now name that parameter; §9 U3 | U3 |
| m2: M2's exit checks a tag that does not exist yet | The check moved to the tag step (E5's exit): `git rev-parse v1.14.0^{commit}` equals the squash SHA that M2 records. M2's exit ends at recording it | E5's tag step |
| m3: 7.18's "evicted at that apply pass" no longer matches the rule | 7.18 now says the next apply pass, once the hashes of ops issued before it resolve, which in this fixture is the same pass | U2 |
| The pass bound | Kept, on the conductor's word | — |

### 15.22 Round 22 (rev 1.18.4): erratum from the E4, E5 and U2 suite audit

Inputs:
- `build/suite-audit-1p14-e4-e5-u2.md`, the suite audit (gaps named; the findings routed to the planner);
- the readings in `build/red-suite-1p14-e4-e5-u2.md`, the red-suite record;
- code review round 1's FIX for frozen GPU prefixes.

The fold is text only, on the approved rev 1.18.3, and the approval stands. No promise, PC or rule of §5 changes.
The audit's findings routed to others (G1–G3, G5–G7, N3–N5, reading 14) are not this plan's text and are not
folded here. G4's reason code and G8's F10–F13 text were the planner's parts of those findings, and are folded.

| Finding | Rev 1.18.4 disposition | Dependent gate |
|---|---|---|
| G9: 6.6's mutant names a parallel K/V path that does not exist | 6.6: the K/V mutant is withdrawn. The hook splits only logits rows and one-row projections (`parallel_for.h:10-18`). The new mutant, the last matvec task's row range ended one row short, is visible to the serial-against-parallel equality on both legs. K/V addressing at rule B stays with 10.1, 10.2 and the leg's re-save identity | E4 |
| G10, reading 5, N2: stale E4 site records | §7 *Geometry*: it applies to 10.1–10.3, and `:181` is 10.3's (`Timing103`). 4.3 runs at the pin with no real fixture, and is struck from the `fx.pinned` list and from §9 E4's real-artifact run. `:272` is `worst_pool`, which is rule-B through `CapPages()`. `Timing103`'s constant `pool(300)` (`:184`) is named as not a site. §15.13's F2 row and §15.15's R1 row are historical and stay as written | E4 |
| G11: 7.17's locator misses `docs/api.md:518` | 7.17: an "ordinal N" with no name on its line pairs with the next line's first backticked name, and is otherwise unpaired and refused. The producer-mismatch floor is 2, with two new must-rejects. Reading 6 is corrected beside it: the check is green when written, and commissioning starts from the mutants | E5 |
| Reading 7, N6: 7.22's unit | 7.22: the unit is the paragraph or the comment, as built, ruled before M1's relabel. "The grep" is defined as the sweep that was built, and the lexical locator's escape is recorded as a known limit | M1 |
| Reading 2, N1: 9.9's bound cases | 9.9: outcome form, 64 step codes equal to the capture's. The bound cases' continuation grades at most one token, and the re-save carries them | E5 |
| Reading 1: 9.9 "mid-turn" | Closed: mid-token, as the strike probe defines it (`layer_index = 1`) | — |
| Code review round 1, the FIX | §4 item 2: freeze sets the snapshot state and `bind_eligible` from `L`, matching restore. §4 item 7 notes that freeze now carries its outcome. 6.8 gains legs A, B and D (built at `50bab8cd`), with the mutant's re-anchor. 8.16 also carries leg D, through its `L = 0` twin's construction, and gains one mutant | F2 |
| G4, reading 10: the warm and drop rows lack 5.4's form; a refused drop delete has no reason | §5.9 *Reason codes*. Warm rows use the request's reasons. New `DropDeleteFailed` and `PrefixSetModelMismatch`, appended before `Count`. A deferred drop is not a fallback (drop diagnostics only). 5.12 now grades the line's reason and the counter delta, with two mutants | U2 |
| G8: the carried T-3060 F10–F13 E4 portions are unverified | Found in the Wizard records (the T-3060 case file and the T-3070 recheck, identical on `develop` and `origin/claude/project-thread-wt2dpx`). Folded as §7 *The carried T-3060 F10–F13 portions at E4*: F10 and A3 go to 4.3, F11 and A4 to 10.1 and 10.2, F12 to 10.3, F13 to 6.6. The authored state was reconciled at `8788a8ea`, and one item is left for the test author (the `Field` helper) | E4 |
| Reading 6 (with G11): 7.17's red-first premise is stale | 7.17: the check is green when written, and commissioning starts from the constructed mutants. Kept on the conductor's word | E5 |

**Does an acceptance form change?** Yes, in three places, and each is what its finding requires.
- **7.17** now rejects a docs file it accepted before: an ordinal statement split across two lines and edited, or
  one left unpaired.
- **9.9's four bound cases** accept outcome codes equal to the capture's, where the text said 64 equal tokens. A
  dead-ended schema cannot produce 64 tokens, so the old form could not be met.
- **5.12** now requires each warm and drop row's log line to carry its reason, and that reason's counter to rise by
  exactly one. Before, a refused delete passed on any single log line, and the warm rows on the result alone. This
  adds two plugin fallback reasons.

The F10–F13 fold changes no form: it records what E4 already carries.

Nothing else changes what it accepts:
- 6.6's mutant changes, but the cell's equality does not.
- 7.22's unit is the one that was built.
- 8.16 fixes its twin's history, and its accepted outcome is unchanged.
- §4 item 2 states the code review's FIX, and the outcome items 2 and 7 promise is unchanged.

**Reconciled against git:** the FIX itself is not on `claude/project-thread-wt2dpx` at its tip, `8788a8ea`. There,
6.8's reference fix is a scratch build only (the red suite's record). The plan states the rule the builder lands.

### 15.23 Round 23 (rev 1.18.5): clarifications from the U2 code review

Input: `code-review/review-1p14-u2-round1.md` (verdict FIX; branch `claude/project-thread-wt2dpx` at `79163cd69`).
Four approved rules are stated for the path the product takes. These are clarifications of the approved promises,
not design changes, and the approval of rev 1.18.3 stands. Findings 5, 6, 8 and 9 need no plan text. Findings 5
and 6 are code against rules the plan already states (§5.8's re-`Configure()` unregistering; §5.9's `DropDeleteFailed`
line with the file API's error). Findings 8 and 9 are observations.

| Finding | Rev 1.18.5 disposition | Dependent gate |
|---|---|---|
| F1 and F2: a drop's delete waiting on a write-back's save is carried into the next `Configure()`, or never run at `Deinitialize()`. The cells drive `BeginTeardown()`, which the CPU host never calls | §5.4 *Re-`Configure()` and teardown* and §5.6 *Generations and teardown*: the drain is a property of the host's abandon-then-pool-destroy path. Such a delete is drained under the old configuration before the pool is destroyed, and never carried into the next `Configure()`. 3.10 (iii) gains three runs on that path (disk tier off, another directory, `Deinitialize()`) with the carry mutant | U2 |
| F3: a GPU release the submission thread skipped is delivered as a returned `Failed`, so it is refunded while still mapped and dropped from the live-handle set | §5.4 *a skipped release is not a release that returned*: it stays pending, refunds nothing, logs no `ReleaseFailed`, is not re-issued, and stays in the live-handle set until teardown's sweep. The host delivers it as skipped. 5.14 gains the sweep's count and an unresponsive-thread leg, with the delivered-as-`Failed` mutant | U6 |
| F4: a warm whose build fails resolves `Cancelled` | §5.6: a new op result `Failed`, with `Out.Reason` the request's §5.9 reason (`BuildFailed`), appended last in the C++ enum. `Cancelled` stays re-`Configure()` only. §5.9 row; 5.12 leg | U2 |
| F7: the hit's `TouchDiskFile` is outside the core's per-key disk serialization, and Windows `SetTimeStamp` opens without delete sharing | §5.7 and §5.5's row: the touch joins the key's disk serialization. §8's per-key overlap counter covers it. 3.10 (vii) gains a held-touch run | U2 |

**Does an acceptance form change?**
- **F4** adds one API value, `Failed`, and the `Out.Reason` field, both additive. A warm's failed build now has an
  accepted result where the plan had none; the code gave `Cancelled`. 7.23's Blueprint parity covers the new field,
  since it compares every output field.
- **F1, F2, F3 and F7** each add cell legs whose expected outcome is the one the approved rule already gave. They
  make no form accept or reject differently from what the promise says.

### 15.24 Round 24 (rev 1.18.6): 8.14's reachable construction; U6's held releases

Source: `code-review/review-box-fixes-2026-10-21.md` (verdict SHIP), on Wizard `afe45536d`. Errata only.

| Item | Rev 1.18.6 disposition | Dependent gate |
|---|---|---|
| Plan note: 8.14 says C is "filled exactly" with "free charge 0", which builds cannot reach. A build is charged `min(ceil(L/B) + 1, P_blk)` until it freezes, so the cells settle with one page free and size each request one page larger | 8.14 states the reachable construction once, ahead of its legs. Each request takes the free page first and then claims the refunds the leg names. The window still has C full with E's `k` carried, the overdraw is still `k − f = 1`, and form 3's zero slack still holds. The review's mutant sweep kills the three named mutants on the cells as built | U2 |
| The builder's change: a cache request's lease cancelled at a re-`Configure()` stays readable as `Cancelled` (queued leases now, running builds and restores being added) | §5.6 *Generations and teardown* states the limit. The kept set is capped at 1,024 entries, oldest dropped first, and a dropped lease reads `Unknown` (callers see `Invalid`). Cell 1.7's `Cancelled` reading holds within that cap | U2 |
| Finding 3: the GPU host's `SetHoldReleases` only sets the flag and never sends held releases on | §9's U6 row gains it as an entry condition, fixed in the U6 red commit before any gated GPU cell runs, as the CPU host now does | U6 |

**Does an acceptance form change?** No. 8.14's readings, outcomes and mutants are unchanged. Only the stated
construction now matches the cells. The U6 item is an entry condition on code, not a cell change. Findings 1, 2 and 4
are code follow-ups, and the plan does not change for them.

### 15.25 Round 25 (rev 1.18.7): plugin 1.3.0 carries the cache and the skip-ahead options

Source: the principal's ruling, 2026-10-21 22:20Z, relayed by the conductor: "it should have both, the work is done".

| Item | Rev 1.18.7 disposition | Dependent gate |
|---|---|---|
| Plugin 1.3.0's contents and order | Plugin 1.3.0 carries the CPU prefix cache **and** the fast-decode plan's two schema skip-ahead options, canonical and exact (its P1 and P1-X). The order is: engine 1.14.0, then engine 1.15.0, then plugin 1.3.0, which re-vendors 1.15.0 with `SUPERSLM_WITH_SCHEMA_JUMP_FORWARD` set to 1. Engine 1.16.0 and plugin 1.4.0 for the GPU follow. The release list, the step line, PC-1 and §9's U5 row are updated: U5 now waits on engine 1.15.0 and on P1 and P1-X being merged, and its commit names both features. Plugin 1.3.0 was the cache alone, vendoring 1.14.0; U0–U4 still build on 1.14.0, and the cache cells re-run on 1.15.0 before the tag | U5 |

**Does an acceptance form change?** No cell changes. U5's entry condition gains engine 1.15.0 and the fast-decode
slices, and the cache cells re-run on the re-vendored engine before the tag. The principal's approval stands: this is
his own ruling.

### 15.26 U4 build erratum (2026-10-22): the packaged prefill count and the restore's per-MiB term

Source: the U4 build, on Wizard `claude/project-thread-wt2dpx` after `576579a24`. The builder records it; no design
changes, and the cells keep their readings.

| Item | Disposition | Dependent gate |
|---|---|---|
| 10.5 and 10.7 read "no prefix prefill" from `sslm_test_only_prefix_prefill_call_count`, which exists only in an engine built with its test hooks. A packaged Development or Shipping game has no such counter | The plugin counts prefix prefills itself. `FSuperSLMPrefixCacheStats` gains `PrefixPrefillCalls` (one per `sslm_prefix_prefill` call, one per chunk) and `PrefixPrefillBuilds` (builds that made at least one). Both are counted at the lane after the call returns, on the CPU and GPU hosts, in every build configuration, and cumulative since the cache host was created, so cells read them as differences. On the CPU the host outlives a re-`Configure()`, so the counts carry across it; on the GPU the host belongs to its configuration, so a GPU re-`Configure()` starts both from 0. They count the cache's builds only: a `CreatePrefix()` prefill, which the engine's counter also saw, is not counted. 10.5's and 10.7's packaged legs read these. The editor cells keep the engine's counter. The example's check confirms the counter on its own reference pass (10 recomputes read as 10 builds) before any leg reads 0, and the disk legs confirm it in their own run (`DiskWrite`'s 10 recomputes read as 10 builds before `DiskRead` reads 0) | U4 |
| §5.5's "a per-MiB cost from the calibration command's existing per-unit model" names no field or command | A cache restore is priced `RestoreCostMs + RestoreCostPerMiBMs × (payload bytes / 2^20)` (`Private/Core/SuperSLMRestorePricing.h`, host-tested). `RestoreCostMs` keeps its meaning: the fixed cost, which the estimate's existing restore arm measures on an empty blob. The new `RestoreCostPerMiBMs` is **not one of the thirteen cost fields**. It is an optional field on `FSuperSLMRuntimeConfig` and on the resolved configuration: the caller's value, else the matching saved estimate's when that profile is used, else 0 (the flat price before U4). `Configure()` refuses a non-finite or negative value. Only the prefix cache's restores use it; sequence restores stay flat | U4 |
| Which command measures it, and where it is saved | `SuperSLM.EstimateCosts`, the shipped calibration command (editor and packaged; `SuperSLM.CalibrateCosts` is the older, test-only command). A new `restore_size` arm, apart from the 22 arm rows and the calls CSV, restores the blob saved at each depth the other arms use, leaving out any blob under 1 MiB (`kRestoreFitMinBytes`; depth 1's blob is a few dozen KiB, so its median noise over its size would set the slope). The suggested slope is fitted from the smallest and the largest remaining readings, `max(0, (median_largest − median_smallest) / (MiB_largest − MiB_smallest))`: the incremental cost per MiB across the measured range, floored at 0. With one remaining size it is the slope of that size's slowest reading through the fixed cost, `max(0, (median − RestoreCostMs) / MiB)`. *Code review S2 (2026-10-22):* this replaces the first draft's upper envelope over every depth, `max over depths of max(0, (median − RestoreCostMs) / MiB)`, which never underpriced a measured size but let the smallest blob's noise set a slope several times the large blobs' and inflate K for every large restore. The two-point fit can underprice a middle size whose median sits above the line; a late result is still counted as a hitch (§5.5). It is reported on `restore_size` and `suggested_per_mib` lines and written by `--write-ini` as `RestoreCostPerMiBMs=` after the thirteen keys of each mode's profile in `Saved/SuperSLM/CalibratedCosts.ini`. A profile without the key reads 0 | U4 |

**Does an acceptance form change?** No. 10.5's and 10.7's "no prefix prefill" reads the plugin's counter of the
cache's prefills in a packaged build instead of the engine's counter of every prefix prefill. The legs make no
`CreatePrefix()` prefill, so the reading is the same (0 in the window). R5's disposition ("priced by the per-MiB
cost (U4)") is now a named field and command. 10.7 and §5.9's one-job build leg are legs of the U4 check
(`Residency`, `OneJobBuild`). *Superseded the same day (the `Text`-entries build, below):* the first U4 tip did not
have U3's `Text` entries, and this record said 10.7 would run on `Tokens` entries only. They are now built, and the
`Residency` leg runs on a mixed `Text` + `Tokens` set as 10.7 asks.

**The `Text`-entries build (2026-10-22, after `d72a2e920`).** §5.8 *Entries defined by text* (approved at rev 1.18.3,
in plugin 1.3.0's scope) was missing from the branch: the asset had no `Input`, `Text` or stored hashes, the build
had no tokenize pass, the cook no `TextChanged`, the runtime no `FindEntrySpec` and no skip, and no tokenize counter
existed. Cells 2.17, 2.18, 7.20 and 9.12 had no tests. It is a build gap, not a waiver, and it is now built; the
cells were written red first (host twins in `test_prefix_set_text.cpp`, 7 of 8 red against stubs; the editor cells
in `SuperSLMPrefixSetTextTests.cpp`, uncompiled until the box). The builder's readings, where §5.8 names no detail:

| Item | Disposition | Dependent gate |
|---|---|---|
| The tokenize pass's failures by name | `TokenizeFailed` (new: the tokenizer's error, e.g. an artifact with no tokenizer section), `EmptyTokens` (empty text, or no ids) and `LongerThanContextCap` (ids over the cap), each a CPU build finding naming the entry. Empty text makes no tokenize call | U3 |
| Per-entry failure (rev 1.18.1) in the synchronous build | Applies at the tokenize pass and to a failed prefill, freeze or save (`BuildFailed`, `UnknownSchema`): the entry keeps its own previous ids, hashes and bytes from the same producer's previous build, and the others are stored in the one replacement. Before this build such an entry was left empty. An entry the manifest refuses (a nested entry before U6, a malformed `Tokens` entry) is still left empty, as before | U3 |
| Staleness, one rule for cook, registration and `FindEntrySpec` | A `Text` entry is current when it has ids, its stored text hash is the SHA-256 of its text, and its stored model hash is the set's model's artifact hash. Otherwise it is `TextChanged`, including a `Text` entry never built or cleared by an edit (one `TextChanged` finding at cook, not `NotBuilt` or `EmptyTokens`). A `TextChanged` entry's bytes are neither checked against a build nor reported `Orphaned`, because the cook clears them | U3 |
| The cooked package's empty entry | `USuperSLMPrefixSet::Serialize`, when the archive is saving and cooking, writes a copy of `Entries` and `Builds` with each `TextChanged` entry's ids and bytes cleared (`SuperSLMPrefixSetTools::CookedImage`) and swaps the object's own arrays back after, so the object and the source package are unchanged. In a cooked game such an entry reads `NoStoredIds` from `FindEntrySpec`; uncooked (PIE) it reads `TextChanged` | U3 |
| `FindEntrySpec`'s reason | The member returns `bool`, as §5.8 states, and logs the reason; `SuperSLMPrefixSetTools::FindEntrySpec` also returns it by name: `PrefixSetModelMismatch` (checked first; a missing or unreadable model on either side is a mismatch), `UnknownEntry`, `NoStoredIds`, `TextChanged` | U3 |
| The change hooks | `PostEditChangeProperty`: an edit to an entry's `Text` or `Input` clears that entry's hashes and, for a `Text` entry, its ids and its bytes in every build; a change of `Model` does so for every `Text` entry. A `Tokens` entry keeps its authored ids and bytes | U3 |
| 7.20's counter | `SuperSLMTokenizeTestSeam` (`WITH_DEV_AUTOMATION_TESTS`): one count per `sslm_tokenize` call at each site, `TokenizeText` and `USuperSLMSubsystem::Tokenize`, the retry after `SSLM_BUFFER_TOO_SMALL` included | U3 |
| 9.12's old asset | No asset saved by the U3 class is pinned in the tree. The cell reads one from `SUPERSLM_PRECACHE_U3_SET_ASSET`, which the U4 box run keeps from the tree it replaces, and fails by name without it | U3 |
| The reflected part | A fifth box patch, `u-plugin-cloud-2026-10-22-u3text/reflected-prefix-set-text.patch`, applied after `reflected-prefix-set.patch`: the enum, the four fields, `FindEntrySpec`, `Serialize` and `PostEditChangeProperty`, each forwarding to `SuperSLMPrefixSetTools` | U3, U4 |
| 10.7's `Text` half | The U4 box run makes and builds a second set, `U4MixedSet` (5 `Text` entries, 5 `Tokens` entries), and runs the `Residency` leg on it with every spec from `FindEntrySpec`. The drop leg's fall is checked against the 10 entries' resident bytes after the reference pass (their lengths now differ), with none left resident; the leg ends by checking each `Text` entry's ids against `Tokenize` of its text | U4 |
| The background build (§5.8, cell 3.11) | *Built 2026-10-22, in its own commit after the `Text` entries; first recorded here as not built.* The build is split into three phases: `BeginBuildCpu` on the game thread (identity, the tokenize pass, the manifest, the model's mapping, the pool and workspace); `RunBuildEntries` on one worker thread (prefill, freeze and save per entry, with a cancel checked between entries); and `FinishBuildCpu` on the game thread (it releases the mapping, then makes the one replacement). The commandlet's `BuildCpu` runs the three phases in a row, so its output is byte-equal to the action's. The builder's readings: the progress dialog is a persistent editor notification showing entries done of the total, with a Cancel button. A cancel takes effect between entries: the entry in progress finishes, and nothing is stored. A set or model edited while the build runs makes the build store nothing and report it refused (a fingerprint of the entries and the model is taken at Begin and checked at Finish). The cell's cancel-after-3 leg uses a test pause seam that holds the worker. The counter of prefill, freeze and save calls on the game thread is a `WITH_DEV_AUTOMATION_TESTS` seam. The cell is `SuperSLMPrefixSetBackgroundBuildTests.cpp`, written red first | U3 |

### 15.27 Round 27 (rev 1.18.8): U4's one-job build reading trips the slicing trigger

Source: the U4 packaged run, Wizard `claude/project-thread-wt2dpx` at `15a617f01`,
`Claude/Plans/precache/build/box-2026-10-22-u4/results/OneJobBuild.json`. The conductor asked for a planner ruling.

**The reading, exactly.**
- One CPU cache build of 2,000 tokens ran as one lane job of 32 prefill calls.
- It was planned at 2.261 ms with K = 1, so its committed delivery tick was 2.
- It took 49,889 ms of worker time, was delivered at tick 15,419, and the leg took 50.8 s in all.
- The hitch count read 2. Both counts come from this one job: the plugin counts a late delivery and a worker overrun
  separately. They are scheduler hitches (a job delivered after its committed tick), not render-frame stalls. The
  game thread did not block, but every job queued on that lane waited about 50 s.

| Item | Ruling | Dependent gate |
|---|---|---|
| 1. Does the reading trip the trigger? | **Yes.** §5.3 item 4 and cell 5.9 name the trigger as "a count above zero", and the count is 2. The one reading survives on the plan's own words. The cause is the one the trigger anticipated: a build priced as a begin (`PrefixBeginCostMs`) runs a whole prefill in one job, so the scheduler commits it 1 tick ahead and then holds the lane for 50 s | U4 (done) |
| 2. The smallest slicing change | §5.3 item 4: a CPU cache build becomes a begin job, then one prefill job per chunk planned and priced by `CreatePrefix`'s existing chunk planner and price (`DispatchPrefixPrefill`'s path). The last chunk's job freezes. The jobs run in order on one lane, with other jobs free to run between them. The build is one request throughout (charge, pins, coalescing, waiters, in flight), and the build hold stays in the first job, so 8.10 (c) is unchanged. A re-`Configure()` between chunks issues no further chunk, resolves `Cancelled`, and releases the unfrozen prefix once. The output is byte-equal to the one-job build's. **Rejected as the fix:** only re-pricing the one job (K from a per-token build price). It would clear the counter, but a 2,000-token build would still hold its lane for 50 s, and every generation on that lane would wait behind it; the trigger exists to stop that. No engine change: `sslm_prefix_prefill` already takes chunks | U5 |
| 2, cells | 5.9's build leg is re-graded: (a) sliced, every job priced within the slice; (b) the frozen bytes and an adopter's tokens equal a one-job build's; (c) a generation on the lane delivers between chunks; (d) hitches recorded, expected 0, and a non-zero count goes back to the calibration. 1.7 gains the between-chunks re-`Configure()` leg. Mutants: one job; a chunk priced as a begin; a freeze before the last chunk; a chunk after teardown or a leaked unfrozen prefix | U5 |
| 3. Plugin 1.3.0, or after? | **In scope for 1.3.0.** The recompute path is in 1.3.0: it is the fallback for any cache miss, not just a test path. Shipping it unsliced ships a known multi-second stall, about 25 ms per token, roughly 3 s for a 128-token prefix and 50 s for 2,000, on every generation that shares the lane. The scheduler is blind to it, because the job is priced at 2 ms. The fix reuses a planner the plugin already has, so it is small. It does not change what the principal approved: rev 1.18 approved the one-job build with slicing as the box refinement if this reading came back, and it came back | U5 |

**The loop: one capped round.** The builder slices, the test author writes the re-graded 5.9 leg (host twin, then box)
and 1.7's leg red first, and the code reviewer reviews once. Findings are folded as text; no strike or re-check is
added. The U5 box run then re-runs 5.9's build leg.

**Does an acceptance form change?** Yes, and it tightens. 5.9's build leg goes from "recorded, not graded" to graded
(a)–(c), with (d) recorded. 1.7 gains a leg. No promise changes, and no principal question is needed.

### 15.28 Round 28 (rev 1.18.9): the roots receipt and the shallow-history refusal

Source: the debunker's post-M1 record, `release-1p14/debunker-post-m1-2026-10-22.md`, findings 1 and 2. Its two
receipts, must-accept (5) and must-reject (5), PASS, and this round does not change them. One capped loop: the builder
fixes the fixture, the test author writes its cell red first, the code reviewer reviews once, and the debunker
re-takes the receipts below in a full clone.

| Item | Ruling | Dependent gate |
|---|---|---|
| 1. The five "parentless 2026-10-12 roots" in §8, §9 M1, M2, PC-14 and the squash message are a shallow clone's boundary (`.git/shallow`). On the remote each has parents, and `a5f3ad1` is an ancestor of `main`. So `git rev-list --max-parents=0 main` lists the five only in a shallow clone | **The receipt changes, and the design does not.** The roots check proved nothing about the squash. What M1 and M2 must show is that the line landed as one commit with no line history. The new receipt, taken in a full clone with `git rev-parse --is-shallow-repository` reading `false` (recorded): (a) the squash commit has exactly one parent, `main`'s pre-squash tip (`2e12a6e9` for M1; already PASS); (b) `git rev-list <pre-squash tip>..<squash>` lists the squash alone; (c) `git merge-base --is-ancestor <line tip> <squash>` fails (M1: `5e174457`; M2: the post-M1 branch tip). M1 re-takes (b) and (c) in a full clone, and M2 takes all three. The squash message's "parentless roots" sentence is a historical record and is not rewritten (`main` is not force-pushed); the next squash message (M2) does not repeat it. The §8, §9 and PC-14 texts carry a correction note. Rev 1.18.2 had already found that ancestry joins the branch to `main`. PC-14's choice of a squash is unaffected | M1 (re-take (b), (c)), M2 |
| 2. §8 says "a shallow history … records no merge-base, and the loader refuses". `write_merge_base.py` refuses only on a `git` error or malformed output, so it recorded a merge-base and graded in a shallow clone | **The plan's rule stands, and the fixture is fixed: a code fix, before 1.14.0.** In a shallow clone, `git merge-base` sees only commits above the boundary. It answers correctly when the true merge-base lies above the boundary, which is why the debunker's runs graded correctly. When it lies below, `git merge-base` can name the wrong commit, and the anchor gate's whole job is to know which `main` code the references were cut on. Refusing every shallow history is the simple sound rule, and its one cost is an unshallow fetch before grading. **Fix:** the fixture runs `git rev-parse --is-shallow-repository` first; on `true` (or on an error from that command) it records no merge-base and refuses with a named message that gives the remedy, `git fetch --unshallow`. **Why before 1.14.0:** E5's tag needs graded references, and a gate that grades where §8 says it refuses is the defect the anchor gate exists to prevent; the fix is a few lines. It lands on the post-M1 branch and reaches `main` in M2, before the tag. **Cell (new, E2's reference cells, cloud):** (i) a `git clone --depth 1` of a local full repository, configured: no merge-base file, the loader refuses with the shallow message, and the reference cell is red (a failure, never a skip); (ii) the same clone after `git fetch --unshallow`: a merge-base is recorded and the cell grades. Mutant: the shallow check removed, so (i) records a merge-base and grades, which turns the cell red. **Then:** the debunker re-runs must-accept (5) and must-reject (5) in an unshallowed clone; the same verdicts are expected, since the merge-base was above the boundary | M2 (the fixture fix and its cell), E5 |

**Does an acceptance form change?** Yes. M1's and M2's ancestry receipts replace the roots receipt; it was never a
valid form outside a shallow clone. One fixture cell is added, and the post-M1 receipts are re-taken in a full clone.
No promise, PC or design changes, and no principal question is needed.

### 15.29 Round 29 (rev 1.18.10): cell 5.9 (a) read as the chunk planner's rule

Source: the builder's grading of 5.9 (a) at Wizard `7f72e51e2`, relayed by the conductor. One capped loop: the code
reviewer checks the graded form once, and the U5 box run reads it.

| Item | Ruling | Dependent gate |
|---|---|---|
| 5.9 (a) says no chunk is "above the tick's slice". At the box's costs (≈ 15 ms per token batched as priced, more than 25 ms measured), one token already costs more than the share (16.6 × 0.7 ms), so the floor of 8 sets every chunk, priced above the share with K > 1. The literal reading cannot pass, and it contradicts the planner the plan named | **The builder's reading is right.** §5.3 item 4 says the chunks are planned and priced "exactly as `CreatePrefix`'s are", and `PlanPrefillChunk` returns the largest chunk priced within the share, but never less than `min(MinPrefillChunk, MaxChunk)`. So a chunk can exceed the share only at that floor, and the floor earns a larger K. "None above the slice" was the planner's own wording, not a separate rule. 5.9 (a) is restated: each chunk is the planner's, at the planner's price and K, and any chunk above the share is exactly the floor, `min(MinPrefillChunk, MaxChunk)` with `MaxChunk = min(MaxPrefillChunkBudget, tokens left)`, at least 1. The floor must be stated that way, so that a last chunk shorter than `MinPrefillChunk` is graded as the floor too. The mutants still die: one 2,000-token job is above the share and is not the floor; a chunk priced at `PrefixBeginCostMs` has the wrong price. The change still meets its aim: at 25 ms per token, an 8-token floor chunk holds its lane for about 200 ms, not 50 s | U5 |
| What the box costs imply for (d) | Not a change to (a). The measured per-token cost (more than 25 ms) is above the priced one (≈ 15 ms), so floor chunks may overrun their K and be counted as hitches. (d) already sends a non-zero count back to the calibration (§5.5), not to slicing, and that is where this goes: the batched price needs re-calibrating on the box | U5 |

**Does an acceptance form change?** 5.9 (a)'s wording is restated to the planner's rule. It is a correction: the
rule is the one §5.3 item 4 already names, and the mutants are unchanged. No code change is required beyond what
`7f72e51e2` grades.

**The code review's S1, folded in this round (the builder, on top of `7f72e51e2`).** A sliced build's chunk was
posted last in the tick, after `PlanNextJobs` and `PlanContinuations`, so a generation decoding without a break on
the build's lane (its continuations queue behind its running job) could hold the next chunk back for its whole
length. The lane now alternates. Once the build's previous job (its begin or its last chunk) has delivered, so its
next chunk waits, and the lane has taken one job since that previous job, the lane takes no other new job, and no
continuation, until the chunk is posted (`SuperSLMCore::LaneHeldForChunk`, read by `FLane::IsFreeForPlanning` and
`PlanContinuations`). So at most one other job runs between two chunks of a build. The rule is in the host twin,
which replicates the tick's order on one lane around it, and its mutant (the chunk dispatched last only) turns two
host tests red. 5.9 (c) gains a bound: at most one generation job between any two of the build's jobs. No acceptance
form loosens; (c) tightens. The review's S2 (the check's comment called the one-call reference "the one-job build")
is reworded: U4's one-job build was 32 calls of 64 tokens, and the reference is one call over the whole prefix,
valid because the engine's prefill is bit-identical at every chunk size and split.

### 15.30 Coverage re-audit erratum (2026-10-22): 11.1's `length < 0` exemption, 8.13's A22 legs, 5.10's mutant line

Source: `release-1p14/coverage-reaudit-post-m1-2026-10-22.md`, planner notes P1a, P2 and P3, on SuperSLM
`claude/precache-1p14-post-m1` at `e0fe1aea`. Wording only, with no rev bump, as in §15.26. Each cell's text now
carries the note in place.

| Item | Correction | Dependent gate |
|---|---|---|
| P1a: 11.1 asks for one check-removal mutant per clause of step 8, except the checked size arithmetic. `length < 0` has no input that kills its mutant (`build/post-m1-cloud-2026-10-22.md` §5) | `length < 0` joins the exemption. With it removed, a negative `L` reaches `kDomain` through `BlobSize`'s checked multiply (`runs · head_dim ≥ 2`), through `n ≠ required` (product 1), or else through the origin clause, because no origin satisfies `0 ≤ origin ≤ L < 0`. The status is the same and nothing is drawn. The clause stays in the reader for "L within the cap before any arithmetic" | E2 |
| P2: 8.13's rev 1.14 note says only the whole_reserve adopter in the contiguous pool discriminates A22 | The whole_reserve adopter in the page pool of a contiguous-pool prefix discriminates as well, through the bind clause: A22 removed → adopt revokes the bind (`sslm_abi.cpp:2862`) before the mapped-count refusal (`:2893`) → "schema bind after the call … got `SSLM_SCHEMA_BIND_REJECTED`" (`e2-ruleb-cells.md` §8 F1, reproduced on `e0fe1aea`). The budget legs alone stay non-discriminating. The revoke-before-refuse ordering goes to code review | E2 |
| P3: 5.10's mutant line says the underdraw-on-private mutant turns 5.10 red | It does not. The underdrawn holder is refused at the map with the same `SSLM_KV_POOL_EXHAUSTED` and releases everything. 2.6 kills it (with 6.3, 7.5 and 9.3), which meets T-3060 F8 (E3 cloud record R4; re-audit §3) | E3 |

**Does an acceptance form change?** No. The cells and their assertions are unchanged. 11.1's exemption list grows by
one equivalent clause; 8.13 records one more discriminating leg; 5.10's mutant line names the cell that kills its
mutant. The same round's test changes (G1, 5.7's 2-page zero-free confirmation; G2, the plan's step-4/5 swap in the
runner beside the renamed step-7/8 swap) are at `51bc3d6a`, and both swaps are killed in 5.7 (`post-m1-cloud` §5).

### 15.31 Round 31 (rev 1.18.11): cell 5.9 (d), chunk overruns and the cache-chunk price

Source: the U5 box readings relayed by the conductor (598 hitches before calibration, 600 after, over 250 floor
chunks of 8 tokens), and the builder's diagnosis at Wizard `f7d6825c7`. A box rerun with `hitch_breakdown` and
`chunk_price` is queued. One capped loop: the test author re-grades (d), the code reviewer reviews once, and the
queued rerun reads the new form.

| Item | Ruling | Dependent gate |
|---|---|---|
| 1. How (d) is defined | **The build's own chunk overruns, not the all-jobs counter.** (d) exists to catch a mispriced chunk (§15.27). The subsystem's hitch counter mixes causes and effects: it counts up to 2 per job, on every job on the lane. At about 300 ticks/s an overrunning chunk counts twice, and generation jobs that wait behind it add more. 598 and 600 therefore measure the lane's lateness, not the chunk's price. (d) reads each chunk job's price, K, worker ms and overrun from the ledger. The global counter and its breakdown by job kind are recorded beside it, and the counter itself is unchanged (§5.5, *What counts as a hitch*) | U5 |
| 2. Is "expected 0" achievable? | **No, as a grade.** A price is a calibrated median, so chunks above it are normal noise. Zero overruns would need a price above the slowest chunk, which over-prices every build. The grade becomes: after calibration, the median chunk's worker ms is at most its K × `TickBudgetMs`. The box reads about 200 ms per 8-token chunk against a price of about 120–200 ms, so that grade would fail today. **So the price must come from an arm that times what it prices.** Calibration's batched arm times `sslm_prefill` at depths spread across the cap. A cache chunk is `sslm_prefix_prefill` of the floor chunk, starting from depth 0. The required change is a `prefix_prefill` arm in `SuperSLM.EstimateCosts`: it builds a prefix from depth 0 in floor-sized chunks up to the arm's longest depth, times each chunk, and writes a price pair the cache's chunk pricing reads. When that pair is absent, the pricing falls back to the batched pair. The rerun's `chunk_price` breakdown decides only whether depth or chunk size explains the gap; the arm covers both | calibration (below) |
| 3. Does it block plugin 1.3.0? | **No. It is recorded at U5 and calibrated later.** The promise the plan makes is that a late result is counted and never hidden, and it holds. Slicing's aim also holds: a lane is held for about 200 ms per chunk, not 50 s. An under-price costs late deliveries during a cache build on a miss only, and they are counted. U5 grades (a)–(c), and runs (d) in its new form with the median grade recorded rather than graded. The `prefix_prefill` arm and (d)'s median grade land in the next plugin release after 1.3.0, and must be in before 1.4.0's U6 work prices GPU chunks the same way. If the arm is ready before U5 closes, it may ride in 1.3.0 | plugin release after 1.3.0 |

**Does an acceptance form change?** Yes. (d) moves from "the global hitch count, expected 0" to the build's chunk
overruns, with a median-price grade. At U5 that grade is recorded rather than graded, and it becomes graded once the
`prefix_prefill` arm lands. No promise changes, and no principal question is needed: the release scope is unchanged.

### 15.32 Round 32 (rev 1.18.12): the `prefix_prefill` arm timed under the build's conditions

Source: the box's §15.31 calibration round, Wizard `claude/project-thread-wt2dpx` at `e6dc58993`, record `259382c6e`
(`build/box-2026-10-22-u5-calibration/`).

**What was measured:**
- 5.9 (d) FAIL: the median chunk's worker time was 1.481 × K × `TickBudgetMs`, and 250 of 250 chunks overran.
- The planned median was 20.0 ms/token, from the `prefix_prefill` pair (15.349 ms/token + 0.004666 ms/position,
  saved-estimate). The worker took 30.7 ms/token.
- Before calibration, on the batched fallback, the ratio was 1.617.
- Everything else in 5.9 passes.

One capped loop: the builder changes the arm, the code reviewer reviews once, and one box round reads (d) and the
attribution runs below.

| Item | Ruling | Dependent gate |
|---|---|---|
| The floor is not the cause | Agreed with the conductor's reading. K is the planner's ceiling of the planned ms over the tick, so it grows with the price. At a correct price of about 30.7 ms/token, an 8-token chunk plans about 246 ms, K = 15, and K × 16.6 = 249 ms, so the median passes. The record's line "pricing alone cannot pass at this floor" confuses the share (11.62 ms) with K × tick. The floor sets every chunk above the share, and (a) already allows that (§15.29) | — |
| 1. The fix: same conditions, a margin, or another grade? | **Time the arm under the build's conditions. No margin factor, and the grade stays.** A margin is an unmeasured constant laid over a mismeasured price, the same fault as pricing a build at `PrefixBeginCostMs`. It would pass on this box and fail on the next. The grade is right: it asks that the price not be systematically low. The arm is wrong: it times `sslm_prefix_prefill` in `SuperSLM.EstimateCosts`, which blocks the game thread for minutes (§15.26), so it measures with the game not ticking, outside the lanes, and away from the build's interleaved generation. **The arm becomes an in-situ arm.** It runs a cache build of its length through the subsystem's real path: begin and chunk jobs on a worker lane, the resolved decode mode, task count and threads, the game ticking, and one interleaved generation on the build's lane. That generation is the reference load cell 5.9 (c) runs. The arm fits the pair (per token, per position) from the ledger's chunk worker ms, which are the same rows (d) grades. **Attribution, the same box round, recorded:** the per-token worker ms of the isolated arm (as now), of an in-situ build with no generation, and of an in-situ build with the generation. These say which condition explains the 1.5×. The fix does not depend on which one it is, because the in-situ arm includes all of them | the plugin release after 1.3.0 (§15.31) |
| Scope note | Every other price `EstimateCosts` writes is timed in isolation too, so the same bias may exist there. That is outside this cell, and it is recorded for the cost plan's next calibration review, not changed here | — |
| 2. Does it block plugin 1.3.0? | **No; §15.31 stands.** At U5, (d) runs in its new form and is recorded, not graded. 1.3.0 may ship the isolated arm's pair: it is less low than the batched fallback (1.48 against 1.62), and an under-price only makes a cache build's chunks late, which is counted and never hidden. The in-situ arm and (d)'s graded median land in the next plugin release, before U6 prices GPU chunks | U5 (recorded), the next plugin release |

**Does an acceptance form change?** No new form. (d)'s grade is unchanged and stays recorded at U5. The arm's method
changes, from isolated timing to in-situ timing from the ledger, and three attribution readings are added to that box
round. No promise changes, and no principal question is needed.
