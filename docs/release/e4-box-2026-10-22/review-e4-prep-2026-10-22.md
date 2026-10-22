# Code review: E4 preparation (`ab62b4cb`, `31747815`), 2026-10-22

Reviewer: code reviewer, SuperSLM 1.14.0. Branch `claude/precache-1p14-post-m1`, fetched; tip `31747815`. Built and
run in a separate worktree (`/home/claude/review-e4`, GCC 13.3 Release, Linux), not in the author's tree. Inputs:
plan §9 E4 row, §7 4.3, 6.6, 10.1–10.3, §7 *Geometry*, the carried T-3060 F10–F13, §10 R1; the test author's
record `release-1p14/e4-cloud-2026-10-22.md`; the run sheet `release-1p14/e4-box-runsheet-2026-10-22.md`.
Stand-ins were the author's scratch artifacts, read-only, with their SHA-256 checked against the record
(`standin0p5-cap32k` `b83563e6…1d3d`, `standin0p5-realwidth` `e8c6379b…`, `standin1p5-cap32k`).

## What I executed

| Run | Result |
|---|---|
| Fresh configure of a new build dir at `31747815` | **`superslm_precache_reference_gen: references would be written unanchored-no-merge-base`**; the compiled `T3035_ENGINE_ANCHOR` is `"unanchored-no-merge-base"`. A second `cmake` on the same dir gives `1948e1f2…`. Finding 1 |
| Generator source vs `origin/curie/t3035-precache-refs` (`972d5b7b`) | `t3035_reference.cpp` and `t3022_precache_reference.cpp` are the **same git blobs** (`dd6c53b4…`, `c566be98…`): verbatim |
| In-tree generator, nested family, `run1` then `run2` (separate processes, after the reconfigure) | exit 0 both; **10 of 10 pairs byte-identical**; header `anchor=1948e1f2… harness=31747815… B=128 cap=32768 decode_modes_available=1`. Rows equal to the author's scratch build of the T-3035 branch (only `harness=` and `tokenizer=` differ in the header, as they should) |
| `ctest -R ^superslm_precache_real_two_process$` on the stand-in, author's scratch licence, my own blob dir | **100% passed**; capture and restore `10.1/E4` (4,184 checks) and `10.2/E4` (4,377) green; `rule B = 128 (engine 128, helpers 128) … page 786432 bytes`; `s = 7, draw = 3` for every persona; `21 hashes pinned … 100 graded adopters` |
| Mutant `hint-s-at-2B`, 10.2/E4 restore phase | **killed**: `RED (4377 checks, 30 failed)`, "nested00 shares 3 pages; floor(1012 / 128) = 7", fill probe "does not draw exactly 3 pages" |
| Mutant `matvec-last-short`, 6.6/E4 + 6.6/E4:rule-b on the real-width stand-in (unmutated build green first, `calls=3008 tasks=10544`) | **killed on both legs**: pinned `RED (31 failed)`, rule-B `RED (32 failed)`, "token differs at 1 (236 vs 29)" |
| Control `restore-phase-prefill`, 10.1/E4 restore | **red**: `sslm_pc_test_only_prefix_prefill_calls() == 0u (1 vs 0)` |
| All five `apply_e4_mutant.py` anchors on a clean copy | each applies exactly once; an unknown name fails |
| 10.3/E4 slices under the author's scratch receipt | `0-1` runs, prints both lines and `0 of 2 … at or above 1000x`, red (stand-in, as expected); `3-x`, `5-3`, `0-1 ` refused by name; `" 0- 1"` accepted as 0-1 (harmless) |
| 10.3/E4:worst restore of the author's capture blob from **another build** of the same commit | green, `bytes_including_digest=469762744`, `blob_sha256=5ea3934e…` equal to the capture's, re-save identity checked. Phase `Restore` (wrong case) refused. A copy with **one bit flipped** at byte 300,000,000: **red** (`SSLM_INVALID_ARGUMENT`) |
| 10.3/E4:worst capture phase, my own build, 1.5B stand-in | green in 20 min 9 s (one thread); `bytes_including_digest=469762744`, **`blob_sha256=5ea3934e…` byte-identical to the author's capture** from a different build; no `.partial` left after the rename |
| All E4 inputs unset | every selected E4 cell red, none vacuously green; `--list` shows the 7 E4 cells B1 expects |
| `e4_box_env.ps1` | parses clean (`Parser::ParseFile`, PowerShell 7.4.6). `E4-Run` exercised on pwsh 7.4 under `Set-StrictMode 2` + `Stop` with a native program writing stderr and exiting 3, into a log path with a space: lines stream, stderr is captured, exit 3 is logged. **Windows PowerShell 5.1 was not available**; the 5.1 findings below are from its documented behaviour and this repo's own `longleg.ps1` notes |
| X1 bundle `e4-cross-compiler-31747815/` | `sha256sum -c` all 16 OK; GNU and Clang blobs equal (`e869127a…`); receipts and jobs name `31747815` |

## Correctness against the plan

- **Rule-B geometry.** `RequireRealRuleB` writes B from the rule (int8 elements, S = 8 KiB above cap 4,096): 128 at
  D = 64, 64 at D = 128, sets `fx.pinned`, and checks engine B, helpers' B, the plan's figure, block bytes and page
  bytes before anything else. `superslm_precache_e1` carries no `PKV_FIXTURE_B_FROM_RULE`. Matches §7 *Geometry*
  rev 1.13–1.15. The plan row's "re-derivation not yet done" is stale, as the author says.
- **Fixture-bytes fix.** `TakeRealArtifact` moves the `AlignedBuf` and the model handle out of `ModelFile`;
  `~Fixture` unmaps the model in its body before the buffer member is freed, so the order is right. CFG1's
  `vocab_size` at +32 matches `model.cpp`'s parser. The fix is real: without it `ArtifactHiddenSize` refuses on any
  artifact, so 6.6 rule-B and the cap-length size check could not have passed.
- **6.6.** `calls > 32` is stronger than F13's "above 0" and is what makes the plan's matvec mutant reachable on the
  rule-B leg (executed above). Correct.
- **10.3 slices.** Per prefix the bar is unchanged: 8 pairs, `lower ≥ 1000` on the 95% log-ratio interval, receipt
  checked in every process, plus `graded == slice size`. No per-slice weakening. What the split loses is the
  *union*: findings 4 and 5.
- **10.3/E4:worst.** The plan asks for the cap-length 1.5B restore "recorded as ms and MiB/s" under the receipt; the
  restore phase does that, and now also checks size, digest (by restoring) and re-save identity, which the
  single-process version did not. The capture runs receipt-free on another machine, which is right because the
  prefill time is not a plan figure. The producer check is engine version + adapter, not build, so a W1 blob from
  another MSVC build restores on the box (shown cross-build above). No weakening found.
- **Generator.** Verbatim, reproducible run to run, and its rows equal the T-3035 branch's own build. Its anchor
  wiring is broken on first configure (finding 1).

## Run sheet

Durations (estimated, I cannot time the box): every box piece is ≤ 15 min except **T1 (~25 min)**, finding 3. Timing
pieces T1, B8a–B8d, B9 are marked ALONE. Every piece is re-runnable: B3/B4 overwrite, B6 clears its blobs, B10 deletes
its build dir, W1 writes `.partial` then renames. Commands and expected lines match the code: the 7 cells in `--list`;
the `rule B = …` lines; `6.6 rule-B hook: calls=N`; the 10.3 line with `bytes_including_digest = 1080 + 6144·L`
(184 + CFG1 hidden 896 + L × 6,144); `10.3 slice … N of N`; the worst lines with 469,763,768 bytes (184 + 1,536 +
32,768 × 14,336) and `MiB=448.002`; `10.2 nestedNN: B = 128, s = 7, draw = 3`; the two-process gate line; the
`PC-TIMING-1` field set and geometry string; the licence's instrument file.

## Findings

1. **Blocking. The first configure of any build directory anchors the generator `unanchored-no-merge-base`.**
   `precache.cmake:78` runs `git merge-base HEAD ${PRECACHE_REFERENCE_LINE}`, but the variable is first set at
   `:104` (`set(PRECACHE_REFERENCE_LINE "origin/main" CACHE STRING …)`). On a fresh cache it is empty, so
   `merge-base HEAD` fails. Executed: fresh configure → unanchored; reconfigure → `1948e1f2`. B1 configures a fresh
   `build-msvc` once, so on the box B1 stops every time (or, with finding 2, does not notice and B3/B4 write
   unanchored references that every grading refuses). The cloud record's anchored header came from a reconfigured
   tree. Fix: move the `set(PRECACHE_REFERENCE_LINE …)` above the generator block. This changes the pinned commit,
   so X1's GNU/Clang jobs must be re-captured at the new commit and the run sheet's commit lines updated.
2. **Blocking. Under Windows PowerShell 5.1, `E4-Run` writes mixed-encoding logs, and B1's guard reads one.**
   `E4-Run` starts each entry with `Out-File -Encoding utf8` (UTF-8 with BOM on first write), then pipes program output
   through `Tee-Object -FilePath $Log -Append`. 5.1's `Tee-Object` has no `-Encoding` and writes UTF-16LE, so the
   program lines land as UTF-16 inside a UTF-8 file. `Select-String -Path $log -Pattern 'unanchored'` (B1) decodes it
   as UTF-8 by the BOM and cannot match the UTF-16 text, so it prints nothing and the guard passes. A1 also has to read
   these logs. This repo's own `tools/longleg_kit/longleg.ps1:325` records the same trap ("Windows PowerShell 5.1's
   redirection writes UTF-16LE into the middle of the log") and uses `[System.IO.File]::AppendAllText` per line. On
   pwsh 7 (what the author parse-checked) `Tee-Object` writes UTF-8 and the logs are correct, so the 7.x check does not
   cover this. Fix: `& $Exe @Arguments 2>&1 | ForEach-Object { $l = "$_"; [System.IO.File]::AppendAllText($Log, $l + "`r`n"); $l }`,
   with the header and exit lines written the same way (no BOM). B5's `Tee-Object` should follow, for one log encoding.
   Not executed on 5.1; inferred from 5.1's documented behaviour and the kit's note.
3. **Blocking (run sheet). T1 is budgeted at about 25 box minutes, over the 20-minute piece limit**, and all 25 are
   marked ALONE, including a 10-minute mutant build that is not timing. Split it: T1a builds the must-reject mutant
   (~10 min, need not be alone), and T1b runs the must-reject and must-accept slices alone (~5 min). The shape stays
   the debunker's to change, but the sheet should not budget a piece over the limit.
4. **Blocking (run sheet). "On a miss, re-run that slice once" weakens the 10.3 bar.** A slice that misses and is
   re-run gets two draws at the 95% interval, and the sheet keeps the green one. The plan has no retry: F12 refuses an
   unresolved result, and §10 R1 makes a miss the fold trigger. Fix: a miss stands and is reported with its log. Any
   re-run rule (for example, a named disturbance on the box) must be the debunker's, written down at T1 before any
   reading, with both readings reported to the planner.
5. **Blocking (run sheet). Nothing checks the union of the slices.** The whole run proved ten prefixes in one process
   under one receipt and one binary. The slices each prove their own prefixes, but no step checks that B8a–B8d cover
   0–9 exactly once under one receipt SHA from one runner, so a skipped slice, or one taken on another build, still
   reads "four greens" (the instrument hash covers `e4_real.cpp`, not the engine or the binary). Fix, run sheet only:
   a B8e step that reads the four logs and requires ten distinct `flatNN` lines, each with `lower ≥ 1000`, one
   `receipt=` value, and every slice "N of N". Better still, print `ShaFile(argv[0])` (the runner's SHA) in each 10.3 line
   and in the worst line, so B8e and A1 can tie them to one build.
6. Optional. `e4_box_env.ps1` finds Visual Studio by the side-by-side component
   `Microsoft.VisualStudio.Component.VC.14.34.17.4.x86.x64`. If the box's 14.34 is its install's own toolset (VS 17.4),
   that component is not registered and the script throws. `longleg.ps1` (the kit known to work on the box) finds the
   install by `Microsoft.VisualStudio.Component.VC.Tools.x86.x64` and selects 14.34 with `-vcvars_ver`; do the same.
   The `VCToolsVersion -notlike '14.34.*'` guard already catches a wrong toolset.
7. Optional. B10: `capture_compiler_job.py` configures with no compiler named. Set `$env:CC='cl'; $env:CXX='cl'` in
   the piece (or pass `-DCMAKE_CXX_COMPILER=cl` through the script), so a gcc or clang on the box's PATH cannot make a
   `GNU.*`/`Clang.*` job that overwrites X1's files in `$xc`.
8. Optional. The generator's anchor compares the merge-base with HEAD's *commit* (`git diff --quiet <mb> HEAD -- src
   include CMakeLists.txt`). A dirty engine tree would still be written as anchored. B0's `status --porcelain`
   covers the box, but adding `git diff --quiet HEAD -- src include CMakeLists.txt` makes the target safe by itself.
   The anchor is also fixed at configure time; the comment says to reconfigure, and that is fine.
9. Optional. 6.6's rule-B leg does not bind the artifact's identity: the vocabulary now comes from CFG1 and nothing
   checks the SHA, so a stand-in passes it (that is how the cloud ran it). B0 hashes the artifact, but the B2 log
   cannot show which file ran. Print the model SHA-256 in the `6.6 rule-B:` line for A1.
10. Optional. With slicing, the first pair of each slice is the process's first prefill, cold, four times instead of
    once. A cold prefill inflates prefill time and so the ratio, which is non-conservative. Add one untimed warm-up pair
    per slice.
11. Optional. Provenance wording: the commit message and the `precache.cmake` comment say both generator files were
    "last changed at abe60753". `t3022_precache_reference.cpp` was last changed at `c9d003c8`. The blobs are identical
    either way.
12. Optional, run sheet hygiene. (a) B1's `Select-String` reads the whole appended log, so after a fix a stale
    "unanchored" line from the failed attempt stops the re-run; start B1 with `Remove-Item $log -ErrorAction
    SilentlyContinue`. (b) W1: say that `$env:SUPERSLM_PRECACHE_REAL_1P5_MODEL` must be set *after* the dot-source line,
    because the env script overwrites it. (c) `E4-Run`'s comment says it "returns its exit code"; it returns every output
    line followed by the code. That is harmless as used, but `$c = E4-Run …` would be wrong. (d) The runner's stdout is
    block-buffered when piped, so "line by line" holds only for lines followed by `fflush`, and a crash loses the
    buffered tail. (e) `apply_e4_mutant.py` opens files in the locale encoding. On Windows that is cp1252, which
    round-trips the three target files today (I checked: no undefined bytes), but pin `encoding="utf-8"`.

## Verdict

**FIX.** The test code is correct against the plan. Rule-B geometry, the fixture-bytes fix, the per-slice timing bar,
and the split worst cell with separate capture and restore are sound, and none of them weakens a pass bar. The
generator is verbatim and byte-reproducible. Three of the five mutants and controls were re-run here and are killed.
Findings 1 and 2 stop the box run or blind its first guard. Findings 3–5 are run-sheet changes: a piece over the
time limit, a retry that weakens the 10.3 bar, and an unchecked union of the slices. Findings 1 and 2 need one small
commit (the cmake line move and `E4-Run`'s logging). The run sheet then needs re-pinning to that commit, and X1 needs
re-capturing at it.

## Re-check (fold `e8c029ed`, 2026-10-22, one round)

Scope: blocking findings 1–5 and anything new the fold introduced. I used a separate worktree at `e8c029ed` with a
fresh build dir (GCC 13.3 Release), and the same read-only stand-ins and scratch licence as before. The fold record is
`e4-cloud-2026-10-22.md` §5, and the run sheet is re-pinned to `e8c029ed`.

| # | Status | Evidence |
|---|---|---|
| 1 | **Fixed** | `set(PRECACHE_REFERENCE_LINE …)` now comes before the generator block. A **single fresh configure** printed no "unanchored" line and compiled `T3035_ENGINE_ANCHOR="1948e1f2…"`, `T3035_HARNESS_COMMIT="e8c029ed…"`. `superslm_precache_real_two_process` from that tree: 100% passed (10.1 4,184 and 10.2 4,377 checks, 100 graded adopters). The dirty-tree check (optional 8) is also in |
| 2 | **Fixed** | `E4-Run` and the new `E4-Append` write with `[IO.File]::AppendAllText(…, UTF8Encoding($false))`, with no `Tee-Object` or `Out-File`. On pwsh 7.4 under StrictMode 2 + Stop, with a log path containing a space and a native program writing to stdout and stderr and exiting 3: one UTF-8 log, no BOM, and `Select-String 'unanchored'` matches. The env script parses clean. Not run on 5.1, but the code path is now .NET-only, the same as `longleg.ps1` |
| 3 | **Fixed** | T1 is split into T1a (≤ 12 min, build, need not be alone) and T1b (≤ 6 min, ALONE). The totals match |
| 4 | **Fixed** | B8 now says "the reading stands. Do not re-run the slice". The only exception is a re-run rule the debunker writes at T1b before any reading, and then both readings go to the planner |
| 5 | **Fixed** | `runner=<SHA>` is on every 10.3 prefix line, the slice summary and the worst line. Executed: slices `0-1` and `2-2` and the worst restore all print `runner=8fd7e9a0…`, equal to `sha256sum` of the binary. B8e, run on pwsh 7.4 / StrictMode 2 against synthetic logs in the real format: five PASS on a well-formed set |

New in the fold, checked:
- The warm-up pair (optional 10) is untimed, uses the slice's first prefix and releases its handles. It adds about one
  prefill per slice, so B8b's ≤ 15 min estimate still holds. It is not graded and changes no bar.
- `RunnerSha()` fails closed (red when the program can't be hashed). The Windows branch declares `GetModuleFileNameA`
  itself. No `<windows.h>` is on `e4_real.cpp`'s include chain (checked with `-M`), so the declaration conflicts with
  nothing in its translation unit, and on x64 the `extern "C"` name links against kernel32 by default. This branch
  has **not been compiled with MSVC**: B1 is its first compile.
- The X1 bundle at `e4-cross-compiler-e8c029ed/` checks out: `sha256sum -c` is clean, 17 files, both jobs carry
  `source_revision=e8c029ed…`, and the GNU and Clang blobs are equal (`e869127a…`, unchanged).

New optional findings (non-blocking):
- R1. B8e reads `$slices.Count` bare. When zero or one slice summaries match (a missing summary, a crashed slice),
  StrictMode 2 throws "property 'Count' cannot be found" (executed on pwsh 7.4), and a missing log throws "Cannot
  find path". Both fail closed, so no false PASS is possible, but the operator sees an exception instead of a FAIL
  line. Use `@($slices).Count`, and test the paths first.
- R2. Watch B1's log for the first MSVC compile of `RunnerSha`'s Windows branch (see above). If it fails to build,
  that is a one-line fix, not a design issue.

**Verdict: SHIP.** Blocking findings 1–5 are fixed, verified by execution where this environment allows (the 5.1
behaviour and the MSVC compile are the two things it does not). The fold introduces no weakening of any bar.
