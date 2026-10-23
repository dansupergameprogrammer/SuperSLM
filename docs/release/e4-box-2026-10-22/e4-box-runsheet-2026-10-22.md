# E4 box run sheet: real-artifact oracle and timings (SuperSLM 1.14.0), 2026-10-22

Plan: `plans/superslm-prefix-precache/plan.md` §9 E4, §7 4.3, 6.6, 10.1–10.3, the carried T-3060 F10–F13 portions, §8
(references, timing harness), §10 R1, R12. Written by the test author. The plan text is not edited here.

**Pinned commit:** `e8c029ed68f6d3fb0a728673c9a41d16d433916d` on `claude/precache-1p14-post-m1` (re-pinned from `31747815` after
the code review `code-review/review-e4-prep-2026-10-22.md`; its findings 1–5 are folded). Build exactly this commit, in a detached worktree, even if
the branch has moved since: 4.3's cloud receipts (piece X1) name it as their `source_revision`, and the timing
instrument's hash is `tests/precache/e4_real.cpp` at this commit. The reference line is `origin/main` = `1948e1f2`
(the M1 squash); `git merge-base` of the pinned commit and `origin/main` is `1948e1f2`, and the engine sources
(`src/`, `include/`, `CMakeLists.txt`) at the pinned commit are byte-identical to `1948e1f2`'s.

**Generator commit (B3g, B3, B4 only):** `47bdcc9e24917f91e5f9bf67e8a0141662ca5114`, plan rev 1.18.13 §15.33 (a). It
changes only `tests/precache/reference/t3035_reference.cpp` (and a comment in `precache.cmake`): the generator's
eighth argument is now `RECORD`, the per-stem discrimination record, with no default. Everything else on this sheet
stays on `e8c029ed` and its build `build-msvc`. The generator is built from its own worktree and build directory, so
`src` and `build-msvc` are never moved.

## Re-run after the L1 refusal (rev 1.18.13 §15.33), 2026-10-23

The debunker refused L1 because every header bound the superseded v1 record (`real0p5-*/<stem>.discrimination.json`).

**Re-run, in this order:** B3g (new, ≤ 12 min), B3 (12), B4 (8), B5 (2), A0 (coverage auditor, records only, no box
time), L1 (debunker, ~10), B6 (10). **About 54 box minutes**: the plan's 42, plus 12 to build the generator at its own
commit.

**Not re-run:** B0–B2, B7, T1a/T1b, B8a–B8e, W1, B9, X1, B10, B11. They run at `e8c029ed`, and `superslm_precache_e1`
rebuilt at `47bdcc9e` is byte-identical (cloud: `2531182b…5d26` at both). X1's receipts name `source_revision`
`e8c029ed`, and B10 compares them against an MSVC job built from `D:\_scratch\e4\src`, which stays at `e8c029ed`, so
X1/B10 pairing holds. If B10 were built from a `47bdcc9e` tree instead, it would refuse "not built from one source
revision", and X1 would need re-capturing there (cloud) plus B10 again (15 min). Do not do that.

**`RECORD` (rev 1.18.13a): `{id}.stream.json`.** Each stem has two commissioned v2 sidecars, `<stem>.stream.json`
(`T-3035-precache-stream-v2`, stream vitality) and `<stem>.byte.json` (`T-3035-precache-byte-v1`, restored-sequence
operand). The header's `discrimination_record_sha256` binds the stream sidecar, because stream-v2 is the instrument
that grades what a reference holds: token streams. The byte sidecars are named in L1's population manifest as
provenance, not in `record_hashes`.

## What E4 needed, and what is already done

Checked against the files, not the plan's status words.

| Item | State before today | State now |
|---|---|---|
| 10.1/10.2 rule-B re-derivation (§7 *Geometry*, rev 1.13–1.16) | **Already done in `e4_real.cpp`** (the plan row's "not yet done" is stale): `RequireRealRuleB` writes the rule B in the test (128 at 0.5B, 64 at 1.5B), sets `fx.pinned`, checks the engine's B, the helpers' B and the plan's figure, and `page_size == B·bytes_per_token`; pools sized through `CapPages()`/`R()`; `:388` is `s = floor(|world|/B)` with the draw `ceil((|world|+|persona|)/B) − s` read by the two-sided fill probe | unchanged; **ran green in the cloud** on a stand-in with the real 0.5B K/V geometry (rule B 128, s = 7, draw 3 for all ten personas) |
| A defect the box would have hit | the hand-built real fixtures never carried the artifact's bytes, so `Header()` and `ArtifactHiddenSize()` refused "lacks an SSLM section header": 6.6's rule-B leg and 10.3's cap-length size check could not pass on any artifact | **fixed** (`ab62b4cb`, `TakeRealArtifact`) |
| Timing instrument (§8, F12) | the harness is `Timing103` in `e4_real.cpp` with the 13-field `PC-TIMING-1` receipt check; **no receipt exists and nothing is commissioned**; one ~25-min uncheckpointed run, and the 1.5B cap-length leg (~1 h of prefill) inside it | sliced (`SUPERSLM_PRECACHE_TIMING_PREFIXES`), one untimed warm-up pair per slice, each line carries `runner=<SHA-256 of the program>`, cap-length leg split into capture and restore phases (`10.3/E4:worst`); **commissioning is the debunker's (T1)** |
| Licensed references at the post-M1 line | the T-3035 population is anchored at `5d5ca58`; never regenerated at `1948e1f2`, never licensed for E4 | generator carried in-tree (`superslm_precache_reference_gen`), builds at the anchor, **reproduces run1 = run2 in the cloud**; regeneration (B3, B4) and the licence (L1) are box/second-seat |
| 4.3 / F10 | three compiler jobs needed | GNU and Clang jobs captured in the cloud at the pinned commit (X1); MSVC job and compare are B10 |
| 6.6 / F13 | both legs written; the rule-B leg never required a projection split | the rule-B leg now requires `calls > 32` so the matvec mutant is reachable there |
| Mutants and controls | 10.2's and 6.6's named mutants not executed on E4 cells | executed in the cloud, all killed, 6.6's on both legs (record: `release-1p14/e4-cloud-2026-10-22.md`) |

## Order and dependencies

```
B0 worktrees ─ B1 build ─┬─ B2 smoke (6.6 both legs)
                         ├─ B7 timing prompt manifest ─ T1a, T1b commission [DEBUNKER] ─ B8a..B8d slices (alone) ─ B8e
                         │                                                       └─ B9 worst restore (alone) ─ needs W1
                         ├─ B10 4.3 MSVC job + compare (needs X1's bundles copied in)
                         └─ B11 6.6 matvec mutant on MSVC (optional)
B0 ─ B3g generator @47bdcc9e ─┬─ B3 refs P1 ─┐
                               └─ B4 refs P2 ─┴─ B5 pair check ─ A0 re-audit [AUDITOR] ─ L1 licence [DEBUNKER] ─ B6 (needs B1)
W1 cap-length 1.5B capture [SPARE PC via the long-leg kit, -E4Worst; not the home box]
A1 E4 discrimination re-audit [COVERAGE AUDITOR; no box time] after B6, B8, B9, B10
```

Nothing in E4 needs a GPU. The long-leg kit's spare PC is needed only for W1, because a 32,768-token
1.5B prefill on one thread runs about an hour (R12).

**Rules for every piece.** Open a **fresh PowerShell window** per piece and dot-source the environment first. Every
piece writes its log under `D:\_scratch\e4\logs` line by line as it runs. A piece is **resumable by re-running it**:
each one either writes complete files under new names or overwrites its own outputs. If the box crashes, re-run the
piece that was running; never run the next piece on a crashed piece's outputs. **Timing pieces (T1b, B8a–B8d, B9) run
alone**: close other programs, no other build or run on the box, and wait about a minute after the previous piece.

```powershell
. D:\_scratch\e4\src\tests\precache\e4_box_env.ps1     # every piece after B0 starts with this line
```

## Pieces

### B0. Worktrees and checks (3 min) — Dan

```powershell
cd D:\SuperSLM
git fetch origin
git rev-parse --is-shallow-repository                 # must print: false
New-Item -ItemType Directory -Force D:\_scratch\e4 | Out-Null
if (-not (Test-Path D:\_scratch\e4\src))   { git worktree add --detach D:\_scratch\e4\src e8c029ed68f6d3fb0a728673c9a41d16d433916d }
if (-not (Test-Path D:\_scratch\e4\t3035)) { git worktree add --detach D:\_scratch\e4\t3035 origin/curie/t3035-precache-refs }
git -C D:\_scratch\e4\src rev-parse HEAD              # must print: e8c029ed68f6d3fb0a728673c9a41d16d433916d
git -C D:\_scratch\e4\src merge-base HEAD origin/main # must print: 1948e1f2ea4528790dc6701e26c47bda5739b003
git -C D:\_scratch\e4\src diff --quiet 1948e1f2 HEAD -- src include CMakeLists.txt; $LASTEXITCODE   # must print: 0
git -C D:\_scratch\e4\src status --porcelain          # must print nothing
Get-FileHash D:\_artifacts\SuperSLM\te442-qwen2.5-0.5b-instruct-dg\qwen2.5-0.5b-instruct-dg.sslm
#   4B27E2D6E132D710A4B174D5FFC15E7705691CA37ABF15ECA65503803D0DBA35
Get-FileHash D:\_artifacts\SuperSLM\t3035-precache-refs\fixtures\pkv_def.sslm
#   994B58B49D3660BCCFDD59DC6315028EA0203F37F4705D70EB7CE4DBECDB1DB3
```

Pass: every line prints what its comment says. If `origin/main` has moved past `1948e1f2` and been merged into the
branch, the merge-base line differs: stop, the population must be regenerated at the new anchor (this sheet's anchor
is wrong).

### B1. Build, MSVC 14.34 Release (≤ 15 min) — Dan

```powershell
. D:\_scratch\e4\src\tests\precache\e4_box_env.ps1
$log = E4-Log 'B1-build'
Remove-Item $log -ErrorAction SilentlyContinue          # a re-run reads only its own log
E4-Run $log cmake @('-S', $E4Src, '-B', $E4Build, '-G', 'Ninja', '-DBUILD_TESTING=ON', '-DCMAKE_BUILD_TYPE=Release',
                    '-DCMAKE_C_COMPILER=cl', '-DCMAKE_CXX_COMPILER=cl')
E4-Run $log ninja @('-C', $E4Build, 'superslm_precache_e1', 'superslm_precache_reference_gen', 'superslm_precache_reference_header')
E4-Run $log $E4Runner @('--list') | Select-String 'E4'
Select-String -Path $log -Pattern 'unanchored'
```

Pass: both commands exit 0; the list shows 7 E4 cells (`10.1/E4`, `4.3/E4`, `10.3/E4`, `10.3/E4:worst`, `10.2/E4`,
`6.6/E4`, `6.6/E4:rule-b`); the last line prints **nothing** (a "references would be written unanchored-…" line means
the configure did not see the anchor: stop). Resume: re-run; Ninja rebuilds only what is missing.

### B2. Smoke: 6.6 both legs (3 min) — Dan

```powershell
. D:\_scratch\e4\src\tests\precache\e4_box_env.ps1
E4-Run (E4-Log 'B2-6.6') $E4Runner @('=6.6/E4', '=6.6/E4:rule-b')
```

Pass: `6.6/E4 green`, `6.6/E4:rule-b green`, the line `6.6 rule-B: model_sha256=4b27e2d6e132d710…0dba35` (the real
0.5B, as B0 hashed it), the line `6.6 rule-B: rule B = 128 (engine 128, helpers 128), cap 32768,
cap_pages 256, page 786432 bytes`, and `6.6 rule-B hook: calls=N` with **N > 32** (projections split at the real
width; F13). Exit 0.

### B3g. Build the generator at its commit (≤ 12 min) — Dan; needs B0

```powershell
. D:\_scratch\e4\src\tests\precache\e4_box_env.ps1
$GenSrc = Join-Path $E4Work 'gensrc'; $GenBuild = Join-Path $E4Work 'build-gen'
if (-not (Test-Path $GenSrc)) { git -C D:\SuperSLM worktree add --detach $GenSrc 47bdcc9e24917f91e5f9bf67e8a0141662ca5114 }
git -C $GenSrc rev-parse HEAD              # must print: 47bdcc9e24917f91e5f9bf67e8a0141662ca5114
git -C $GenSrc merge-base HEAD origin/main # must print: 1948e1f2ea4528790dc6701e26c47bda5739b003
git -C $GenSrc status --porcelain          # must print nothing
$log = E4-Log 'B3g-gen-build'
Remove-Item $log -ErrorAction SilentlyContinue
E4-Run $log cmake @('-S', $GenSrc, '-B', $GenBuild, '-G', 'Ninja', '-DBUILD_TESTING=ON', '-DCMAKE_BUILD_TYPE=Release',
                    '-DCMAKE_C_COMPILER=cl', '-DCMAKE_CXX_COMPILER=cl')
E4-Run $log ninja @('-C', $GenBuild, 'superslm_precache_reference_gen')
Select-String -Path $log -Pattern 'unanchored'
E4-Run $log (Join-Path $GenBuild 'superslm_precache_reference_gen.exe') @()
```

Pass: both commands exit 0; the `unanchored` line prints **nothing**; the last command exits 1 with a usage line
ending `RECORD [COUNT]  (RECORD: record path with {id} for the stem, no default)`. Resume: re-run.

### B3. Regenerate references, P1 real 0.5B flat, run1 and run2 (≤ 12 min) — Dan; needs B3g

The T-3035 generator at `47bdcc9e`, the T-3035 command form, and each stem's commissioned v2 sidecar as `RECORD`. Two
separate processes. First set the refused population aside (once, kept as L1's evidence), then write a fresh
directory.

```powershell
. D:\_scratch\e4\src\tests\precache\e4_box_env.ps1
$E4Record = '{id}.stream.json'     # rev 1.18.13a: the header binds the stream-v2 sidecar
$Gen = Join-Path $E4Work 'build-gen\superslm_precache_reference_gen.exe'
$V2 = Join-Path $E4T3035 'tests\t3035-precache-refs\discrimination-v2'
$refused = "$E4Refs-v1bound-refused"
if ((Test-Path $E4Refs) -and -not (Test-Path $refused)) { Move-Item $E4Refs $refused }
$out = Join-Path $E4Refs 'P1-real0p5-flat'; New-Item -ItemType Directory -Force $out | Out-Null
$tok = 'TokenizerView::Encode(rel10,4da4c4c)'
foreach ($run in 'run1', 'run2') {
  E4-Run (E4-Log "B3-P1-$run") $Gen @($E4Model05, $E4Prompts, $out, 'flat', $run, $tok, 'ids', (Join-Path $V2 "P1-real0p5-flat\$E4Record"))
}
```

Pass: each run exits 0 and prints `flat00 captured` … `flat09 captured`; 20 files `flatNN-runN.ref`. Each header
starts `# anchor=1948e1f2ea4528790dc6701e26c47bda5739b003 harness=47bdcc9e24917f91e5f9bf67e8a0141662ca5114
model_sha256=4b27e2d6… B=128 cap=32768 decode_modes_available=1`. B5 checks the record hashes. About 4–6 min per run.
Resume: re-run the piece (it overwrites both runs; the set-aside happens only once).

### B4. Regenerate references, P2 real 0.5B nested, run1 and run2 (≤ 8 min) — Dan; needs B3g

```powershell
. D:\_scratch\e4\src\tests\precache\e4_box_env.ps1
$E4Record = '{id}.stream.json'     # the same value as B3
$Gen = Join-Path $E4Work 'build-gen\superslm_precache_reference_gen.exe'
$V2 = Join-Path $E4T3035 'tests\t3035-precache-refs\discrimination-v2'
$out = Join-Path $E4Refs 'P2-real0p5-nested'; New-Item -ItemType Directory -Force $out | Out-Null
$tok = 'TokenizerView::Encode(rel10,4da4c4c)'
foreach ($run in 'run1', 'run2') {
  E4-Run (E4-Log "B4-P2-$run") $Gen @($E4Model05, $E4Prompts, $out, 'nested', $run, $tok, 'ids', (Join-Path $V2 "P2-real0p5-nested\$E4Record"))
}
```

Pass: each run exits 0 with `nested00 captured` … `nested09 captured`; headers as B3 plus the `world_*` and
`persona_*` hashes. About 3 min per run.

### B5. Pair check and population hashes (2 min) — Dan

```powershell
. D:\_scratch\e4\src\tests\precache\e4_box_env.ps1
$log = E4-Log 'B5-pairs'; $bad = 0
Get-ChildItem $E4Refs -Recurse -Filter '*-run1.ref' | ForEach-Object {
  $twin = $_.FullName -replace '-run1\.ref$', '-run2.ref'
  $a = (Get-FileHash $_.FullName).Hash; $b = (Get-FileHash $twin).Hash
  if ($a -ne $b) { $bad++ }
  "{0} {1} {2}" -f $a, ($(if ($a -eq $b) { 'pair-equal' } else { 'PAIR-DIFFERS' })), $_.FullName.Substring($E4Refs.Length + 1)
} | ForEach-Object { E4-Append $log $_; $_ }
"pairs differing: $bad" | ForEach-Object { E4-Append $log $_; $_ }
$E4Record = '{id}.stream.json'     # the same value as B3
$V2 = Join-Path $E4T3035 'tests\t3035-precache-refs\discrimination-v2'; $unbound = 0
Get-ChildItem $E4Refs -Recurse -Filter '*-run1.ref' | ForEach-Object {
  $stem = $_.Name -replace '-run1\.ref$', ''; $grp = $_.Directory.Name
  $want = (Get-FileHash (Join-Path $V2 ("$grp\" + $E4Record.Replace('{id}', $stem)))).Hash.ToLower()
  $got = ([regex]'discrimination_record_sha256=([0-9a-f]{64})').Match((Get-Content $_.FullName -TotalCount 1)).Groups[1].Value
  if ($got -ne $want) { $unbound++ }
  "{0} {1} {2}" -f $got, ($(if ($got -eq $want) { 'binds-v2' } else { 'NOT-V2' })), $stem
} | ForEach-Object { E4-Append $log $_; $_ }
"headers not bound to their v2 sidecar: $unbound" | ForEach-Object { E4-Append $log $_; $_ }
```

Pass: 20 lines, all `pair-equal`, and `pairs differing: 0`; then 20 `binds-v2` lines and `headers not bound to their
v2 sidecar: 0`. A differing pair is a reproducibility failure: stop and send
the log to the test author.

### A0. Pre-licence coverage re-audit — **SECOND SEAT: the coverage auditor** (records only, no box time); before L1

§15.33 (5): the auditor re-audits the re-bound population and its v2 bindings against the cells that will grade it
(10.1/10.2's exact-state assertions; the 6.1/6.4/6.7/8.4/8.5 rows T-3071 names, where E2's audit has not already
cleared them at this anchor, cited by record). Inputs: B5's log and the 20 run1 headers. L1 does not issue before it
clears. It does not replace A1.

### L1. Licence the regenerated population — **SECOND SEAT: the debunker** (about 10 box min if run on the box); after A0

Not the test author's. The debunker independently checks the population at `1948e1f2` (B5's log, the headers, and the
v2 sidecars named by `discrimination_record_sha256`) and writes, if it passes:
- `D:\_artifacts\SuperSLM\t3035-precache-refs-post-m1\manifest.json` (the population manifest), and
- `D:\_artifacts\SuperSLM\t3035-precache-refs-post-m1\license.txt`, the 9-field `PC-DISCRIMINATION-1` licence the
  loader reads (`format`, `status=COMMISSIONED`, `commissioning_verdict=PASS`, `must_accept=PASS`, `must_reject=PASS`,
  `anchor=1948e1f2…`, `population_manifest_sha256`, `instrument_source_sha256` = the SHA-256 of
  `tests\precache\precache_reference.h` at the pinned commit (the same file at `47bdcc9e`) unless the debunker names
  another instrument and changes `SUPERSLM_PRECACHE_DISCRIMINATION_INSTRUMENT`, and `record_hashes`, comma-separated:
  **exactly the 20 `<stem>.stream.json` hashes** the headers carry, and nothing else).
- The manifest also names the 20 `<stem>.byte.json` (byte-v1) sidecars with their SHA-256s, as provenance for the
  cells' exact-state assertions. They are **not** in `record_hashes`, so the licence accepts no hash a header does not
  carry (rev 1.18.13a).

**L1 note (rev 1.18.13 §15.33):**
- **Standing is read from the project registry, never from the sidecar's JSON.** Every v2 sidecar still carries its
  historical `"status": "QUARANTINED"` label (T-3071 §1). That label is not its standing. The registry lists
  `T-3035-precache-stream-v2` and `T-3035-precache-byte-v1` as COMMISSIONED (D-SLM8030).
- Every header's hash must be its stem's `<stem>.stream.json` in `discrimination-v2/P1-real0p5-flat/` or
  `P2-real0p5-nested/`, whose `instrument` (`T-3035-precache-stream-v2`) the registry lists as COMMISSIONED. No v1 hash (`discrimination/real0p5-*/<stem>.discrimination.json`)
  may appear in `record_hashes`: v1 is superseded, and nothing licenses against it (§15.33 (b)).
- **Each record is marked "measured at B = 16".** The sidecars carry `page_positions: 16`, and the population is at
  B = 128. The licence's manifest records "discrimination measured at B = 16" for every bound record. §15.33 (4)
  rules that no re-measure is needed for this licence: B8's mutants and A1 carry the B = 128 kill.
- Hand-edited headers are refused. The references come from B3/B4's generator, or not at all.

The cloud dry-run (`e4-cloud-2026-10-22.md` §6, §7) shows the loader passing with a licence whose `record_hashes` are
exactly the 20 stream hashes, and refusing one over v1 hashes only. E4's B6 cannot pass before L1 exists. The E2 (`6.2/E2:reference`) and E3
(`6.3/E3:reference`) populations at the same anchor are not in this sheet.

### B6. 10.1 and 10.2: the two-process leg (≤ 10 min) — Dan; needs B5 and L1

```powershell
. D:\_scratch\e4\src\tests\precache\e4_box_env.ps1
Remove-Item -Recurse -Force $env:SUPERSLM_PRECACHE_BLOB_DIR -ErrorAction SilentlyContinue
Push-Location $E4Build
E4-Run (E4-Log 'B6-two-process') ctest @('-R', '^superslm_precache_real_two_process$', '-V', '--timeout', '1500')
Pop-Location
```

Pass (all of):
- the fixtures `superslm_precache_reference_header_identity` and `superslm_precache_reference_anchor` Passed, and
  `reference merge-base fixture: origin/main -> 1948e1f2ea4528790dc6701e26c47bda5739b003`;
- capture process: `10.1/E4 green`, `10.2/E4 green`, each after `rule B = 128 (engine 128, helpers 128), cap 32768,
  cap_pages 256, page 786432 bytes`;
- restore process: `10.1/E4 green` and `10.2/E4 green`, with ten lines `10.2 nestedNN: B = 128, s = 7, draw = 3
  (world 1012, persona 183..211)`;
- `two-process gate: capture exited, 21 hashes pinned, restore exited, 100 graded adopters`; ctest `100% tests passed`.

`D:\_scratch\e4\blobs\capture-manifest.json` is F11's receipt (21 blob hashes, model, prompts, anchor). Resume: re-run
the piece (it clears its blobs first). Expected: about 3 min capture, 4–6 min restore.

### B6r. 10.1 and 10.2 again at the G1 fix (≤ 12 min build + ≤ 10 min run) — Dan; needs L1's licence; after A1's G1

The coverage auditor's A1 re-audit (`a1-e4-reaudit-2026-10-23.md`) found gap G1: 10.2's restore phase never asserted
`sslm_pc_test_only_prefix_prefill_calls() == 0`, and a nested restore that ran 30 scratch prefills stayed green.
Commit `1c145b53c7cf2e81e0b0ad8535356d9763c68eeb` adds that assertion at the end of `Real102`'s restore phase. In the
cloud the auditor's mutant goes red, "(30 vs 0)", and the unmutated leg stays green. **Only B6 re-runs**, from that
commit's own worktree and build folder, on the same licensed v2 population (`$E4Refs`, its `license.txt` and
`manifest.json`). B8, B9, X1 and B10 stay on the `e8c029ed` tree and `build-msvc`.

**The instrument hash.** The fix changes `tests/precache/e4_real.cpp`, and so its hash: on the box (CRLF) it is
`183de6de…` at `e8c029ed` and `b1cd172f…` at `1c145b53`. The change is in `Real102` (10.2) only. 10.3's timing code is
untouched, so the commissioned `PC-TIMING-1` receipt still names `e8c029ed`'s instrument (`183de6de…`). B8 and B9
keep running from `D:\_scratch\e4\src`, where `SUPERSLM_PRECACHE_TIMING_INSTRUMENT` points (the env script sets it
from `$E4Src`). B6r reads no timing receipt.

```powershell
. D:\_scratch\e4\src\tests\precache\e4_box_env.ps1
$G1Src = Join-Path $E4Work 'g1src'; $G1Build = Join-Path $E4Work 'build-g1'
if (-not (Test-Path $G1Src)) { git -C D:\SuperSLM fetch origin; git -C D:\SuperSLM worktree add --detach $G1Src 1c145b53c7cf2e81e0b0ad8535356d9763c68eeb }
git -C $G1Src rev-parse HEAD              # must print: 1c145b53c7cf2e81e0b0ad8535356d9763c68eeb
git -C $G1Src merge-base HEAD origin/main # must print: 1948e1f2ea4528790dc6701e26c47bda5739b003
git -C $G1Src status --porcelain          # must print nothing
$log = E4-Log 'B6r-build'
Remove-Item $log -ErrorAction SilentlyContinue
E4-Run $log cmake @('-S', $G1Src, '-B', $G1Build, '-G', 'Ninja', '-DBUILD_TESTING=ON', '-DCMAKE_BUILD_TYPE=Release',
                    '-DCMAKE_C_COMPILER=cl', '-DCMAKE_CXX_COMPILER=cl')
E4-Run $log ninja @('-C', $G1Build, 'superslm_precache_e1', 'superslm_precache_reference_header')
Remove-Item -Recurse -Force $env:SUPERSLM_PRECACHE_BLOB_DIR -ErrorAction SilentlyContinue
Push-Location $G1Build
E4-Run (E4-Log 'B6r-two-process') ctest @('-R', '^superslm_precache_real_two_process$', '-V', '--timeout', '1500')
Pop-Location
```

Pass: both build commands exit 0. Then everything B6 requires, from `build-g1`: the fixtures passed, the merge-base
`1948e1f2…`, capture and restore `10.1/E4 green` and `10.2/E4 green` with the ten `10.2 nestedNN: B = 128, s = 7, draw
= 3` lines, `two-process gate: capture exited, 21 hashes pinned, restore exited, 100 graded adopters`, and `100% tests
passed`. The restore's 10.2 line counts one check more than B6's (cloud: 4,378 against 4,377 on the stand-in). This
log, not B6's, is 10.1/10.2's E4 record. Resume: re-run (Ninja rebuilds only what is missing; the blobs are cleared
first).

### B7. Timing prompt manifest (1 min) — Dan; before T1

```powershell
. D:\_scratch\e4\src\tests\precache\e4_box_env.ps1
0..9 | ForEach-Object { $f = Join-Path $E4Prompts ('flat{0:D2}.ids' -f $_)
  '{0}  flat{1:D2}.ids' -f (Get-FileHash $f).Hash.ToLower(), $_ } |
  Set-Content -Encoding ascii $env:SUPERSLM_PRECACHE_TIMING_PROMPT_MANIFEST
Get-Content $env:SUPERSLM_PRECACHE_TIMING_PROMPT_MANIFEST
(Get-FileHash $env:SUPERSLM_PRECACHE_TIMING_INSTRUMENT).Hash.ToLower()   # the instrument SHA for T1
```

Pass: ten lines; the first two hashes are `ef7fae58fd1935ff…` (flat00) and `9278830bfa301291…` (flat01). Never edit the
file after T1: the receipt binds its hash.

### T1a, T1b. Commission the timing instrument — **SECOND SEAT: the debunker** (two pieces, about 12 + 6 box min)

The debunker owns the constructions (§8, F12): a **must-reject**, a restore mutant that runs prefill internally, which
the 1,000× bar must refuse; and a **must-accept**, the built restore. The shape below is a suggestion; the debunker may
change it, but each piece stays at 20 minutes or less.
- **T1a (≤ 12 min, need not run alone):** build the must-reject mutant: a mutant source worktree with
  `sslm_prefix_restore` running a prefill of the blob's length, built like B1 into its own directory.
- **T1b (≤ 6 min, ALONE):** slice `0-1` of `10.3/E4` on the mutant under a provisional receipt (red expected), then
  slice `0-1` on B1's build (the must-accept).
- Any re-run rule for B8 (for example, a named disturbance on the box) is the debunker's, written down in T1b's record
  **before** any B8 reading. Without one, a B8 reading stands as taken.

On PASS the debunker writes `D:\_scratch\e4\timing\PC-TIMING-1.txt`, 13 lines:

```
format=PC-TIMING-1
status=COMMISSIONED
commissioning_verdict=PASS
must_accept=PASS
must_reject=PASS
threshold=1000
box_id=home-3950x
threads=1
geometry=0p5:24x2x64x32768:B=128;1p5:28x2x128x32768:B=64
model_sha256=4b27e2d6e132d710a4b174d5ffc15e7705691ca37abf15eca65503803d0dba35
prompt_manifest_sha256=<SHA-256 of D:\_scratch\e4\timing\prompt-manifest.txt>
instrument_sha256=<B7's last line>
anchor=1948e1f2ea4528790dc6701e26c47bda5739b003
```

Readings taken before this exists are refused by the cell, so none can be acted on early.

### B8a–B8d. 10.3 paired timing, four slices — Dan; needs T1b; **each ALONE**

Eight prefill/restore pairs per prefix, one thread, rule B 128, restore including the digest. Each slice is its own
process and grades its own prefixes.

| Piece | Prefixes | Prompt tokens × 8 | Expected |
|---|---|---|---|
| B8a | `0-4` (159–501 tokens) | 16,392 | 5–9 min |
| B8b | `5-7` (1,000–1,506) | 29,712 | 8–15 min |
| B8c | `8-8` (1,801) | 14,408 | 4–8 min |
| B8d | `9-9` (2,000) | 16,000 | 5–9 min |

```powershell
. D:\_scratch\e4\src\tests\precache\e4_box_env.ps1
$env:SUPERSLM_PRECACHE_TIMING_PREFIXES = '0-4'          # B8a; '5-7' B8b; '8-8' B8c; '9-9' B8d
E4-Run (E4-Log ('B8-timing-' + $env:SUPERSLM_PRECACHE_TIMING_PREFIXES)) $E4Runner @('=10.3/E4')
```

One untimed warm-up pair runs first in each slice. Pass per slice: one line per prefix `10.3 receipt=<sha> runner=<sha>
box=home-3950x threads=1 geometry=0p5 B=128 flatNN pairs=8
bytes_including_digest=<1080 + 6144·L> prefill_ms=… restore_ms=… ratio95=[lower,upper]` with **lower ≥ 1000**, then
`10.3 slice flatAA-flatBB: N of N prefixes resolved at or above 1000x runner=<sha> receipt=<sha>` and `10.3/E4 green`.
E4's 10.3 needs all four slices green and B8e. Predicted (§7 10.3): prefill ~15–30 ms/token, restore ~1–12 ms, ratio ~1,200–4,000×.

**On a miss** (a `lower` below 1,000, or an interval that straddles it): the reading stands. **Do not re-run the
slice**: a second draw at the 95% interval, keeping the green one, would weaken the bar. Stop the timing pieces and
send the logs; a miss is §10 R1's trigger (the digest moves to a separate verb), a planner fold. The only exception is
a re-run rule the debunker wrote at T1b before any reading, and then both readings go to the planner.

### B8e. 10.3 coverage check over the four slices (1 min) — Dan; after B8a–B8d

One paired run proved ten prefixes in one process; the slices must add up to the same thing: ten distinct prefixes,
each `lower ≥ 1000`, one receipt, one runner, every slice N of N.

```powershell
. D:\_scratch\e4\src\tests\precache\e4_box_env.ps1
$log = E4-Log 'B8e-coverage'; Remove-Item $log -ErrorAction SilentlyContinue
$logs = '0-4', '5-7', '8-8', '9-9' | ForEach-Object { E4-Log ('B8-timing-' + $_) }
$lines = Select-String -Path $logs -Pattern '^10\.3 receipt=(\w+) runner=(\w+) .* (flat\d\d) .*ratio95=\[([\d.]+),'
$slices = Select-String -Path $logs -Pattern '^10\.3 slice flat\d\d-flat\d\d: (\d+) of (\d+) .* runner=(\w+) receipt=(\w+)'
$names = @($lines | ForEach-Object { $_.Matches[0].Groups[3].Value })
$checks = [ordered]@{
  'ten distinct prefixes 00-09' = (($names | Sort-Object -Unique) -join ',') -eq ((0..9 | ForEach-Object { 'flat{0:D2}' -f $_ }) -join ',') -and $names.Count -eq 10
  'every lower >= 1000'         = @($lines | Where-Object { [double]$_.Matches[0].Groups[4].Value -lt 1000 }).Count -eq 0
  'one receipt'                 = @(@($lines | ForEach-Object { $_.Matches[0].Groups[1].Value }) + @($slices | ForEach-Object { $_.Matches[0].Groups[4].Value }) | Sort-Object -Unique).Count -eq 1
  'one runner'                  = @(@($lines | ForEach-Object { $_.Matches[0].Groups[2].Value }) + @($slices | ForEach-Object { $_.Matches[0].Groups[3].Value }) | Sort-Object -Unique).Count -eq 1
  'four slices, each N of N'    = $slices.Count -eq 4 -and @($slices | Where-Object { $_.Matches[0].Groups[1].Value -ne $_.Matches[0].Groups[2].Value }).Count -eq 0
}
$checks.GetEnumerator() | ForEach-Object { $l = '{0}: {1}' -f $_.Key, $(if ($_.Value) { 'PASS' } else { 'FAIL' }); E4-Append $log $l; $l }
'runner = ' + (Get-FileHash $E4Runner).Hash.ToLower() | ForEach-Object { E4-Append $log $_; $_ }
```

Pass: five `PASS` lines, and the runner SHA in the logs equals the last line (the B1 build). A slice re-run under a
debunker rule leaves two readings in one log: send it rather than reading this check.

### W1. Cap-length 1.5B capture for 10.3's worst setting — **SPARE PC, through the long-leg drive kit**; about 60–90 min

This does not run on the home box (R12: one uncheckpointed prefill of 32,768 tokens at 1.5B). It runs on the spare PC
(i5-14400F) as one more piece of the long-leg kit, `tools/longleg_kit` (README, *E4's W1*). The piece builds
`superslm_precache_e1` at `e8c029ed68f6d3fb0a728673c9a41d16d433916d` with MSVC 14.34 Release and B1's flags, from the
drive's `e4w1.bundle`, then runs `=10.3/E4:worst` with `SUPERSLM_PRECACHE_WORST_PHASE=capture` on the drive's 1.5B
model.

1. Home box, with the drive plugged in (a few minutes). This refreshes the kit and adds W1's source and model. The
   long legs' source is not touched.
   ```powershell
   powershell -NoProfile -ExecutionPolicy Bypass -File D:\SuperSLM\tools\longleg_kit\fill_drive.ps1 -Drive F: -KitOnly
   ```
2. Spare PC: run `START.cmd -Toolchains msvc-asan -RetryCrashed -E4Worst` from the drive's root. W1 goes first in the
   queue, beside the ASan re-run.

Pass: in `results\SUMMARY.md`, the builds table has an `msvc-e4` row with status `ok`, and the piece
`msvc-e4__10.3-E4-worst` is **GREEN**. Its log `results\msvc-e4__10.3-E4-worst.attempt<N>.out.log` shows:
- `10.3 real 1.5B: rule B = 64 (engine 64, helpers 64), cap 32768, cap_pages 512, page 917504 bytes`;
- `10.3 worst capture cap=32768 geometry=1p5 28x2x128 B=64 bytes_including_digest=469763768 blob_sha256=<sha> prefill_ms=…`;
- `10.3/E4:worst green`.

The kit makes the piece GREEN only when the blob on the drive has exactly that size and SHA-256. It writes
`results\e4-w1\cap1p5.spx1`, `cap1p5.spx1.sha256` and `W1.txt` (the commit, the runner's SHA-256, the compiler and
the capture line). A crash leaves only `cap1p5.spx1.partial`, and the kit runs the piece again.

### B9. 10.3 worst setting: restore timing (≤ 5 min) — Dan; needs W1 and T1b; ALONE

When the drive is back, copy `F:\longleg\results\e4-w1\` as a binary copy, then check it before timing:

```powershell
. D:\_scratch\e4\src\tests\precache\e4_box_env.ps1
$w = Split-Path $env:SUPERSLM_PRECACHE_WORST_BLOB          # D:\_scratch\e4\worst
New-Item -ItemType Directory -Force $w | Out-Null
Copy-Item F:\longleg\results\e4-w1\* $w -Force
$log = E4-Log 'B9-worst-restore'
Get-Content (Join-Path $w 'W1.txt') | ForEach-Object { E4-Append $log $_; $_ }
$want = ((Get-Content (Join-Path $w 'cap1p5.spx1.sha256')) -split '\s+')[0]
$got = (Get-FileHash $env:SUPERSLM_PRECACHE_WORST_BLOB).Hash.ToLower()
"blob sha256 $got; W1 printed $want; bytes $((Get-Item $env:SUPERSLM_PRECACHE_WORST_BLOB).Length)" | ForEach-Object { E4-Append $log $_; $_ }
if ($got -ne $want) { throw 'cap1p5.spx1 differs from what W1 recorded: copy it again' }
$env:SUPERSLM_PRECACHE_WORST_PHASE = 'restore'
E4-Run $log $E4Runner @('=10.3/E4:worst')
```

Pass:
- `W1.txt` shows `commit=e8c029ed68f6d3fb0a728673c9a41d16d433916d`;
- the blob's SHA-256 equals W1's, at 469763768 bytes;
- the runner prints `10.3 receipt=<sha> runner=<B8's runner sha> box=home-3950x threads=1 worst cap=32768
  geometry=1p5 28x2x128 B=64 bytes_including_digest=469763768 blob_sha256=<W1's> restore_ms=… MiB=448.002 MiB/s=…`
  and then `10.3/E4:worst green`.

What B9 checks:
- The restore runs under the commissioned `PC-TIMING-1` receipt, so `runner=` is the box's runner, B8's.
- The blob's size must be exactly the header, the hidden state and 32,768 canonical rows.
- Each of the three timed restores must succeed against the box's mapped 1.5B artifact. A restore validates the blob,
  digest included, against that model and the producer (engine version, CPU), so a blob from another model or source is refused,
  not timed.
- The first restore must re-save the identical bytes.

W1's runner SHA-256 and the box's are recorded side by side, not compared: they are two MSVC builds of one commit on
two machines. The figures are recorded, not graded: they are the plugin cost model's per-MiB figures (R5).

### X1 (done in the cloud). 4.3's GNU and Clang compiler jobs

`release-1p14/e4-cross-compiler-e8c029ed/GNU.*` and `Clang.*` (receipt, job, runner, compiler, cmake record, logs,
blob), captured by `tests/precache/capture_compiler_job.py` at the pinned commit; the GNU and Clang blobs
are byte-equal. Copy all 17 files of the folder (16 job files and SHA256SUMS.txt) unchanged (a binary copy, no line-ending conversion) into
`D:\_scratch\e4\xc\`, then check them:

```powershell
Get-Content D:\_scratch\e4\xc\SHA256SUMS.txt | ForEach-Object { $h, $n = $_ -split '\s+', 2
  if ((Get-FileHash (Join-Path D:\_scratch\e4\xc $n)).Hash.ToLower() -ne $h) { "MISMATCH $n" } }   # prints nothing
```

### B10. 4.3: the MSVC job and the three-way compare (≤ 15 min) — Dan; needs B0 and X1's files

```powershell
. D:\_scratch\e4\src\tests\precache\e4_box_env.ps1
$xc = Join-Path $E4Work 'xc'; $jb = Join-Path $E4Work 'build-4.3-msvc'
$env:CC = 'cl'; $env:CXX = 'cl'      # the job configures with no compiler named; never let a gcc or clang on PATH overwrite X1's files
Remove-Item -Recurse -Force $jb -ErrorAction SilentlyContinue
E4-Run (E4-Log 'B10-4.3-msvc-job') python @((Join-Path $E4Src 'tests\precache\capture_compiler_job.py'),
  '--source', $E4Src, '--build-dir', $jb, '--bundle-dir', $xc, '--generator', 'Ninja')
$env:SUPERSLM_PRECACHE_CROSS_COMPILER_DIR = $xc
$env:SUPERSLM_PRECACHE_COMPILER_PHASE = 'compare'
$env:SUPERSLM_PRECACHE_RUNNER_EXE = Join-Path $jb 'superslm_precache_e1.exe'
E4-Run (E4-Log 'B10-4.3-compare') $env:SUPERSLM_PRECACHE_RUNNER_EXE @('=4.3/E4')
```

Pass: the job prints `captured MSVC 19.34.<n> job e8c029ed68f6d3fb0a728673c9a41d16d433916d: <runner sha>`; the compare prints `4.3/E4 green`, exit 0:
three receipts from three distinct runners, one source revision, and the MSVC, GNU and Clang `SPX1` blobs byte-equal.
Resume: re-run (it deletes its build directory first). If the job refuses "requires a clean source revision", run
`git -C D:\_scratch\e4\src status` and send it.

### B11. 6.6's matvec mutant on the rule-B leg (≤ 15 min) — Dan

**Optional; skip unless the coverage auditor asks for the MSVC build.** The cloud killed this mutant on both legs, the
rule-B leg on a stand-in with the real 0.5B widths (hidden 896, intermediate 4,864; `e4-cloud-2026-10-22.md` §3).

```powershell
. D:\_scratch\e4\src\tests\precache\e4_box_env.ps1
$m = Join-Path $E4Work 'msrc'; $mb = Join-Path $E4Work 'build-mutant'
if (-not (Test-Path $m)) { git -C D:\SuperSLM worktree add --detach $m e8c029ed68f6d3fb0a728673c9a41d16d433916d }
git -C $m checkout -- .
python (Join-Path $E4Src 'tests\precache\apply_e4_mutant.py') $m matvec-last-short
$log = E4-Log 'B11-matvec-mutant'
E4-Run $log cmake @('-S', $m, '-B', $mb, '-G', 'Ninja', '-DBUILD_TESTING=ON', '-DCMAKE_BUILD_TYPE=Release', '-DCMAKE_C_COMPILER=cl', '-DCMAKE_CXX_COMPILER=cl')
E4-Run $log ninja @('-C', $mb, 'superslm_precache_e1')
E4-Run $log (Join-Path $mb 'superslm_precache_e1.exe') @('=6.6/E4', '=6.6/E4:rule-b')
git -C $m checkout -- .
```

Pass (the mutant is killed): both `6.6/E4 RED` and `6.6/E4:rule-b RED`, each with `restored parallel token differs at
…`. Then `git worktree remove --force D:\_scratch\e4\msrc`.

### A1. E4 discrimination re-audit — **SECOND SEAT: the coverage auditor** (no box time)

On B2, B6, B8a–B8d, B9, B10, B11's logs and the cloud record. E4 is not green before it clears.

## Totals

| Who | Pieces | Box minutes (estimate) |
|---|---|---|
| Dan, test-author pieces on the home box | B0 3, B1 15, B2 3, B3g 12, B3 12, B4 8, B5 2, B6 10, B7 1, B8a–d 39, B8e 1, B9 3, B10 15 (B11 optional, +15) | **about 124** |
| Dan, re-run after the L1 refusal (§15.33) | B3g 12, B3 12, B4 8, B5 2, B6 10 | **about 44**, plus L1's ~10 = 54 |
| Debunker, on the box | L1 ~10, T1a ~12, T1b ~6 (alone) | about 28 |
| Spare PC (long-leg kit, `-E4Worst`) | W1 | 60–90 (not home-box time) |
| Dan, after A1's G1 | B6r: build ≤ 12, run ≤ 10 | **about 22** |
| Coverage auditor | A0 (before L1), A1 | none |

E4's plan row estimated ~45 min; it did not count the reference regeneration, the licence, the commissioning or the
4.3 job builds.

## Blockers

- B6 cannot pass until A0 and L1 exist; B8 and B9 cannot run until T1 exists. Both are second-seat work that has not been
  requested yet.
- B9 needs W1 from another machine.
- If `origin/main` moves and is merged into the branch before B6, the anchor changes and B3–B6 and L1 must be redone at
  the new anchor; B10 and X1 must then use the new pinned commit too.
