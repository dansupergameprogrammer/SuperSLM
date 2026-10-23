# E5 box run sheet: 1.14.0's release gates (SuperSLM 1.14.0), 2026-10-23

Plan: `plans/superslm-prefix-precache/plan.md` rev 1.18.14, §9 E5 and M2, §7 7.17, 7.22, 9.5, 9.9, 9.10, §8 (7.17's
and 7.22's registrations, the 9.5 pin), and errata §15.20–§15.22 and §15.28 (7.17's widened locator and floor 2;
9.9's outcome form and mid-token reading; 7.22's unit; the full-clone ancestry receipt), and §15.34 (7.17's must-accept;
9.5's geometries and pin tool). Written by the test author.
The plan text is not edited here, and a copy of it is not kept in this tree: the GPU label check (7.22) sweeps every
tracked text file, and the plan names 1.14.0 beside the GPU verbs. That is why the E4 packet's `plan.md` copy was
removed in the commit that adds this sheet.

**Pinned commit:** `ce82c898` on `claude/precache-1p14-post-m1`: the release edits (`f7411b8b`) plus this packet,
which also removes the E4 `plan.md` copy. At `f7411b8b` itself, 7.22 is red on that copy. Every later commit
changes only `docs/release/` or adds R2's pin tool (`tools/build_precache_9p5_pins.sh` and
`tests/precache/reference/pin_9p5_spx1.cpp`). No box piece builds either one, so the pieces stay valid at `ce82c898`. The version still reads **1.13.1**: the bump to 1.14.0 is the held final commit (piece
R1), made only when every gate below is clear (PC-7). Every piece builds this commit in a detached worktree, even if
the branch has moved.

## Where E5 stands (cloud, 2026-10-23)

| Item | State |
|---|---|
| (a) Merge-base | `git merge-base HEAD origin/main` = `1948e1f2`, the recorded anchor; `origin/main` has not moved. No regeneration is owed |
| 9.9's capture and pin | **Done in the cloud.** `tools/build_precache_v1p13p1_captures.sh` is a Linux script by design. It built v1.13.1 (`d697927f`), captured the 8 cases twice, found both runs bit-identical, and pinned them (`6337447f`: `tests/precache/reference/v1.13.1_pkv_def_seqsaves.ref`, `pins/*.zrl`). 9.9/E5 on GCC 13.3: green, 136 checks |
| 7.17 | The check exists (M1) and is **green on this tree** (4 statements). Rev 1.18.4 recorded that it is green when written, because `docs/api.md` has stated status 30 since `31e41d82`. It is **red on the anchor doc** (`5d5ca58`) for its own reason: 0 statements of the new status, floor 2. `7.17/run_717_constructions.py` dry-runs §8's constructions for the debunker. Rev 1.18.14 (§15.34) ruled: the must-accept is the two-statement case (one own-line, one paired) and the floor stays at 2; the one-statement document is the sixth must-reject. The dry run prints 8 verdicts, 0 unexpected, and exits 0 |
| 7.22 | Green on this tree (9 mention units, 17 label sites at 1.16.0), once the E4 packet's `plan.md` copy is removed |
| Release edits | `f7411b8b`: CHANGELOG, `docs/releases/1.14.0.md` (names `SSLM_HAS_PREFIX_SAVE_ABI` and the GPU twins as exported but unreleased until 1.16.0). `docs/api.md` already states status 30 and documents the macro. The held bump is `version-bump-1.14.0.patch` |
| Full CPU suite, Linux GCC 13.3 | `cloud-suite-2026-10-23.md` |

## Order and dependencies

```
E0 worktrees ─ E1 MSVC CPU build ─┬─ E2 7.17 + 7.22 + version checks on MSVC
                                  └─ E3 9.9 on MSVC (release tree)
E0 ─ G1 v1.13.1 GPU build ─ G2 9.10 capture on the RTX 2080 SUPER ─┐
E0 ─ G3 release-tree GPU build (cells) ────────────────────────────┴─ G4 9.10 on the RTX 2080 SUPER
W1–W4: G1–G4 on the RX 7900 XTX           [WORK MACHINE]
Seats: debunker commissions 7.17 (needs E2) · code reviewer, coverage auditor, adversary strike §3.3–3.5 · publish-gate audit
R1 version bump (held, PC-7) ─ R2 9.5 pin on the 1.14.0 build ─ R3 receipts ─ M2
```

**Rules for every piece.** Open a fresh PowerShell window for each piece. Every piece writes its log under
`D:\_scratch\e5\logs`. A piece can be resumed by re-running it. None of these pieces is a timing piece, so none has to
run alone. GPU pieces need the box's RTX 2080 SUPER as adapter 0 (`SSLM_GPU_ADAPTER_INDEX=0`), as in E2's GPU runs.

```powershell
# every piece after E0 starts with these lines
. D:\_scratch\e5\src\tests\precache\e4_box_env.ps1      # MSVC 14.34 (vcvars) and the E4 helpers: E4-Log, E4-Run, E4-Append
$E5 = 'D:\_scratch\e5'; $E5Src = "$E5\src"; $E5Logs = "$E5\logs"; New-Item -ItemType Directory -Force $E5Logs | Out-Null
$FX = 'D:\_artifacts\SuperSLM\t3035-precache-refs\fixtures'   # pkv_def 994b58b4..., the fixture 9.9's capture names
function E5-Log([string]$n) { Join-Path $E5Logs "$n.log" }
```

`e4_box_env.ps1` is used only for MSVC 14.34 and its log helpers. Its E4 paths are not read by any piece here.

## Pieces

### E0. Worktrees and checks (3 min) — Dan

```powershell
cd D:\SuperSLM
git fetch origin
git rev-parse --is-shallow-repository                  # must print: false
New-Item -ItemType Directory -Force D:\_scratch\e5 | Out-Null
if (-not (Test-Path D:\_scratch\e5\src))  { git worktree add --detach D:\_scratch\e5\src ce82c898 }
if (-not (Test-Path D:\_scratch\e5\v1131)) { git worktree add --detach D:\_scratch\e5\v1131 v1.13.1 }
git -C D:\_scratch\e5\src rev-parse HEAD               # must print: ce82c898
git -C D:\_scratch\e5\src merge-base HEAD origin/main  # must print: 1948e1f2ea4528790dc6701e26c47bda5739b003
git -C D:\_scratch\e5\v1131 rev-parse HEAD             # must print: d697927f6d5ce5fa73ecbb8660020a425518b923
git -C D:\_scratch\e5\src status --porcelain           # must print nothing
Get-FileHash D:\_artifacts\SuperSLM\t3035-precache-refs\fixtures\pkv_def.sslm
#   994B58B49D3660BCCFDD59DC6315028EA0203F37F4705D70EB7CE4DBECDB1DB3
```

Pass: every line prints what its comment says. **If the merge-base is not `1948e1f2`, stop.** `main` has moved, so the
references must be regenerated and licensed again at the new anchor before anything is graded (§8, D-SLM8013).

### E1. MSVC CPU build of the release tree (≤ 15 min) — Dan

```powershell
$log = E5-Log 'E1-build'; Remove-Item $log -ErrorAction SilentlyContinue
E4-Run $log cmake @('-S', $E5Src, '-B', "$E5\build-msvc", '-G', 'Ninja', '-DBUILD_TESTING=ON', '-DCMAKE_BUILD_TYPE=Release',
                    '-DCMAKE_C_COMPILER=cl', '-DCMAKE_CXX_COMPILER=cl')
E4-Run $log ninja @('-C', "$E5\build-msvc", 'superslm_precache_unpinned')
```

Pass: both commands exit 0. Resume: re-run (Ninja builds only what is missing).

### E2. 7.17, 7.22 and the version checks on the MSVC tree (2 min) — Dan; needs E1

```powershell
Push-Location "$E5\build-msvc"
E4-Run (E5-Log 'E2-ctest') ctest @('-R', '^superslm_precache_(api_ordinals|api_ordinals_locator|gpu_prefix_labels|version_header|version_header_commission)$', '-V')
Pop-Location
E4-Run (E5-Log 'E2-717-dryrun') python @("$E5Src\docs\release\e5-box-2026-10-23\7.17\run_717_constructions.py", $E5Src)
```

Pass:
- ctest `100% tests passed`, with 5 tests, including:
  - `check_api_ordinals: 4 stated ordinal(s) match the X-macro`;
  - `test_check_api_ordinals: PASS`;
  - `check_gpu_prefix_labels: 9 mention unit(s) ... label site(s) read 1.16.0`, with no problem lines;
  - `check_version_header: version matches`.
- The dry run prints the same eight verdicts as the cloud (`cloud-suite-2026-10-23.md`, 7.17). The box runs the
  script as pinned at `ce82c898`, from before rev 1.18.14's ruling. Its names `accept-one` and `accept-two` are the
  later `reject-one-below-floor` and `must-accept`, and its `<- differs from the section 8 wording` note on
  `accept-one` is the expected refusal under §15.34, not a failure. The verdicts are identical.

### E3. 9.9 on the release tree, MSVC (≤ 5 min) — Dan; needs E1

```powershell
$env:SUPERSLM_PAGED_KV_FIXTURE_DIR = $FX
Push-Location $E5Src
E4-Run (E5-Log 'E3-9.9') "$E5\build-msvc\superslm_precache_unpinned.exe" @('=9.9/E5')
Pop-Location
```

Pass:
- `pkv_def: rule B = 128 (engine 128, helpers 128), cap 4096, cap_pages 32, last page holds 128`;
- `[E5] 9.9/E5 green (136 checks, 0 failed)`, exit 0.

The pins were captured on GCC. A green here also shows that MSVC's 1.14.0 tree restores them, re-saves them
byte-identical and decodes the pinned continuations. Run it from `$E5Src`: the cell reads the pins from
`tests\precache\reference`.

### G1. v1.13.1 GPU build for 9.10's capture leg (≤ 20 min) — Dan

```powershell
$V = "$E5\v1131"; $log = E5-Log 'G1-v1131-build'; Remove-Item $log -ErrorAction SilentlyContinue
E4-Run $log cmake @('-S', $V, '-B', "$V\build", '-G', 'Visual Studio 17 2022', '-A', 'x64', '-T', 'v143,version=14.34',
                    '-DSUPERSLM_BUILD_GPU=ON', '-DBUILD_TESTING=OFF')
E4-Run $log cmake @('--build', "$V\build", '--config', 'Release', '--target', 'superslm', 'superslm_gpu_shaders')
E4-Run $log cmd @('/c', "$E5Src\tests\paged-kv-gpu\build_paged_kv_gpu.bat", ('"--out=' + "$E5\pkvg" + '"'), ('"--capture910-tree=' + $V + '"'), 'capture910')
```

Pass:
- the two cmake commands exit 0;
- `$V\build\Release\superslm.lib` and `$V\build\Release\shaders\*.cso` exist;
- the batch file prints `LINKED (v1.13.1): ...\v1131\pcg_cell_9_10_capture.exe`.

A `REFUSED: --capture910-tree ... is at ..., not v1.13.1` line means the worktree is not at `d697927f`: stop.
Each `--name=value` argument is wrapped in its own double quotes because cmd splits unquoted arguments at `=`. Without the quotes, the batch file treats the flag and the path as two targets (box run, 2026-10-23). The `'vswhere.exe' is not recognized` line comes from VsDevCmd and is harmless.
**Unverified here:** the cloud cannot build D3D12, so these commands are untested. If one fails, send its log and
the test author fixes the sheet.

### G2. 9.10 capture on the RTX 2080 SUPER (≤ 10 min) — Dan; needs G1

```powershell
$env:SSLM_GPU_ADAPTER_INDEX = '0'
E4-Run (E5-Log 'G2-9.10-capture-2080s') "$E5\pkvg\v1131\pcg_cell_9_10_capture.exe" @("--fixture-dir=$FX", "--out=$E5\captures910")
Get-ChildItem "$E5\captures910" -Recurse -File | Get-FileHash | ForEach-Object { E4-Append (E5-Log 'G2-9.10-capture-2080s') ("{0}  {1}" -f $_.Hash.ToLower(), $_.Path) }
```

Pass: exit 0, and `$E5\captures910\v1.13.1_10de_<device>\` holds `captures.txt` and five `.bin` blobs. The header line
is `writer=v1.13.1 commit=d697927f... adapter=... fixture=994b58b4...`. Five case lines follow:
`slm5_dedicated_resting`, `slm5_dedicated_mid_token`, `slm6_pooled_resting`, `slm6_pooled_mid_token` and
`slm6_pooled_adopted_before_first_finish`. **These files are the pin.** They are committed with the receipts (R3)
under `tests/precache-gpu/captures/`, exactly as captured.

### G3. Release-tree GPU build and the 9.10 cell (≤ 20 min) — Dan; needs E0

```powershell
$log = E5-Log 'G3-release-gpu-build'; Remove-Item $log -ErrorAction SilentlyContinue
E4-Run $log cmake @('-S', $E5Src, '-B', "$E5Src\build", '-G', 'Visual Studio 17 2022', '-A', 'x64', '-T', 'v143,version=14.34',
                    '-DSUPERSLM_BUILD_GPU=ON', '-DBUILD_TESTING=OFF')
E4-Run $log cmake @('--build', "$E5Src\build", '--config', 'Release', '--target', 'superslm', 'superslm_gpu_shaders')
E4-Run $log cmd @('/c', "$E5Src\tests\paged-kv-gpu\build_paged_kv_gpu.bat", ('"--out=' + "$E5\pkvg" + '"'), 'e5')
```

Pass: the batch file prints `LINKED` for `pcg_cell_9_10`. **Unverified here**, as G1 is.

### G4. 9.10 on the RTX 2080 SUPER (≤ 10 min) — Dan; needs G2 and G3

```powershell
$env:SSLM_GPU_ADAPTER_INDEX = '0'
E4-Run (E5-Log 'G4-9.10-2080s') "$E5\pkvg\bin\pcg_cell_9_10.exe" @("--fixture-dir=$FX", "--captures=$E5\captures910")
```

Pass: `9.10` green, exit 0. Each of the five captures must do three things on the 1.14.0 release build: restore, re-save
byte-identical to its pinned bytes, and continue to its pinned 64 tokens. A line `no v1.13.1 capture is pinned for
adapter ...` means G2's folder is not where `--captures` points.

### W1–W4. G1–G4 on the RX 7900 XTX — **WORK MACHINE**

Run the same four pieces on the work machine with the RX 7900 XTX as the adapter, setting `SSLM_GPU_ADAPTER_INDEX` to
its index. Its captures land in `captures910\v1.13.1_1002_<device>\`. W1's v1.13.1 build may be copied from the box
instead of rebuilt, because its CPU library and shaders are adapter-independent. W2's capture and W4's grading must run
on the 7900 XTX itself. Bring back `captures910\v1.13.1_1002_*\` and the W2 and W4 logs.

### R1. The version bump (held; PC-7) — test author, cloud; only after every gate below is clear

Apply `version-bump-1.14.0.patch` (this folder) with the release date in place of `@RELEASE_DATE@`, as one commit.
The commit sets `project(superslm VERSION 1.14.0)`, `SUPERSLM_VERSION_MINOR 14` and `_PATCH 0`, the install consumer's
`find_package(superslm 1.14.0)` and README's current-release paragraph, and it dates the CHANGELOG entry and the
release note. It also carries the code review's one blocking fix (review of `6337447f..50fe019b`): the release
note's opening now states the behavior change plainly. Model mapping refuses a model with colliding or zero schema
name hashes, and the caller gets `SSLM_ARTIFACT_REJECTED` (`SSLM_DEVICE_LOST` on the GPU) and no handle. The fix sits
in the patch rather than the tree so that `ce82c898` stays the box's pin. The receipts in the same session are:
- `python tests/ci/check_version_header.py`: matches;
- `python tests/precache/check_version_values.py --expect 1.14.0`: 0 failures;
- 7.17 and 7.22 green.

In the cloud on the patch, at this sheet's tip: all three pass, and 7.22 reads 9 units and 19 sites. M2 then re-runs 7.17, 9.9, 9.10 on both GPUs, and 7.22 on the squash.

### R2. The 9.5 blob pin (after R1, ≤ 5 min) — test author, cloud

§8: "At E5, one 1.14.0 `SPX1` blob per geometry is pinned for 9.5". F4 runs 9.5 on 1.16.0 against them: a blob
written by 1.14.0 must be refused there with `SSLM_RESTORE_PRODUCER_MISMATCH`. The blob carries its producer version,
so it can be written only by the 1.14.0 build, after R1. Rev 1.18.14 (§15.34) names the geometries: pkv_def, pkv_odd,
pkv_32k and pkv_qk.

`tools/build_precache_9p5_pins.sh` (Linux) drives `tests/precache/reference/pin_9p5_spx1.cpp`. Per fixture it:
- asserts `sslm_kv_page_positions == PageRuleB(D, cap, 1)`, with the rule computed in the tool, and records the B it read;
- pins a budget prefix of `L = B + 13`, not a multiple of B, so the last page is partial;
- writes the raw blob and a `manifest.txt` line with the fixture's SHA-256, D, cap, B, L, the blob's SHA-256 and the
  build commit.

The script refuses unless CMake and `sslm_version.h` read 1.14.0 and the tree is clean. It runs the pin twice, requires
the runs bit-identical, and reads each producer independently (`check_version_values.py --expect 1.14.0 --blob`).
Output goes to `tests/precache/reference/pins-9.5/`.

```bash
# on the release tree, at R1's commit
tools/build_precache_9p5_pins.sh "$WORK"
git add tests/precache/reference/pins-9.5   # commit: [skip ci] E5 R2: 9.5's 1.14.0 SPX1 pins
```

Pass: `check_version_values: 0 failure(s) against 1.14.0`, four manifest lines reading B=128, 256, 256, 64 (pkv_def,
pkv_odd, pkv_32k, pkv_qk), and `pins written to`.

Dry run in the cloud, on this sheet's tree with the R1 patch applied (`--allow-dirty`, nothing copied): green in about
1 min, with B=128/256/256/64, L=141/269/269/77 and the four producers reading 1.14.0. A mutant rule (B capped at 64) is
refused on pkv_def, pkv_odd and pkv_32k, and the run exits 1 with no manifest. The committed pins come from R1's
commit, so their SHA-256s are R2's to record, not this dry run's.

### R3. Receipts (5 min) — Dan or the coordinator

Commit these to the branch under `docs/release/e5-box-2026-10-23/box-results/`, with SHA256SUMS:
- the box logs E0–E3 and G1–G4, and the work machine's W1–W4;
- `captures910\` under `tests/precache-gpu/captures/`, as captured;
- the R2 pins.

## What each seat owes

| Seat | Owes | Needs |
|---|---|---|
| Debunker | **Commission 7.17** (§8, as rev 1.18.14 restates it): its own constructions, independent of `7.17/run_717_constructions.py`. The must-accept is the two-statement case; the one-statement document is the sixth must-reject (below the floor of 2). Then 7.22's must-accept and must-rejects at the publish gate (§8, 7.22) | E2's log; this folder |
| Code reviewer | Final review of `6337447f` (9.9 pins), `f7411b8b` (release edits), this packet and R1's bump. The §3.8 header sentence (audit N8). Its part of the strike on §3.3–3.5 | the branch |
| Coverage auditor | **9.9's mutants**: one `SSB6` header field moved in 1.14.0's writer, and the reader's exact-size rule tightened. Both are owed red on 9.9/E5, and neither has been run yet. The 9.10 audit on both GPUs. **The publish-gate audit**, including 7.22's label check. Its part of the strike on §3.3–3.5 | E3, G4, W4; the cloud suite record |
| Adversary | The strike on the shipped contract sentences, §3.3–3.5, against `docs/api.md` and the release note as they ship | `f7411b8b` |
| Test author | R1 when the gates are clear; R2 right after R1; re-runs at M2 | the seats above |
| Principal | After M2, the tag `v1.14.0` on M2's squash commit, then `git rev-parse v1.14.0^{commit}` equal to the recorded SHA | M2's receipt |

## Totals

| Who | Pieces | Minutes (estimate) |
|---|---|---|
| Dan, home box | E0 3, E1 15, E2 2, E3 5, G1 20, G2 10, G3 20, G4 10, R3 5 | **about 90** |
| Work machine (RX 7900 XTX) | W1–W4 | about 60 (W1 15 if the v1.13.1 build is copied) |
| Cloud (test author) | R1, R2, re-runs at M2 | about 30 |
| Seats | 7.17 commissioning, the strike, the publish-gate audit, the final review, 9.9's mutants | no box time |

## Blockers

- G1 and G3 are unverified in the cloud (no D3D12 there). Expect a first-run fix.
- R1 waits on every gate, and R2 waits on R1.
- If `origin/main` moves before M2, E0's merge-base check fails. The references are then regenerated and licensed
  again before any grading (§8).
