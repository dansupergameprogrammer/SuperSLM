# E5 cloud record: Linux GCC 13.3, 2026-10-23

The cloud half of E5. Source tree: `f7411b8b` (code identical to `6337447f`; the delta is docs only).
Toolchain: gcc 13.3.0 (Ubuntu 24.04), Release, 4 cores. The box half is in
`e5-box-runsheet-2026-10-23.md`.

## (a) Merge-base

`git merge-base HEAD origin/main` = `1948e1f2`, which is `origin/main`'s tip after a fetch on 2026-10-23.
Main has not moved, so E5 proceeds.

## 9.9: captured and pinned in the cloud

`tools/build_precache_v1p13p1_captures.sh` built v1.13.1 (`d697927f`), captured twice and compared the two
captures (identical). Pinned in `6337447f`:
`tests/precache/reference/v1.13.1_pkv_def_seqsaves.ref` plus 8 `pins/v1.13.1_pkv_def_*.zrl`.
Header: `writer=v1.13.1 commit=d697927f… fixture=pkv_def sha256=994b58b4… B=128`.

The cell `9.9/E5` (`superslm_precache_unpinned`) is **green on GCC: 136 checks, 0 failed.** MSVC is box
piece E3.

The coverage auditor still owes 9.9's two mutants. They have not been run.

## 7.17: the test author's construction dry run

`7.17/run_717_constructions.py` builds each case from the anchor doc (`git show 5d5ca58:docs/api.md`) and
grades it with `check()` against this tree's `sslm_abi.h`:

| case | verdict |
|---|---|
| anchor | REFUSED: RESTORE_PRODUCER_MISMATCH 0 statements, floor 2 (**red on the anchor**) |
| must-accept | ACCEPTED: 4 statements |
| reject-one-below-floor | REFUSED: 1 statement, floor 2 |
| reject-kv-30 | REFUSED at :787: KV_BUDGET_EXCEEDED stated at 30, the X-macro has 29 |
| reject-new-31 | REFUSED at :860: RESTORE_PRODUCER_MISMATCH stated at 31, the X-macro has 30 |
| reject-paired-31 | REFUSED at :862: same, paired form |
| reject-kv-reworded | REFUSED: KV_BUDGET_EXCEEDED 1 statement, floor 2 |
| reject-unpaired | REFUSED at :865: "ordinal 12" with no status name, an unpaired statement |

The script ends with `8 constructions, 0 unexpected` and exits 0.

Rev 1.18.14 (§15.34) ruled on the floor disagreement this record first raised. The must-accept is the two-statement
case (one own-line statement and one paired statement), and the floor stays at 2. The one-statement document, which
was §8's must-accept as first worded, is the sixth must-reject: it is below the floor. The script follows the ruling:
`accept-two` became `must-accept`, and `accept-one` became `reject-one-below-floor`, expected refused.

On the tree, 7.17 is green: 4 stated ordinals match (`api.md:529`, `:813`, `:1139`, `:1150`). That
agrees with rev 1.18.4's "green when written".

## 7.22

Green: 9 mention units carry "unreleased until 1.16.0", and 19 label sites read 1.16.0.

7.22 had been red since `198ded47`. The cause was `docs/release/e4-box-2026-10-22/plan.md`, a plan copy
whose GPU lines name other releases. The E5 packet removes the copy.

## (d) Full CPU suite

| run | result |
|---|---|
| `superslm_tests` | **123,818 checks, 0 failures** |
| `superslm_precache_e1` (all 58 cells) | 205,013 checks; 5 red cells, 7 failures, **all environmental** (below) |
| `superslm_precache_unpinned` (E2, E3, E5: 23 cells) | **23 green, 34,292 checks, 0 failures** |
| ctest (34, `superslm_tests` run above) | 30 pass, 4 fail, **all environmental** (below) |

The unpinned cells, all green:
- 4.7/E2 and 9.8/E2 on pkv_def, pkv_odd, pkv_32k, pkv_qk and synthetic0p5
- 8.13/E2 and 7.12/E2:pool-kind
- 4.7/E3 and 9.8/E3 on the same five fixtures
- 9.9/E5

The e1 reds:
- 2.13/E2, 2.14/E2, 7.15/E2 and 11.6/E2 read E0's box artifacts, which are not in the cloud.
- 5.7/F2:real-gpu needs a D3D12 device.

All five were green on the box on 2026-10-22.

The ctest reds:
- `superslm_precache_real_two_process`, `_e2_reference` and `_e3_reference` need box references and
  environments. They are box-only.
- `t2919_g5_schema_compiler_tests`: 72 passed, 20 skipped, 1 failed. The failure is
  `ModuleNotFoundError: tokenizers`, a Python module missing from the cloud image. It is not a code
  result.

Not run here, because they are GPU or box only: 9.10, 5.7/F2, and every MSVC leg.

## 9.5: pin tool dry run (rev 1.18.14, §15.34 (2))

`tools/build_precache_9p5_pins.sh WORK --allow-dirty` was run on a scratch worktree of this tree with
`version-bump-1.14.0.patch` applied. It took 1 min 8 s. All four producers read 1.14.0, the two runs were
bit-identical, and the rule B was asserted on each fixture:

| fixture | D | cap | B read | L | last page |
|---|---|---|---|---|---|
| pkv_def | 48 | 4096 | 128 | 141 | 13/128 |
| pkv_odd | 48 | 4100 | 256 | 269 | 13/256 |
| pkv_32k | 48 | 32768 | 256 | 269 | 13/256 |
| pkv_qk | 128 | 4096 | 64 | 77 | 13/64 |

A mutant tool rule (B capped at 64) was refused on pkv_def, pkv_odd and pkv_32k, and the run exited 1 with no
manifest. No pins are committed: R2 writes them from R1's commit.

## R1 and R2 (after the box receipts `80fcd3a1`, `f8eca988`)

**R1, `7585adac`.** It applies `version-bump-1.14.0.patch` dated 2026-10-23, including the review fix to the release
note's opening. Every binary was relinked first. A stale `EXCLUDE_FROM_ALL` binary that still wrote producer 1.13.1
had turned 7.12/E2:independent-parser red; it is green after the relink. Results:
- `check_version_header` matches, and `check_version_values --expect 1.14.0` reports 0 failures.
- 7.17 and 7.22 are green.
- `superslm_tests`: 123,818 checks, 0 failures.
- Unpinned cells: 23 of 23 green, including 9.9/E5 with 136 checks.
- e1: the same 5 environmental reds as before the bump.
- ctest: 30 of 34 pass; the same 4 environmental failures.

**R2.** `tools/build_precache_9p5_pins.sh` ran at `7585adac` on a clean tree and took 1 min 6 s. Both runs were
identical and all four producers read 1.14.0. It wrote `tests/precache/reference/pins-9.5/`: four blobs and
`manifest.txt`, which records the commit as `7585adac…`. Each blob's B, L and SHA-256 equal the dry run's.

The box's `tests/precache-gpu/captures/.gitattributes` (`* -text`):
- All six committed capture files hash to the G2 log's values, and `captures.txt` is LF.
- 9.10's reader opens the file in binary mode. It strips a trailing `\r` from record lines but not from the header,
  so a CRLF checkout would fail the header's `fixture=` match. `-text` is what prevents that.
- 7.22 sweeps `captures.txt` (a `.txt`) and passes. It skips the attributes file by suffix.

`pins-9.5/` carries the same `* -text`.
