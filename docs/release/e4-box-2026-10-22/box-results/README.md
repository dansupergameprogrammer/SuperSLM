# E4 on the home box: results, 2026-10-22 (PDT)

Home box: AMD Ryzen 9 3950X, Windows 10, MSVC 14.34 (19.34.31937) Release, Ninja, a full clone of SuperSLM. The
pieces follow `../e4-box-runsheet-2026-10-22.md` (at `198ded47`, then at `efab3d1a` for the re-run after the L1
refusal). Pinned commit `e8c029ed`, generator commit `47bdcc9e`, reference anchor `origin/main` = `1948e1f2`.

Contents:
- `logs/`: every piece's log, UTF-8, written by `E4-Run`. `logs/v1bound/` holds the refused first population's
  B3–B5 logs.
- `pieces/`: each piece's script as run, with the console capture.
- `l1/`: the debunker's licence record:
  - `L1-verdict.md`, the refusal and then the issue;
  - the audit scripts and their JSON output;
  - the construction tool;
  - the issued `manifest.json` and `license.txt`.
- `t1/`: the debunker's timing commissioning:
  - `T1-record.md`, with the re-run rule written before any reading;
  - the mutant script;
  - `PC-TIMING-1.txt` and `prompt-manifest.txt`.
- `b10-msvc-job/`: the text files of 4.3's MSVC job. The binaries stay off the branch, as X1's do: runner
  `54bf1e35…3006`, compiler `36d32d60…763d`, blob `e869127a…b314`.

Runners: B1 `superslm_precache_e1` `aba74f21…e86e`; generator at `47bdcc9e` `eccb8035…4b1f`; the T1 mutant
`0db0e1cc…f019`.

## Pieces

| Piece | Result | Key figures |
|---|---|---|
| B0 | PASS | full clone; HEAD `e8c029ed`; merge-base `1948e1f2`; engine sources identical to the anchor; clean tree; 0.5B `4B27E2D6…0DBA35`, pkv_def `994B58B4…1DB3` |
| B1 | PASS | configure and build exit 0; the 7 E4 cells listed; no "unanchored" line |
| B2 | PASS | 6.6/E4 green (116 checks); 6.6/E4:rule-b green (141); rule B = 128 (engine 128, helpers 128), cap 32768, cap_pages 256, page 786432; model `4b27e2d6…`; hook calls=3008 (> 32) |
| B3/B4/B5, first population (v1-bound) | ran, pairs equal | 20 files, run1 = run2. **Refused at L1:** the headers bound the superseded v1 records (QUARANTINED, measured at B=16). Set aside as `t3035-precache-refs-post-m1-v1bound-refused` |
| B7 | PASS | flat00 `ef7fae58…`, flat01 `9278830b…`; instrument `e4_real.cpp` `183de6de…b090` |
| T1a | PASS | mutant `T1-restore-prefills` built (MSVC, its own worktree and build dir) |
| T1b | PASS | detail below; receipt `PC-TIMING-1.txt` SHA-256 `0f0c56af…9372` |
| B8a–B8d | PASS, all four slices | lower bounds 3536–4330×; detail below |
| B8e | PASS | five PASS lines; the runner SHA in the logs equals B1's binary |
| B10 | PASS | MSVC 19.34.31937.0 job captured at `e8c029ed`; 4.3/E4 green (27 checks); the MSVC, GNU and Clang SPX1 blobs are byte-equal (`e869127a…`) |
| B11 | PASS (mutant killed) | `matvec-last-short`: 6.6/E4 RED (31 failures), 6.6/E4:rule-b RED on the real 0.5B (31), each "restored parallel token differs" |
| B3g | PASS | generator at `47bdcc9e`, merge-base `1948e1f2`, clean; no "unanchored" line; usage names `RECORD … no default` |
| B3, B4 (v2-bound) | PASS | 20 files, every run exit 0, 10/10 per run; harness `47bdcc9e` |
| B5 | PASS | 20 pair-equal, 0 differing; 20 `binds-v2`, 0 unbound |
| L1 | ISSUED, after A0 CLEAR | detail below; manifest `bfb0037f…733f`, licence `8bbb7c35…d130` |
| B6 | PASS | detail below |
| B9 | not run | needs W1 from another machine |

### T1b: commissioning the 10.3 timing instrument

Slice 0-1, alone, with the U3 PAUSE file present.

| Run | flat00 prefill / restore ms, ratio95 | flat01 | Slice | Verdict |
|---|---|---|---|---|
| must-reject (restore prefills `length` tokens) | 2024.2 / 1921.4, [1.0, 1.1] | 3318.6 / 3177.9, [1.0, 1.0] | 0 of 2, red, only the bar failures | PASS |
| must-accept (B1's build) | 2017.5 / 0.544, [3674.1, 3736.9] | 3338.8 / 0.885, [3714.7, 3828.5] | 2 of 2, green | PASS |

### B8: 10.3 paired timing

Eight pairs per prefix; restore includes the digest; one thread; rule B 128.

| Prefix | Bytes incl. digest | Prefill ms | Restore ms | ratio95 |
|---|---|---|---|---|
| flat00 | 977,976 | 2025.8 | 0.558 | [3536.2, 3732.0] |
| flat01 | 1,586,232 | 3323.6 | 0.884 | [3717.6, 3799.2] |
| flat02 | 2,335,800 | 4954.8 | 1.286 | [3827.8, 3878.8] |
| flat03 | 3,079,224 | 6692.0 | 1.707 | [3875.6, 3967.0] |
| flat04 | 4,615,224 | 10433.9 | 2.585 | [3879.6, 4207.3] |
| flat05 | 6,145,080 | 14497.5 | 3.430 | [4092.0, 4372.1] |
| flat06 | 7,423,032 | 18036.5 | 4.167 | [4298.3, 4358.0] |
| flat07 | 9,253,944 | 23567.9 | 5.519 | [4204.1, 4338.6] |
| flat08 | 11,066,424 | 29478.0 | 6.915 | [4216.5, 4310.5] |
| flat09 | 12,289,080 | 33600.6 | 7.702 | [4329.9, 4396.4] |

Every slice ran alone, with PAUSE present at start and end. No other process used more than 13.3 CPU-s in any slice,
against the re-run rule's 60, so no re-run was due. Prefill is 12.7–16.8 ms per token. Restore is about 0.6 ms per
MiB, digest included. §7 10.3 predicted ~1,200–4,000×. §10 R1 is not triggered.

### L1 and B6

- **L1, first pass: refused.** All 20 headers bound `discrimination/real0p5-*/<stem>.discrimination.json`, the
  historical T-3035 records that T-3071 leaves QUARANTINED. The ruling (plan rev 1.18.13/13a §15.33) re-bound the
  headers to the commissioned stream-v2 sidecars.
- **L1, re-run after A0 CLEAR: issued.** The static audit found nothing:
  - every header field was recomputed from the inputs;
  - every binding is its stem's stream-v2 sidecar, and none is a v1 hash;
  - registry standing: stream-v2 and byte-v1 are COMMISSIONED;
  - A0's sidecar checks hold: model and length match, and `reference_sha256` names the 5d5ca58 file whose rows equal
    the new rows.

  The loader constructions ran on scratch copies:

  | Construction | Result | Verdict |
  |---|---|---|
  | must-accept | two-process green; restore 10.1 4,184 checks, 10.2 4,377; 100 graded adopters | PASS |
  | must-reject: one token edited and re-licensed | 10.1 RED with only "flat05 adopter 0 differs from pinned prefill" and graded 49 vs 50 | PASS |
  | must-reject: licence over the v1 hashes | 10.1 RED, every reference refused "lacks its commissioned discrimination record identity" | PASS |

- **B6: PASS.** ctest `superslm_precache_real_two_process` gave 100% of 3 passed:
  - the fixtures `reference_header_identity` and `reference_anchor` both Passed, with origin/main → `1948e1f2`;
  - capture: 10.1 and 10.2 green;
  - restore: 10.1/E4 green (4,184 checks) and 10.2/E4 green (4,377);
  - all ten nested lines read `B = 128, s = 7, draw = 3`, with world 1012 and personas 183–211;
  - "21 hashes pinned, restore exited, 100 graded adopters"; 526 s.

## Notes

- The box checks out with CRLF (`core.autocrlf=true`). Exact-text mutant tools must follow the file's line ending.
  T1a's first application refused for this reason before it was fixed. `apply_e4_mutant.py`'s single-line anchor was
  unaffected.
- Still owed elsewhere:
  - W1/B9: the cap-length 1.5B worst setting, on another machine;
  - A1: the coverage auditor's E4 re-audit over these logs;
  - the planner fold of A0's Finding A and §4.

## B6r: 10.1 and 10.2 at the G1 fix (`1c145b53`), 2026-10-22 18:17–18:27 PDT

The run sheet at `ef952027`. `g1src` is a detached worktree at `1c145b53`: merge-base `1948e1f2`, clean tree. It was
built into `build-g1` (MSVC 14.34 Release), and both build commands exit 0. The test is ctest
`superslm_precache_real_two_process` on the licensed v2 population.
- The fixtures `reference_header_identity` and `reference_anchor` Passed; the merge-base reads origin/main →
  `1948e1f2`.
- Capture: 10.1 and 10.2 green.
- Restore: 10.1/E4 green (4,184 checks). 10.2/E4 green with **4,378 checks**, one more than B6's 4,377; the extra one
  is G1's zero-prefill assertion.
- All ten nested lines read `B = 128, s = 7, draw = 3`.
- "21 hashes pinned, restore exited, 100 graded adopters"; 100% tests passed (3/3); 525 s.
- **PASS.** This log, `logs/B6r-two-process.log`, is 10.1/10.2's E4 record.
