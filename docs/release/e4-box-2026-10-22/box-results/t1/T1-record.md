# T1: commissioning the 10.3 timing instrument (debunker), E4, home box 2026-10-22

Written **before any 10.3 reading** (T1b or B8). Instrument: `tests/precache/e4_real.cpp` at e8c029ed
(SHA-256 `183de6de8a881d97842ffecdde3c8f4f4d231a644c7c6d4613ac9cf6dccdb090`), cell `10.3/E4` (`Timing103`): 8 paired
prefill/restore observations per prefix, one untimed warm-up pair per process, a two-sided 95% log-ratio interval
(t = 2.365), and a bar of lower ≥ 1000.

## Constructions (mine)

- **Must-reject (T1a): `T1-restore-prefills`** (`make_mutant.py`). In `sslm_prefix_restore`, after the blob validates
  and before its own draw, a scratch prefix of the blob's `length` is begun on the same pool, prefilled in 64-token
  chunks, and released; stderr prints `MUTANT T1-restore-prefills: prefilled N of L`. The restore's output is
  unchanged; its cost becomes a prefill's plus its own. Built like B1 (MSVC 14.34 Release, Ninja) from a detached
  worktree at e8c029ed into `D:\_scratch\e4\build-mut`.
- **Must-accept:** B1's build (`D:\_scratch\e4\build-msvc\superslm_precache_e1.exe`), unmodified.

Both run slice `0-1` (flat00, 159 tokens; flat01) of `10.3/E4`, alone on the box with the U3 PAUSE file present,
under the provisional receipt `D:\_scratch\e4\t1\PC-TIMING-1.provisional.txt`. Its 13 lines are the run sheet's,
with `prompt_manifest_sha256` of B7's manifest and `instrument_sha256` above. The mutant runs first.

## Pass criteria, fixed now

- **must_reject = PASS** only if all of these hold in the mutant's log:
  - the receipt is accepted (no "timing receipt identity" failure);
  - every timed restore returns `SSLM_OK`;
  - the mutant line prints `prefilled L of L` for each timed restore, so the construction ran;
  - both prefixes print a `ratio95` with **lower < 1000**;
  - the slice reports `0 of 2` and the cell is red.

  A red for any other reason (receipt, crash, a restore status, the timer floor) is a broken construction: FAIL.
- **must_accept = PASS** only if B1's run accepts the same receipt, both prefixes read **lower ≥ 1000**, the slice
  reports `2 of 2`, and the cell is green.
- PASS on both → I write `D:\_scratch\e4\timing\PC-TIMING-1.txt` (the same 13 lines). Either FAIL → no receipt, and
  B8 does not run.

## Re-run rule for B8a–B8d (binding before any reading)

A B8 slice reading **stands as taken**, green or red, with one exception. A slice may be re-run once, with both
readings sent to the planner, only on one of these named disturbances, recorded at the time:

1. the slice process did not finish: a box crash, a reboot, or a runner killed before printing its slice summary.
   There is no reading, so the re-run is the first reading;
2. the U3 PAUSE file was absent at the slice's start or end;
3. another process used more than 60 CPU-seconds during the slice. This is read from a process CPU-time snapshot
   taken immediately before and after the slice, excluding the runner, the System Idle process, and this shell.

A miss with none of these stands; it is §10 R1's trigger, and the timing pieces stop. "The number looks low" is never
a disturbance.

## T1a and T1b readings (16:44–16:49 PDT)

- **T1a.** The first application refused: the anchor was not found because the box checks out with CRLF. The script
  was fixed to follow the file's line ending, then applied (`src/sslm_abi.cpp`, +16 lines, plus `<cstdio>`). The
  incremental build rebuilt `sslm_abi.cpp`. The mutant string is present in `build-mut`'s runner and absent from B1's.
- **Must-reject** (runner `0db0e1cc…f019`, log `T1b-mutant.log`): the receipt was accepted.
  - The mutant line printed `prefilled 159 of 159` 9 times (8 timed plus the warm-up) and `prefilled 258 of 258`
    8 times.
  - flat00: prefill 2024.2 ms, restore 1921.4 ms, ratio95 [1.0, 1.1].
  - flat01: 3318.6 ms against 3177.9 ms, [1.0, 1.0].
  - `0 of 2`, red, with 3 failures: two bar misses and graded 0 of 2. Nothing else failed.
  - **PASS.**
- **Must-accept** (B1's runner `aba74f21…e86e`, log `T1b-accept.log`):
  - flat00: prefill 2017.5 ms, restore 0.544 ms, ratio95 [3674.1, 3736.9].
  - flat01: 3338.8 ms against 0.885 ms, [3714.7, 3828.5].
  - `2 of 2`, green (199 checks, 0 failed).
  - **PASS.**
- Both runs: PAUSE present at start and end; no other process over 5 CPU-s.
- **Verdict: COMMISSIONED.** `D:\_scratch\e4\timing\PC-TIMING-1.txt` was written with the same 13 lines as the
  provisional receipt (receipt SHA-256 `0f0c56afe86b771c2a6696d139c756adc5bea71d0ca364d4b6188bced69c9372`).
