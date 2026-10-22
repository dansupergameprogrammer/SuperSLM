# SuperSLM 1.14.0: home-box receipts, 2026-10-22 (short pieces, before E4)

Branch `claude/precache-1p14-post-m1`. The engine source is identical at `51bc3d6a`, `2256b657` and `d129b5fa`: the
commits between them touch only `tools/longleg_kit/`. Home box: AMD Ryzen 9 3950X, Windows 10, a full clone
(`git rev-parse --is-shallow-repository` reads false).

## C ABI verbs list

`tests/t2807-api-slot/gen_consumed_symbols.ps1` was run against this branch's headers, with the committed file's two
recorded consumers, writing to scratch. The result: **53 verbs, identical** to the committed `c_abi_verbs.txt`. Only
the provenance comment differs ("at 51bc3d6a" in place of "at the merge of a5f3ad1 into fcce0e1").

## The MSVC GEMM pin and the 7-token prefill

- `tools/msvc_pin/run-pinmap.ps1` on `sslm_decode_threading_bench` (MSVC 19.44 Release): `GemmInt8AccumulateCols` 0,
  `DotRowAvx2` 16, `TiledGemmAvx2` 32 mod 64. **PIN OK**, exit 0.
- The one-thread 7-token prefill A/B, against v1.13.1 (`d697927f`), uses 2026-10-21's `run-abp.ps1` method: the
  bench source from 2af41659, compiled decode-only against each library; 15 rounds after a warm-up; one process per arm
  per round, the order rotated; the aex-pr model (`9ccf7e37...`).
  - v1.13.1: **183.4 ms** [182.4, 184.4].
  - post-M1 (`51bc3d6a`, a fresh library): **147.4 ms** [147.3, 147.5].
  - **post-M1 / v1.13.1 = 0.806 [0.799, 0.809]**, the pinned fast state. The M1 squash read 0.810 the same morning.

## B1: the bounded precache suites, three Windows builds

| Build | E2 (44 cells) | E3 (13) | Unpinned E2/E3 (22, no 1.5B) |
|---|---|---|---|
| MSVC 19.44 Release | 202,569 checks, 0 failures | 2,486 / 0 | 34,156 / 0 |
| clang-cl 15.0.1 Release (VS Community, MSVC 14.34) | 202,569 / 0 | 2,486 / 0 | 34,156 / 0 |
| MSVC 19.44 ASan (`/fsanitize=address`) | 202,569 / 0, no ASan report | 2,486 / 0, no ASan report | not run (by ruling; green on the other two builds) |

The selection:
- E2: every `*/E2*` cell of `superslm_precache_e1` except `4.1/E2:synthetic-shapes`, `6.2/E2:reference` and `9.1/E2`.
- E3: every `*/E3*` cell except `4.2/E3` and `6.3/E3:reference`.
- Unpinned: the E2/E3 legs except the 1.5B (`synthetic1p5`) ones.

**The E0 artifact directory** (`SUPERSLM_PRECACHE_E0_ARTIFACT_DIR`, the T-3030 commission) was set, so cells
**2.13/E2, 2.14/E2 and 11.6/E2** ran. All three are in the green E2 runs.

**clang-cl:** BuildTools' MSVC 14.44 STL refuses clang older than 19 (`STL1000`). So clang-cl 15 builds in VS
Community's own environment (MSVC 14.34), started clean: inheriting the BuildTools environment first reproduces
`STL1000`.

## 7.12 value receipt

7.12/E2:independent-parser saved a prefix through `sslm_prefix_save` (`prefix.spx1`).
`tests/precache/check_version_values.py --expect 1.13.1 --blob <it>` reads ok for all three: the CMake project
version, `sslm_version.h` and the SPX1 producer. 1.13.1 is this branch's version until E5's release commit; at E5 and
M2 the receipt is re-run with `--expect 1.14.0`.

## 6.4 current_token mutant (MSVC)

With `SUPERSLM_PRECACHE_CURRENT_TOKEN_MUTANT=1`, 6.4/E2 goes **red**, as it must: "6.4 nested restored token field
differs from fixture input" (`e2_root.cpp:112`). The same cell is green in the unmutated E2 run.

## Not run here (over about 20 minutes; on the long-leg drive kit or another machine)

- The 1.5B legs: 4.1/E2:synthetic-shapes, 4.2/E3, and the `synthetic1p5` unpinned legs. These are on the long-leg
  drive (`tools/longleg_kit`), running on a spare PC.
- E4's timing and real-artifact cells.
- The long TSan cells (Linux only).
