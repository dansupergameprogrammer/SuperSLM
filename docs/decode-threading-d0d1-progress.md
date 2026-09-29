# Decode threading, steps D0 and D1: build progress

The plan of record is the SuperSLM decode-threading plan, rev 1.2, approved by the owner. This series builds its
steps D0 and D1 (§6): the one-row (M = 1) matvec of decode and one-token prefill split across the host's
`sslm_parallel_for` hook, opt-in through the new bit `SSLM_PARALLEL_FOR_MATVEC` (bit 1), with output bit-identical to
the serial path. D2–D5 (the box, the release and the plugin) are not in this series.

**Base.** Branch `claude/project-thread-c8iecr` at `c28f171` (the attention and per-row sites series, S1–S5, on tiled-matmul
slice 1). Delivered as a patch series (`git am` onto `c28f171`), not pushed.

**Host.** A 4-vCPU cloud Xeon (AVX2, AVX-512F/BW/DQ, AVX-512 VNNI; GCC 13.3.0, Clang 18.1.3), shared with other agents, so
every timing is indicative and noisy.

States: **done**, **CI-only**, **box-only**, **not done** (with why).

## Resume here

1. Clone SuperSLM, `git checkout -b decode-threading c28f171`, `git am` this series.
2. Generate the fixtures: `python tools/gen_decode_threading_fixture.py <dir>` (numpy; 5 to 12 minutes on this host, deterministic), then set
   `SUPERSLM_DECODE_THREADING_FIXTURE_DIR=<dir>`. A missing fixture fails the D1 cells; it never skips them.
3. Run the suites from the repository root. `SUPERSLM_ATTN_ROWSITES_ARTIFACT` (the S1–S5 0.5B-width 1-layer artifact) adds
   the attn-rowsites 11.1(d) checks; every count below is with it set.
4. The forced-tier suites, the scalar-tier decode-threading binary and the seam bench are `EXCLUDE_FROM_ALL`: build them by
   name.

## The series

| # | Commit | What |
|---|---|---|
| 1 | D0 | `src/forward/parallel_split.{h,cpp}`: `SplitColumns`, `RunExactlyOnce`, `SaturationCounters`, the shared setter validation with an empty implemented-bit mask. `LogitsSiteParallel` moved onto them. Cells 4.1, 7.1 and the snapshot round trip |
| 2 | D1 red | The fixture generator, the two test seams, the API surface stubbed, and the §8 suite (`tests/test_decode_threading_d1.cpp`), the scalar-tier runner, the 7.4 allocation binary and the WIN32 setter block. |
| 3 | D1 green | Bit 1 in the mask, the §3.1 rule in `MatvecGroupSplit`, both retry guards |
| 4 | D1 | The fixture-generation steps in `tests.yml`, the contract text (`parallel_for.h`, `docs/api.md`, CHANGELOG [Unreleased]) and `tools/decode_threading_bench.cpp` |
| 5 | fix-ups | The census and forward-leaf checks the pytest comparison found (see deviations 12 and 13) |
| 6 | D0 follow-up | The t2296 fp-free suite's 17-object pins read 18 (deviation 13) |
| 7 | evidence | Cell 2.1 gains the o site (coverage-found, deviation 11); this file and `docs/decode-threading/` |

## D0 and D1 items

| Item | State | Evidence |
|---|---|---|
| D0: shared split, exactly-once runner, counter snapshot, setter validation (empty mask); logits on the shared split | done (1) | 4.1: 23,040 splits, 13,260 threaded; 7.1: 765,000 splits and 35,199 logits calls against v1.9.0's formula, 16,320 one-task calls. The base's suites stay green (121,658 = 121,651 + 7) |
| Red-first: the §8 suite authored before the implementation | done (2) | `red-suite.txt`: 478 failures, all in the D1 cells (420 tagged to 21 cells, 58 the shared install check); 7.4 red; without the fixture directory 27 failures, no skip |
| Bit 1, `GemmThreading`, the overloads, `MatvecGroupParallel`, `GemmDispatch`, decode and one-token-prefill wiring | done (3) | `suites.txt` |
| Both retry guards (§3.6) | done (3) | Cells 5.1–5.5, 2.1; mutants M6–M8c |
| The §3.9 fixture generator and CI steps | done (2, 4); CI-only for the legs themselves | `fixture-premise.txt`; `.github/workflows/tests.yml` (fourteen legs, deviation 7) |
| Contract text, `docs/api.md`, CHANGELOG [Unreleased] | done (4) | — |
| Suites, GCC 13 and Clang 18: auto plus forced SSE2, AVX2 and AVX-512, the scalar tier, 7.4 | done | `suites.txt`: 0 failures on all twelve binaries; the D1 cells are 2,108 checks on every tier and both compilers; superslm_tests 123,766 (base 121,651) |
| AVX-512-forced suite executed on AVX-512 (4.5, F9) | done | `suites.txt` quotes the suite's own tier line |
| Output bit-identical: the digest legs | done | `digests.txt`: GLOBAL `f740f833…` on all five tiers, both compilers, equal to the base section by section |
| Sanitizers | done | `suites.txt`: ASan+UBSan (GCC) and TSan (Clang) on auto, AVX2- and AVX-512-forced: 0 reports |
| §9 mutants | done | `mutants.txt`: 23 of 23 executable mutants killed (the §9 rows, with M6 split per counter and M7b both ways), plus the coverage-found M7-o; M9, M9c, M12 not executable here (WIN32, a two-bit engine, the plugin) |
| fp-free scan, both compilers | done | `fp-scan.txt`: PASS, 18 objects, `parallel_split.cpp.o` clean; allow-lists unchanged. |
| The Python suites, against the base | done | `pytest.txt`: the checker suite 915 passed, 1 skipped (= base); t2296 the base's 4 environmental failures, 154 passed |
| Named checkers | done | `pytest.txt`: guard parity, CI claims, present-tense defects, geometry census, forward-leaf: all OK |
| `gpu_layer_loop_guards.def` re-cited by hand | done (1, 2) | The parity check passes at the head |
| 6.3 and 9.1 against v1.9.0 | done, local only (deviation 9) | `v190.txt`: byte-equal both fixtures, both directions |
| 9.3 old symbols still exported | done | `symbols.txt`: 3 of 3 base overloads present on GCC and Clang |
| Coverage replica, with and without AVX-512 | done, indicative | `coverage.txt` |
| Bench tool | done; figures indicative | `bench.md` |
| 2.2 and 7.3 on WIN32, 4.6 MSVC | CI-only | `cell_setters.cpp`'s sibling block and the Windows legs |
| B0–B5 (box, release, plugin) | box-only | §7 |

## Coverage (indicative; `coverage.txt`)

| File | Floor | Base, five profiles | D1, five profiles | Base, no AVX-512 | D1, no AVX-512 |
|---|---|---|---|---|---|
| `src/matmul.cpp` | 72.22 | 288/312 92.31% | 288–289/312 92.31–92.63% | 205/296 69.26% | 205–206/296 69.26–69.59% |
| `src/intmath.cpp` | 87.93 | 222/242 91.74% | 221–222/242 91.32–91.74% | 203/242 83.88% | 202–203/242 83.47–83.88% |

D1 edits neither file; the ranges are two runs of the D1 tree (one branch of run-to-run jitter). With all five profiles
both files clear their floors. The no-AVX-512 projection (the hosted runner's view) is below both floors on the base
already, as S2–S5 recorded; D1 does not move it. `src/forward/` is outside the leg's glob: exported separately,
forward_sites.cpp goes from 731/910 to 779/948 and parallel_split.cpp is 45–46/48. The replica found o's guard arm
untaken, which became the 2.1 o case (deviation 11). Floors not re-pinned.

## Bench (indicative; `bench.md`)

On a synthetic 2-layer, 0.5B-width artifact: B0's verify is byte-equal at `max_tasks` 1, 2, 4, 8, with 9 `run` calls per
token (4L + 1, §3.1's 97 at L = 24). Decode tok/s off against on, n = 15 pairs: 1.27–1.57x; one-token prompt 2.41 →
1.57 ms at 4 tasks. The no-artifact `groups` mode at 0.5B shapes: one layer's matvecs 2.49 → 1.29 ms at 4 tasks, 2.62 →
1.71 at 2. The threshold sweep (64 KiB, 256 KiB, 1 MiB) does not separate here. None of these is a plan reading.

## Deviations from the plan as written

1. **F-QK has head_dim 128, not F-DEF's 48.** §3.9 says F-QK is "the same geometry with QK-norm kept", but the loader
   admits a QK-norm artifact only at head_dim 128 (`src/model.cpp`, `ValidateFusedKHeadDim`). F-QK keeps hidden 192 and
   intermediate 512 with 4 query heads and 2 KV heads of 128 (query width 512). §3.9's task counts are F-DEF's; F-QK's
   q, k + v and o widths differ, so the cells that need exact counts run on F-DEF, and F-QK carries the guard, trace
   and blob cells.
2. **The generator has one calibration knob (R6's fallback, taken).** With the pipeline's own calibration no decode step
   satisfies the F1 preamble: rope_q and rope_k clamp about once a layer. The K calibration peak is multiplied by 0.2
   for F-DEF (`fixture-premise.txt` has the sweep). A third artifact, `fnoclamp.sslm` (K and V peaks x 8), is 11.1's
   vitality variant.
3. **The F1 preamble is vacuous for 2.1's q and k + v cases.** A failure at q or k + v of layer 1 comes before layer 1
   lands anything, so "layer 1's counters restored" holds whatever the guard does there. The o, gate + up and down cases
   and 5.1 carry the guard's weight; M7 and M7b die on them.
4. **rope_q ≠ rope_k is required only where both move.** On F-QK rope_k never moves (QK-norm path), so its preamble asks
   for kv, k_channel_landing and rope_q, and rope_q ≠ rope_k then holds trivially.
5. **The seams are on the forced-tier libraries too, not on `superslm_test_injection` alone** (§3.9). 4.5 runs 4.2 and
   6.1–6.3 on every forced tier, and those need the 8 KiB seam. `SUPERSLM_ENABLE_MATVEC_TEST_SEAMS` is PUBLIC on
   `superslm_test_injection` and the forced libraries; the production `superslm` library has no seam in its path.
6. **The scalar tier has its own binary.** There is no scalar-forced test suite in the base (only a digest), so 4.5's
   `superslm_scalar_forced` leg is `superslm_decode_threading_scalar_forced`: the D1 cells alone, on a seam-enabled
   scalar-forced library. The CI SSE2 leg builds and runs it.
7. **CI: fourteen legs, not the six §3.9 lists.** Every leg that runs a suite binary gets the Python and generator steps,
   because a missing fixture fails the cell: windows-x64, linux-x64, ASan, macOS, Debug, TSan, the three Linux forced
   tiers, the four Windows forced tiers and branch coverage. linux-x64 also runs the 7.4 binary.
8. **7.4 is its own executable** (`superslm_decode_threading_alloc`): the counting global `operator new` replaces the
   allocator for the whole process, so it cannot live inside `superslm_tests`. It measures 81 allocations per decode
   token and 96 per one-token prefill, bit 1 off and on alike.
9. **6.3 and 9.1 against v1.9.0 are local evidence, not a CI step.** The in-suite 6.3 compares bit 1 on against chunk-1
   no-hook in the same library. The v1.9.0 comparison (§8 6.3's "the generator's artifact run on the v1.9.0 library")
   and 9.1 were run here with a probe built against the v1.9.0 tag (`v190.txt`, `v190_blob_probe.cpp`); no workflow
   step builds v1.9.0.
10. **8.1's LoRA and Option-G halves run through the C++ layer loop; the damped-greedy and schema half and the prefix
    adoption through the ABI.** The fixture carries no adapter section and the ABI does not expose the Option-G landing
    mode, so a synthetic rank-1 adapter and the Option-G mode are driven through `RunLayerLoop`'s new overload; each is
    compared with its own no-hook run.
11. **Cell 2.1 gains the o site, after green (coverage-found).** §8 2.1 names q, k + v, gate + up, down and the finish.
    The coverage replica showed o's failure arm (`fail_layer`) taken by no cell. Four hostile cases at o are added; the
    mutant that routes o's failure around the guard dies on them and on nothing else (`mutants.txt`, M7-o).
12. **The fault seams sit above their GS-12 markers.** `ArmLayerSiteFault`'s check at q and o, in both loops, first sat
    between a `SSLM-GEOMETRY-SITE: GS-12` marker and its anchored call, which the geometry census reads as a regressed
    site. The seam now precedes the marker, and the decode q call is wrapped so that its first line carries the output
    width as the census expects; the census fixture that quotes that line follows it.
13. **D0's new translation unit broke three exact-count pins the plan does not mention.** `check_no_forward_leaf_calls.py`'s
    expected file list, and the t2296 suite's 17-source / 17-object pins (manifest, real build, COFF and ELF archive
    member counts, the carrier sweep). Each now names `parallel_split.{h,cpp}` or reads 18. Found by the pytest comparison
    against the base, fixed in commits 5 and 6. The COFF cells run only on the Windows leg.
14. **The WIN32 halves are CI-only.** 2.2's GPU setter block (`tests/t2956-token-finish-red-suite/cell_setters.cpp`,
    returning 14/15, guarded by `SSLM_PARALLEL_FOR_MATVEC`), 7.3 and 4.6 are compiled and run only by the Windows legs;
    none executed here. The older Windows test recipes (`tools/build_*.bat`) gained the new source only.
15. **8.2's two-bit parts are N/A at D1.** Only "bit 0 is rejected on a D1-only engine" applies; the rest returns at S2-A.
16. **The bench adds a `groups` mode, and no real 0.5B artifact was available here.** `groups` times one decode layer's
    five groups at Qwen2.5-0.5B shapes on synthetic weights through the reference pool, with no artifact. `verify`,
    `time` and `onetoken` ran on a synthetic 2-layer artifact at 0.5B width. B0–B1 on the real artifact stay box-only.
17. **The base's check count depends on an environment variable.** attn-rowsites 11.1(d) runs only with
    `SUPERSLM_ATTN_ROWSITES_ARTIFACT` set (238 checks). Every count here is with it set; the commit messages' counts
    (123,464 at green) are without it.

## What the plan got wrong

1. **§3.9's "F-QK: the same geometry with QK-norm kept" cannot load** (deviation 1): QK-norm requires head_dim 128.
2. **R6 was not "low" likelihood.** At the pipeline's calibration no decode step of the fixture satisfies F1, so the
   generator must carry a knob; the plan's fallback ("raise the generator's activation scale") works as a K-peak factor
   of 0.2 (deviation 2).
3. **§3.9 puts the seam on `superslm_test_injection` alone, but 4.5 needs it on every forced tier** (deviation 5), and
   names a `superslm_scalar_forced` suite that does not exist (deviation 6).
4. **§9's M4 row lists 4.2, which cannot kill it.** Every 4.2 count is M = 1, and M4 widens the test to M ≤ 7; 4.4 kills it.
5. **§8 2.1 omits the o site**, leaving o's guard arm untested (deviation 11).
6. **§3.9's CI leg list is short by eight legs** (deviation 7): every leg that runs a suite binary needs the fixture.
7. **D0 is not "no change" for the CI checks.** A new core source moves the exact-count pins of three checks (deviation 13).

## ID-PENDING list

Nothing minted here.
