# Attention and per-row sites, slice S1: build progress

The plan of record is the attention and per-row sites plan, rev 3.1, approved by the owner. This series builds
its slice S1 only (§4.1): 256-entry per-row tables for the SwiGLU sigmoid (`MlpActSite`), the residual landing
(`ResidualReconcileSite`) and the RMSNorm divide (`RmsNormSite`). Same pure function, same arguments; every tier
bit-identical to the v1.9.0 per-element code.

**Base.** Tiled-matmul slice 1's branch at `fb56397` (the plan names `c3e0004`; `fb56397` is that plus one
documentation commit). Delivered as a patch series (`git am`), not pushed: that branch carries slice 1's draft PR.

**Host.** A 4-vCPU cloud Xeon with AVX2, AVX-512F/BW/DQ and AVX-512 VNNI; GCC 13.3.0, Clang 18.1.3. The host is
shared with other agents, so every timing is best-of-N and noisy.

States: **done**, **CI-only** (runs on the hosted legs; not runnable here), **box-only**, **not done** (with why).

## Resume here

1. Clone SuperSLM, `git checkout -b attn-s1 origin/claude/project-thread-c8iecr`, `git am` the series.
2. Cell 11.1(d) needs the 0.5B-width 1-layer synthetic artifact (sha256 `f0fd4886…6ed3`; about 9 minutes and
   7.4 GiB with `tools/consumer_reach/synth_artifact.py`). Point `SUPERSLM_ATTN_ROWSITES_ARTIFACT` at it and run each
   suite binary from the repository root; without it the cell prints SKIPPED and the rest runs.
3. The golden pin is regenerated with `tools/gen_attn_rowsite_golden.cpp` built against the v1.9.0 tag (recipe in
   its header). The CMake target of the same name builds it against the current tree and must print the same hash.
4. Mutants: `docs/attention-rowsites/s1/mutation-scripts/*.py`, each applied from the repository root to a clean
   copy of the implementation commit; rebuild `superslm_tests` and `superslm_tests_avx2_forced` and run both.

## S1 items

| Item | State | Evidence |
|---|---|---|
| Red-first: cells, row-table counters (declared, not incremented), digest section, golden pin | done (commit 1) | `docs/attention-rowsites/s1/red-suites.txt`: GCC auto and forced SSE2/AVX2/AVX-512, 258 of 1,756 S1 checks fail, every one a counter assertion; every value assertion passes on the base |
| The three tables, `kRowTablesOn` (off only under forced scalar), `kRowTableMinWidth` = 512, counters (§3.6) | done (commit 2) | `src/forward/forward_sites.cpp` |
| GCC 13.3 suites: auto (AVX-512 here), forced SSE2, AVX2, AVX-512 | done, 0 failures | `suites.txt`: 27,288 / 27,230 / 27,246 / 27,246 checks (S1: 1,757 each) at the series head |
| Clang 18.1.3 suites, same four binaries | done, 0 failures | `suites-clang.txt` |
| MSVC / clang-cl forced AVX2, macOS arm64 (tables on, counters asserted) | CI-only | the hosted legs; nothing Windows- or arm64-specific in S1 (no SIMD, no switch) |
| Digests (6.2): auto and scalar/SSE2/AVX2/AVX-512 forced, GCC and Clang | done | all ten equal: GLOBAL `3a829091…`; `c_rowsites` `8836d5eb…`; sections 1–9 byte-identical to slice 1's (`18e45694…` GLOBAL before the new section) |
| Golden pin (6.3) from the v1.9.0 tag | done | `tests/attn_rowsite_golden_pin.h`: `8836d5eb…` over 634,120 values, matched by all eight suite binaries; SiLU-LUT, matmul and tiled goldens unchanged |
| End-to-end blobs (6.4) | done, all EQUAL | `blob-protocol.txt`: in-tree fixture (24 tokens), wide fixture, 0.5B-width 1- and 2-layer at 8/128/512 tokens × chunk budgets 1/8/whole, each + 32 greedy decode steps; blob and decoded tokens equal to the base's in all 33 comparisons (the N = 8 "whole" rows repeat the chunk-budget-8 rows) |
| Mutants (§9 S1 rows and the "all" counter rows) | done, 15 of 15 killed on both binaries | `mutants.txt` (below) |
| Path counters (§3.6, 11.1(a), 11.1(d)) | done | every 4.S1/1.S1/5.S1/7.S1 call asserts its own delta; 11.1(d) on p05_l1: prefill 256 / 128 / 256 taken, decode 64 / 32 / 64, every skipped 0, chain trace records 1,536 and 384 (128 and 32 per site name) |
| 3.S1 concurrency; sanitizers | done | new cell (see deviations). GCC TSan and ASan+UBSan builds of `superslm_tests` with the artifact: 27,288 checks, 0 failures, 0 sanitizer reports each (`suites.txt`); the hosted TSan and ASan legs are the CI copies |
| Branch-coverage floors (11.6) | not owed | S1 edits neither `intmath.cpp` nor `matmul.cpp` |
| Bench (10.1) and the saving | done | `bench.md`: **0.80 ms/token** measured against the plan's 0.87 |

### Mutants

Each killed on the auto binary and on forced AVX2 (the tables are tier-independent, so every S1 mutant sits in shared
code). The cell that turns red is the plan's named one in every case:

| Mutant | Killed by |
|---|---|
| norm / SiLU / landing table read one entry off ("indexed by code + 127") | 4.S1, 1.S1, 5.S1, 6.3 (41 / 77 / 73 failures) |
| SiLU −128 read from an unset table entry | 4.S1's small-gate-scale −128 rows (11) |
| landing −128 read from an unset table entry | 4.S1 residual rows with −128 (32) |
| norm table built for [−127, 127] only (extra) | 4.S1 norm −128 rows (18) |
| landing flag OR-ed over the whole table | 7.S1c (7) |
| norm / SiLU / landing table built once per call site | 1.S1 and 4.S1 (30 / 57 / 41) |
| threshold `n ≥ 0` | counters at widths 1–511 (123) |
| threshold `n ≥ 513` | counters at width 512 (36) |
| threshold `SIZE_MAX` | counters, including 11.1(d) prefill and decode (138) |
| skipped increment deleted ("fallback increment deleted") | counters (123) |
| taken counted before the guard ("fast increment moved before the guard") | counters (123) |

"AVX-512 dispatch runs the AVX2 body" does not apply to S1 (no SIMD body, no tier split).

## Deviations from the plan, and why

- **11.1(c) (the QK-norm attention fixture) is not built in S1.** Its table's first column is "S5 landed"; the
  fixture exists to drive the Q31 path, and its row-table rows (every `…_skipped`, widths 64 and 256) add nothing
  4.S1 does not already assert at widths 1 to 511. S5 builds it and adds its forward hash to the pin header.
- **The golden pin is one hash per slice**, not one hash over every section's inputs, so no later slice
  regenerates S1's. S1's covers `c_rowsites`' S1 entries (the digest section's value equals it until S3 appends).
- **11.1(d) is an environment-gated cell in the suite binaries**, since the plan does not commit the artifact.
  S1 asserts the rows that exist in S1: the `rowtable_*` closed forms and the chain trace records. `requant_row`,
  `softmax`, `pv` and `grouped` rows arrive with their slices.
- **3.S1: the plan's "existing concurrent-read stress cell … on the 0.5B-width artifact" does not exist.** The
  suite's concurrent-read cells cover `SiluSigmoidQ15` and `GemmInt8AccumulateRow` only, and none takes an artifact.
  S1 adds its own cell (8 threads × all three sites at table widths, compared with the reference), which the hosted
  TSan leg runs; it was also run here under GCC TSan (below).
- **The blob tool needed two options** the plan's protocol assumes: `--chunk-budget=B` and `--decode=D` in
  `--dump-blob` mode (`tools/t2147_chunk_batched_pins.cpp`). The in-tree fixture's context cap refuses 128 + 32
  positions on the base too, so its rows run at 24 tokens.
- **Base defect fixed in passing:** `sslm_bench_prefill_{avx2,avx512}_forced` did not compile on the base (the
  forced library exports the instrument macro, and the tool then includes a `tests/` header without `tests/` on its
  path). One `target_include_directories` line each.
- **Decode saving not resolved on this host** (bench.md): about 2% of a reduced-layer decode step against ±10%
  pair-to-pair noise. The prefill reading agrees with the per-site method.

## What the plan got wrong or left loose (for the next revision)

- §0/§4.1's 0.87 ms/token is close: measured 0.80 by the plan's own per-site method, 0.96–0.99 from reduced-layer
  prefill. The loop-only speedups the plan quotes (6.0×, 3.2×, 2.57×) are not what a site call sees (1.50×, 1.39×,
  1.25×), because the funnel and the allocation stay; the absolute saving is what matches.
- 3.S1 cites a cell that does not exist (above).
- The engine base is `fb56397`, not `c3e0004` (one documentation commit later).

## ID-PENDING list

Nothing minted here.
