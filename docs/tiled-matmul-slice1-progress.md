# Tiled int8 prefill GEMM, slice 1: build progress

The plan of record is the tiled-matmul plan, rev 10, approved by the owner. This branch builds its slice 1
only: steps S1-0, S1-B and S1-C of the plan's §12. Slice 2 (threading) and slice 3 (VNNI) are out of scope.

**Base substitution.** The plan was written on the 1.8.0 base (its constraint C10). The engine has since
released v1.9.0, and the Unreal plugin vendors 1.9.0. This branch therefore starts from v1.9.0 plus two
build/CI housekeeping commits (`d86aa3c`), and every bit-identity reference (golden, digest and
real-artifact blobs) is taken from **v1.9.0** (`d870d27`). Wherever the plan says "1.8.0 base" or "the v1.8.0
pins", read "1.9.0". The lead of the build thread decided this substitution.

**Host.** A 4-vCPU cloud Xeon with AVX2, AVX-512BW and AVX-512 VNNI; GCC 13.3.0. Another agent runs timing
experiments on the same host, so every timing here is best-of-N and noisy.

**Record IDs.** Decision and ticket IDs are minted on the owner's machine. Where the plan asks for one, this
file writes `ID-PENDING`; the list is at the end.

States: **done**, **red** (written and failing, as red-first requires), **box-only** (cannot run on this
host; stated why), **blocked**, **pending**.

## S1-0: baseline on the v1.9.0 base

| Item | State | Commit | Evidence and numbers |
|---|---|---|---|
| v1.9.0 digest, forced GCC AVX2/AVX-512 suites and digests on this host | done | 4def16a | `docs/tiled-matmul-slice1/s1-0-base-v1.9.0/`. All five digests (auto, scalar, SSE2, AVX2, AVX-512 forced) GLOBAL `cc267693…`; `c17_matmul` `ee456f50…`. Suites: auto 25,299 checks, AVX2-forced and AVX-512-forced 25,278 checks each, 0 failures; matmul golden `932478a4…` |
| Token-id mode upstreamed into `tools/t2147_chunk_batched_pins.cpp` | done | 4def16a | `ids:N[:V]` prompt, `--layers=L`, `--repeat=R`; also fixes `--boundary-sweep=` reading one character too far |
| Wide fixture and reduced-layer real-width artifacts | pending | | built in scratch with `tools/consumer_reach/synth_artifact.py` (adds the Qwen2.5-0.5B geometry) |
| §11.R guarded route run once against the base | done | 4def16a | `docs/tiled-matmul-slice1/s1-0-base-v1.9.0/route-e-r11-v1.9.0.txt`: SuperEmbedder `1ee74bf` pins, configures ("pin verified: 1.9.0 @ d870d27…") and builds against v1.9.0 |
| `probes/route_e/` and `probes/route_p_hook/` upstreamed as the 10.0 tool and its checks | done | 4def16a | `tools/consumer_reach/` (current-revision scripts only; the probes' rev-N history and outputs stay with the plan). The 10.0 tool is `tools/consumer_reach/route_e/route_e_reach.cpp`, not `tools/sslm_consumer_reach.cpp`: it links SuperEmbedder, so no CMake or `build.bat` recipe in this repository can build it, and the runner script is its recipe |
| Baseline whole prefill (reduced-layer artifacts) | pending | | |

## S1-B: the kernel

Evidence files: `docs/tiled-matmul-slice1/s1-b/` (`suites.txt`, `mutants.txt`, the five digests, and the mutation
scripts under `mutation-scripts/`). Every mutant below was applied to a synced copy of the branch and run on the
auto binary (which dispatches AVX-512 on this host) and the forced-AVX2 binary. "Killed" means the cell named turned
red on the mutant; the unmutated control is green in the same run.

| Item | State | Commit | Evidence |
|---|---|---|---|
| `Cols` entry over the shipped loop (review W8), both entries | done | 2eebbbc | cell 3.3 green; mutant XC (range ignored) red, 102 of 108 checks |
| Tiled kernel, activation prep, packer | done | 3c9a7cf | auto, forced AVX2, forced AVX-512 and forced SSE2 suites: 25,531 / 25,489 / 25,489 / 25,473 checks, 0 failures; tiled cells 211 (195 on SSE2, where the tiled path is not taken). The shipped matmul golden is unchanged (`932478a4…`) |
| Red-first kernel mutants | done | 3c9a7cf | killed on both binaries: X1 flush deleted (6.1), X2 window 66,053 (6.1), X3a pair order (134 checks + 3.5), X3b 8-output halves (134), X4c store guard dropped (abort, heap write past the output), X4e both pads nonzero (51), X9 lane permutation at the in-loop flush (6.1, only the past-window random cases), XCa guard `n < N` (24), XCb j_begin rounded down (36), XCc widened entry ignores its range (34). X3c (AVX-512 panel pair swapped) is killed on the auto binary and survives forced AVX2, as it must: it mutates the AVX-512 driver only |
| MSVC AVX-512 switch and pure selector (4.7(a)) | done | 3c9a7cf | 32-row selector table plus the dispatch wiring. XS (switch term deleted) and XS2 (whole MSVC term deleted) killed. X4.7b (selector result ignored) killed. **X4.7a (`false` passed for `is_msvc_build`) survives on GCC, as it must: it is equivalent on every non-MSVC build. It is killed only by the wiring assertion on an MSVC build: box-only (Windows)** |
| Tiled-entry counter (11.1), threshold (4.1) | done | 3c9a7cf | D∞ (threshold `SIZE_MAX`): 23 checks red on both binaries. X11.1a (count per panel) and X11.1b (count on the other tier's counter): 11 red each |
| First tiled call race (3.5) | done | 3c9a7cf | new crash-probe `matmul_first_tiled_call_race`: 8 threads each make the process's first GEMM call at M = 8; every output 70, tier probe run exactly once, 8 tiled entries. Red under X3a (68) and X3c (0). In the auto binary only (forced binaries have no probe to race), like the cell it extends |
| Tiled golden pin (6.1) | done | 3c9a7cf | `tools/gen_matmul_tiled_golden.py` → `tests/matmul_tiled_golden_pin.h`: 41 cases, 127,600 bytes, 25 leave int32; hash `b7c5b06c…` computed from the scalar construction's exact sums, matched by all four binaries |
| Digest cases (6.2) | done | 3c9a7cf | new section `c17_matmul_tiled` (M ∈ {8, 32} × N ∈ {32, 48} × K ∈ {17, 64, 1023}), `aac2f53a…`, 9,600 values, equal on all five digests (auto, scalar, SSE2, AVX2, AVX-512 forced); new GLOBAL `18e45694…`. **Sections 1–8 are byte-identical to v1.9.0's** (`c17_matmul` still `ee456f50…`). Under X3b both matmul sections go LOCAL-FAIL |
| Other T cells: 1.1, 4.2, 4.3, 4.4, 5.1, 6.3 | done | 3c9a7cf | 5.1 includes the ABI leg: a 12-token prefill with the tiled scratch allocation failing returns `SSLM_ALLOCATION_FAILED`, commits no state, and the retry's save blob equals a never-failed run's |
| 4.6: every T cell under forced AVX2 and forced AVX-512 on every toolchain | GCC done; rest box-only | | GCC 13 forced AVX2 and AVX-512 green here. Clang 18 legs run in CI (Linux). MSVC legs: box-only (Windows) |
| Build-configuration record | done | 3c9a7cf | `superslm_build_config_record` appended to `src/matmul.cpp` as `probes/route_e/standin_buildcfg.patch` writes it (`SSLM-BUILDCFG/2`, D:/U fields, `used, retain`, MSVC `/INCLUDE`). A suite cell reads it back (marker, force macro, `TILED_MIN_TOKENS=U`, seam). Its E3 reading is exercised with route E at S1-C |
| Windows forced legs | written; box-only | 3c9a7cf | `.github/workflows/tests.yml`: `windows-msvc-avx2-forced` and `windows-msvc-avx512-forced` (suite + digest; the AVX-512 leg builds with `SUPERSLM_TILED_AVX512_MSVC=1` and probes `IsProcessorFeaturePresent(41)` first), both added to `axis-digest-compare`. Not runnable on this host |
| Symbol check (11.3) | done (Linux); Windows box-only | 3c9a7cf | `tools/ci/check_tiled_matmul_linkage.py` plus CI job `tiled-matmul-linkage`: OK on the auto, AVX2-forced and AVX-512-forced objects (record ×1, external; `TiledGemmAvx2/512` local; micro-kernels, packer, transpose, widen and `RunTiledGemm` proven inlined by their signature instructions). Vitality: plant XL (an external tiled template instantiation, nm `W`) FAIL; the non-static packer table (nm `R`) FAIL; a stale listed name FAIL. `dumpbin` (Windows) not implemented: box-only |
| Named-branch coverage (11.4) | written; box-only (CI) | 3c9a7cf | `tools/ci/check_tiled_matmul_coverage.py`, wired into the `branch-coverage` job (the job now saves the AVX-512 binary's exit code for it). **Not executed here: this host's clang-18 has no profile runtime (`libclang_rt.profile` is absent) and packages cannot be installed**, so no instrumented binary links. Its vitality legs (D∞ profile, misspelt anchor) are owed on the first CI run or on the box |
| Isolation-prose scrub (11.2) | done | 3c9a7cf | `tools/ci/check_matmul_avx_isolation.py`'s two "ONLY two functions" sites now name the tiled functions; the checker stays green |
| Other CI checkers | done | | recipe checkers, ABI header inventory, bad_alloc contract, stdout, CI-claims, GPU and forward-leaf checks all green. The `pytest` suites under `tests/ci/` were not run: pytest is not installed here |
| 9.1 blob protocol | done | 98c3379 | `docs/tiled-matmul-slice1/s1-b/blob-protocol-9.1.txt`: `tools/t2147_chunk_batched_pins.cpp` built against v1.9.0's `libsuperslm.a` and against this branch (auto, and forced AVX2), `--dump-blob` at chunk_budget = N. Save blobs byte-equal on the in-tree fixture (plain and DGC1; 8, 20 and 48 tokens, 8 layers), the wide fixture (8, 32, 128 tokens) and the 0.5B-width 1-layer artifact (8, 32, 128). The seam build reports 56 tiled entries per 8-layer prefill (7 GEMMs x 8 layers) on the tier it runs |

## S1-C: commissioning

Evidence: `docs/tiled-matmul-slice1/s1-c/`. Host timings are best of N on a shared host.

| Item | State | Commit | Evidence |
|---|---|---|---|
| 10.1 per-shape GEMM bench and decision rule | done, PASS | 98c3379 | `tools/sslm_gemm_bench.cpp`, `tools/sslm_gemm_bench_rule.py`; `rule-10.1.txt`. Forced AVX2, production against D∞, 1.5B gate/up, M = 32, n = 15 interleaved pairs (best of 5 calls per point): **median 2.048, Q1 1.986, Q3 2.056, spread 0.034 → PASS** (floor 1.4). Must-reject, D∞ against D∞: median 1.012, Q1 1.001 → **FAIL, as required**. Report only, forced AVX-512 against its D∞: median 2.144 → PASS. The floor and must-reject are the plan's; §16 wants them authored by a seat other than the builder, which they were (the plan fixed them) |
| Per-GEMM speedups, AVX2 and AVX-512, M = 8/32/128/512 | done | 98c3379 | `pergemm.md`: 15 shapes (0.5B, 0.6B, 1.5B: q, k/v, o, gate/up, down) × 4 M. Tiled against the shipped loop on the same tier: AVX2 1.67–2.99×, AVX-512 1.63–4.27×; best of 3 interleaved passes × 5 calls. The two maxima are 1.5B `down` at M = 512, where the shipped loop slows (445 ms against about 308 ms scaled from M = 128), not where the tiled kernel is fastest. An M = 8 rerun (2026-09-29) lands at 1.53–2.39× AVX2, 1.56–2.25× AVX-512 |
| Whole prefill before/after | done (0.5B width 1 and 2 layers, 0.6B width 1 layer) | the S1-C commit (the one after 98c3379) | `prefill-*.md`: the `sslm_bench_prefill` tool built against v1.9.0's library and against this branch's, batched (chunk_budget = N), best of 7 interleaved runs, pins checked every run; save blobs equal before/after on each artifact. 128 tokens, per layer: 0.5B 1-layer auto (AVX-512 here) 117.8 → 68.3 ms (**1.72×**), forced AVX2 161.3 → 82.3 ms (**1.96×**); 0.5B 2-layer 1.63× / 1.61×; 0.6B 1-layer 1.59× / 1.62×. At 32 tokens 1.47–1.63×; at 8 tokens 1.20–1.38×. The 2-layer runs overlapped another artifact build on the host. **Superseded 2026-09-29:** an independent rerun (`prefill-remeasure-2026-09-29.md`) gives **1.54–1.64×** at 128 tokens (best of 7; 1.48–1.67× by median). The 1.96× did not reproduce: its v1.9.0 base (161.3 ms) reran at 129.1 ms, giving 1.54×. The 1.72× reran at 1.54×. The 8- and 32-token figures agree within noise |
| Reduced-layer real-width artifacts | partial | | built with `tools/consumer_reach/synth_artifact.py`: wide 8-layer (1,834 s here against the plan's 124.5 s: the host is shared), 0.5B 1- and 2-layer (515 s, 980 s; 7.4–7.8 GiB peak), 0.6B 1-layer (545 s, 7.6 GiB). **Not built: 0.6B 2-layer and 1.5B 1-layer** (the 1.5B needs 13.3 GiB on a 15 GiB host already running another agent's 6–8 GiB build). The 1.5B shapes are covered per GEMM (10.1 and `pergemm.md`), not by a whole-prefill figure |
| 10.0 consumer reach, route E | done, GREEN | the S1-C commit (the one after 98c3379) | `route-e-10.0.txt`, `tools/consumer_reach/route_e/run_route_e_slice1.sh` on the candidate (98c3379 tagged v1.9.0 in a scratch clone), SuperEmbedder `1ee74bf`, 0.5B-width 1-layer artifact. Leg i (production + seam) PASS: pin verified 1.9.0 @ the candidate, code identity equal to the reference engine, **E6: 7 tiled entries at 28 and at 276 tokens** (one per projection GEMM, one layer). Must-reject ii (D∞) FAIL on E6 alone; ii graded as production FAIL on E3, E6; c (forced scalar) FAIL on E3, E6; a (no seam) FAIL on E3, E4, E6. Leg c's expectation was first written as the stand-in's (E3 alone) and was wrong for the real kernel, which never tiles on the scalar tier; it is corrected in the runner and regraded. The full kill-leg census of `run_route_e_legs.sh` (flag channels, fixtures, Q1 pairs) was not re-run: it is the plan's evidence on the stand-in, and that script still hardcodes the 1.8.0 tags |
| 10.0 route P | not run | | the plugin route. The plan's own reading (§3.6) is that the plugin prefills one token per call (0 of 448 GEMM calls at M ≥ 8), so it gains nothing from slice 1; its H witness belongs to slice 2 |
| platform-support.md rows, release notes | done | | engine-level claim only, with its host and artifact; no consumer claim. Corrected 2026-09-29 to the rerun's whole-prefill range (about 1.5–1.6× at 128 tokens, was 1.7–2.0×) |

## ID-PENDING list

Records this build touches, for minting on the owner's machine. Nothing here was minted.

- `ID-PENDING-1`: plan §17 record 5 (the base) now reads **1.9.0**: the build started from v1.9.0 plus two
  build/CI housekeeping commits, and every bit-identity reference is v1.9.0's. The lead of the build thread
  decided this; the owner's veto applies as to every §17 record.
- `ID-PENDING-2`: the 10.1 reading (the first read of the floor and the must-reject, after which a change to
  the rule is a new fold, per the plan): PASS at median 2.048, must-reject FAIL, on the cloud host.
- `ID-PENDING-3`: plan §17 record 6 (the tiled golden set has its own pin): the pin is
  `tests/matmul_tiled_golden_pin.h`, hash `b7c5b06c…`.
