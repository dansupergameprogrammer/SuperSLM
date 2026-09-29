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
| v1.9.0 digest, forced GCC AVX2/AVX-512 suites and digests on this host | done | (S1-0 commit) | `docs/tiled-matmul-slice1/s1-0-base-v1.9.0/`. All five digests (auto, scalar, SSE2, AVX2, AVX-512 forced) GLOBAL `cc267693…`; `c17_matmul` `ee456f50…`. Suites: auto 25,299 checks, AVX2-forced and AVX-512-forced 25,278 checks each, 0 failures; matmul golden `932478a4…` |
| Token-id mode upstreamed into `tools/t2147_chunk_batched_pins.cpp` | done | (S1-0 commit) | `ids:N[:V]` prompt, `--layers=L`, `--repeat=R`; also fixes `--boundary-sweep=` reading one character too far |
| Wide fixture and reduced-layer real-width artifacts | pending | | built in scratch with `tools/consumer_reach/synth_artifact.py` (adds the Qwen2.5-0.5B geometry) |
| §11.R guarded route run once against the base | done | (S1-0 commit) | `docs/tiled-matmul-slice1/s1-0-base-v1.9.0/route-e-r11-v1.9.0.txt`: SuperEmbedder `1ee74bf` pins, configures ("pin verified: 1.9.0 @ d870d27…") and builds against v1.9.0 |
| `probes/route_e/` and `probes/route_p_hook/` upstreamed as the 10.0 tool and its checks | done | (S1-0 commit) | `tools/consumer_reach/` (current-revision scripts only; the probes' rev-N history and outputs stay with the plan). The 10.0 tool is `tools/consumer_reach/route_e/route_e_reach.cpp`, not `tools/sslm_consumer_reach.cpp`: it links SuperEmbedder, so no CMake or `build.bat` recipe in this repository can build it, and the runner script is its recipe |
| Baseline whole prefill (reduced-layer artifacts) | pending | | |

## S1-B: the kernel

| Item | State | Commit | Evidence |
|---|---|---|---|
| `Cols` entry over the shipped loop (review W8), both entries | done | (Cols commit) | cell 3.3 green; mutant XC (range ignored) red, 102 of 108 checks |
| Tiled kernel, activation prep, packer | pending | | |
| MSVC AVX-512 switch and pure selector | pending | | |
| Tiled-entry counter (11.1) | pending | | |
| Tiled golden pin (6.1), digest cases (6.2) | pending | | |
| Build-configuration record | pending | | |
| Windows forced legs | pending | | |
| Symbol check (11.3), named-branch check (11.4), isolation-prose scrub (11.2) | pending | | |

## S1-C: commissioning

| Item | State | Commit | Evidence |
|---|---|---|---|
| 10.1 per-shape GEMM bench | pending | | |
| Whole prefill before/after | pending | | |
| 10.0 consumer reach | pending | | |
| platform-support.md rows, release notes | pending | | |

## ID-PENDING list

(none yet)
