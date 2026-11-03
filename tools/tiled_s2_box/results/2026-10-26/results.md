# Tiled-matmul slice 2 (threaded CPU batched passes): workhorse box run, 2026-10-26

- Built: SuperSLM `29ecdefaa6e6f4db13cf3ca964c49a5e8ab2bd23` (branch claude/project-thread-ta74yw)
- Compiler: MSVC 19.34.31948 x64 (VS 2022), CMake + Ninja, Release (`-DBUILD_TESTING=ON`)
- Box: workhorse, Ryzen 9 5950X 16C/32T, 64 GB, Windows 11 Pro. CPU only. No msedge/chrome running during T1-T3.
- Model: `qwen2.5-0.5b-instruct-cap4096-aex-pr.sslm`, sha256 `9ccf7e378bbe47aa0ffd3810aa82875926cbe51af1bdfa08393c3fae657463da`
  (the A-EX example artifact with the tokenizer and schemas). It is not in `D:\_artifacts\SuperSLM\aex\`, which holds only
  `-combined.sslm` (no potion_shop_order schema: the bench refuses it). This run read the same-hash copy in
  `D:\_scratch\sfd-1p17-kit\kit-1p17-5060-x\model\`.

## V0 correctness: PASS

Fixtures from `tools/gen_decode_threading_fixture.py` and `tools/gen_schema_fast_decode_fixture.py`, with
SUPERSLM_DECODE_THREADING_FIXTURE_DIR and SUPERSLM_SCHEMA_FAST_DECODE_FIXTURE_DIR set. Both tests run with cwd = repo root.

| binary | checks | failures |
|---|---|---|
| superslm_tests | 332494 | 0 |
| superslm_decode_threading_alloc | 94 | 0 |

The GPU shader-provenance cells need the exe under the source tree with `out\shaders\*.cso` beside it. So
superslm_tests ran as `<worktree>\out\superslm_tests_s2.exe`, with the 35 shaders compiled as build.bat compiles them
(cs_6_2 -O3 -HV 2018 -WX), using this box's dxc 10.0.26100.0. build.bat pins 10.0.19041.0, which is not installed here.

## V1 byte identity: PASS

`extract potion_shop_order --example --max-new 64 --pair off,on:exact --tasks 4 --pairs 3`

| bits | off digest | on:exact digest | runs |
|---|---|---|---|
| 2 | 7e46330608d11704 | 7e46330608d11704 | 3 + 3 |
| 3 | 7e46330608d11704 | 7e46330608d11704 | 3 + 3 |

## T1 Exact gain under threading (fallback pair: off,on:exact)

There is no frozen D2 learned draft table on this box, and no B7 streams file to build one from. Searched: D:\_scratch
(no D:\_scratch\oldbox exists), D:\_artifacts, and the repo. So T1 ran `--pair off,on:exact` with no learned arm.
`--pairs 15 --tasks 4 --max-new 64`. All 60 runs have digest 7e46330608d11704.

| bits | off gen ms (median, q1..q3) | on:exact gen ms | ratio off/on (median, q1..q3) | off prefill ms |
|---|---|---|---|---|
| 2 | 537.43 (534.40..550.62) | 550.73 (544.09..567.05) | 0.980 (0.962..0.995) | 373.05 |
| 3 | 534.20 (526.47..546.98) | 613.01 (610.62..629.53) | 0.862 (0.859..0.866) | 260.71 |

- The ratio is off/on. Below 1 means on:exact is slower than off. At bits 3, on:exact is about 14 % slower; at bits 2,
  about 2 % slower.
- Off's own generation time is about the same at bits 3 as at bits 2 (534.2 vs 537.4 ms, -0.6 %). Prefill is 30 %
  faster at bits 3 (260.7 vs 373.1 ms).

## T2 prompt reading: T(M) ms median (depth 41, --pairs 15)

| M | tasks 4, bits 2 | tasks 4, bits 3 | tasks 8, bits 3 | tasks 16, bits 3 | one-row x M (tasks 4, bits 3) |
|---|---|---|---|---|---|
| 2 | 47.25 | 31.42 | 32.04 | 32.29 | 37.43 |
| 4 | 89.46 | 44.11 | 44.37 | 44.58 | 73.82 |
| 8 | 100.30 | 67.36 | 65.08 | 66.06 | 146.35 |
| 16 | 167.30 | 112.99 | 107.59 | 112.87 | 292.48 |
| 32 | 300.10 | 203.71 | 193.00 | 192.87 | 583.17 |
| 64 | 577.33 | 405.95 | 380.83 | 372.08 | 1171.94 |

## T3 GEMM only: serial vs column split, speedup (24 layers, --pairs 15)

| rows \ tasks | 2 | 4 | 8 | 16 |
|---|---|---|---|---|
| 2 | 1.56 | 1.56 | 1.53 | 1.54 |
| 4 | 1.59 | 2.39 | 2.39 | 2.36 |
| 8 | 1.53 | 2.15 | 2.47 | 2.43 |
| 64 | 1.77 | 3.03 | 4.49 | 4.60 |

outputs_equal = 1 in all 16 cells.

Raw output: `v0/`, `v1/`, `timing/` beside this file (bench --json + logs; T3 is stdout).
