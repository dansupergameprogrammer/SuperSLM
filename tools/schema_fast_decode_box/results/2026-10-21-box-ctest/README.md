# MSVC build, placement pin and fast ctest set on the box, 2026-10-21

**Build.** SuperSLM `aa0bc94cf84ebdd479198c772cc0e8fb9244854a` (`claude/project-thread-nu1czu`: the precache merge, TF,
J1-X and the MSVC GEMM pin), MSVC 19.44, Release, Ninja, AMD Ryzen 9 3950X. The box was clear (no editor, build, build
lock or U3 driver).

## The pin

A fresh Release build of `aa0bc94c`, relinked with `/MAP` (`pinmap.txt`). Kernel addresses mod 64:

| Kernel | Expected | Measured |
|---|---|---|
| `GemmInt8AccumulateCols` | 0 | 0 |
| `DotRowAvx2` | 16 | 16 |
| `TiledGemmAvx2` | 32 | 32 |

`.sslmgem$a` (the 16-byte head) sits at section offset 0, and the kernels' `.sslmgem$m` at 0x10. **Pass.**

## The fast ctest set

A CPU build (`BUILD_TESTING=ON`, `SUPERSLM_BUILD_GPU=OFF`) in its own build folder. The two small precache identity cells
(EXCLUDE_FROM_ALL) were built explicitly. The long precache grid was skipped: `superslm_precache_real_two_process`,
`superslm_precache_e2_reference` and `superslm_precache_e3_reference` (the `superslm_precache_e1` target). No timing
bench is a ctest cell. 32 cells, 10-minute per-cell timeout (`run-ctest.ps1`).

The first pass ran without CI's setup. Seven cells failed: the D1 and F-SFD fixture variables were unset, and the
t2919/t2933 Python cells create temp folders under the source tree's `build\`, which this layout did not have. The seven
were then re-run by name with `tests.yml`'s windows-x64 setup: fixtures from `tools/gen_decode_threading_fixture.py` and
`tools/gen_schema_fast_decode_fixture.py`, plus an empty `build\` (`run-rerun.ps1`).

**Result: 28 pass, 4 fail.**

| Failing cell | What fails | Reading |
|---|---|---|
| `superslm_tests` | 326,327 checks, 3 failures, all in cells 9.2 and 9.X2: "c_abi_verbs.txt does not list sslm_seq_forced_tokens / sslm_seq_jump_forward", and 9.X2 reads the jump verb at line 0, not 46. Every other suite in the binary is green (D1 2,153/0, tiled GEMM 263/0, attn row sites 91,772/0, G3 ledger 93/0). | **Test portability defect.** The file does list both verbs at lines 45 and 46. The cell searches for `"\n" + verb + "\n"`, and this checkout has CRLF (`core.autocrlf=true`, no `.gitattributes` rule for the file), so the match never hits. Fix: normalise `\r\n` when reading, or add `tests/t2807-api-slot/c_abi_verbs.txt eol=lf`. |
| `superslm_schema_fast_decode_alloc` | Cells TF.1, TF.3, TF.4 (a), TF.5, TF.6 and TF.7 all read "RED -- setup failed". The TF sweep has 984 trials, 9,059 setup checks and 18 setup failures, all inside `TF.7 (guard off)`, arm (b), k 4 and up (`sfd_alloc_main.cpp:808`, `:846`, and the TF.3 state check). TF.1d, TF.4 (a) (d), TF.5 (d) and 5.X3 (ii) are green, the J1-X sweep is 944 trials with 0 setup failures, and the alloc sweep is 100 trials with 0 failures. | **Harness defect (likely).** TF.7 runs `TfSweep` with the finish guard stubbed to prove the oracle can fail, and the rest is meant to differ there. But `TfSweep`'s `CHECK_MSG`s are unconditional, so the expected differences count as setup failures and turn every TF cell red. |
| `t2933_schema_query_native_state` | cl: "Cannot open source file: 'COMMON=…\src\spx1_format.cpp'". | **Script defect.** `tests/t2130-g5-red-suite/build_t2922_gpu_schema_accepting_red.bat:35` carries a stray second `COMMON=` token, introduced by `3a00abb6` (one SPX1 writer and reader); `origin/main`'s line is clean. With that token removed locally (not committed), the three cells built and read GREEN, and the script then stopped on `build\gpu-shaders-staged` missing: these are D3D12 bench cells that need a GPU build's staged shaders. CI excludes them too. |
| `t2933_schema_lifecycle_native_state` | Same `.bat` line, then the same missing shader folder. | As above. |

Passing (28): `superslm_g3_core_tests`, the five precache Python checks (version header, its commission, converter hash
guards, SPX1 oracle, status mirrors), `superslm_precache_reference_header_identity`,
`superslm_precache_compiler_job_identity`, `superslm_precache_reference_anchor`, `sha256_portable_forced_tests`,
`gpu_kv_page_addr_tests`, `gpu_kv_pool_geometry_tests`, `superslm_decode_threading_alloc`, `geometry_site_census`,
`check_matmul_avx_isolation`, `check_matmul_avx_isolation_population_vitality`, `t2919_g5_schema_compiler_tests`,
`t2933_schema_query_contract`, `t2933_schema_real_model`, and the nine t2139/t2141 compile gates.
