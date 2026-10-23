# 1.15.0 cloud record: Linux GCC 13.3, 2026-10-23

This is the cloud half of 1.15.0. The box half is `README.md` beside this file.

- **Source tree:** engine `e82d7d40` on `claude/sfd-1p15`, the merge of main 1.14.0 at `0fb84358`.
- **Toolchain:** gcc 13.3.0, Release, 4 cores.
- **Fixtures:** all generated.
  - The paged-KV fixtures (`pkv_def` `994b58b4…`) and the decode-threading and schema fast-decode fixtures.
  - The precache synthetic 0.5B models (from an earlier session, not made by any cloud generator) for the e1 and
    unpinned rows.
- **Version:** 1.14.0 on the branch.

| Run | Result |
|---|---|
| `superslm_tests` | **315,967 checks, 0 failures** (one cell skipped: attn-rowsites 11.1(d), which needs an artifact) |
| Schema fast-decode cells: M1, J1, TF and J1-X (inside `superslm_tests`) | 192,148 checks, 0 failures |
| `superslm_schema_fast_decode_scalar_forced` | 192,148 checks, 0 failures |
| `superslm_schema_fast_decode_alloc` (ctest) | passed |
| 7.22, `check_gpu_prefix_labels.py` | green: 9 mention units, 20 label sites at 1.16.0 |
| Version checks: `check_version_header.py`, `check_version_values.py --expect 1.14.0` | green |
| `check_api_ordinals.py` (7.17's check) | 4 stated ordinals match |
| `check_ci_claims.py` | OK |
| `superslm_precache_e1`, 1.14.0's 58-cell selection | 205,013 checks; 5 red cells, 7 failures, **all environmental**. The same as 1.14.0's record |
| `superslm_precache_unpinned` (27 cells) | 23 green (1.14.0's 23, 9.9/E5 among them). 4 red: the `synthetic1p5` cells, whose model is not in the cloud |
| Paged-KV CPU suites, routine (`run_paged_kv_suites.py`) | r0 5, c2 4, c3 6, c4 27, c5 63 cells, 0 failures, after the `pkv_engine_api.h` fix in `311c6b9a` |
| ctest (35, without `superslm_tests`) | 30 pass. 5 fail, **all environmental**: `real_two_process`, `e2_reference` and `e3_reference` need box references; `t2919` needs the Python `tokenizers` module; `t2933` needs a `build/` folder in the source tree (it passes once there is one) |

The e1 reds:
- 2.13/E2, 2.14/E2, 7.15/E2 and 11.6/E2 need `SUPERSLM_PRECACHE_E0_ARTIFACT_DIR`;
- 5.7/F2:real-gpu needs a D3D12 device.

## The held version bump, applied in scratch

On `dc8e0abe` with `version-bump-1.15.0.patch` applied:
- `check_version_values.py --expect 1.15.0`, `check_version_header.py` and 7.22 are green;
- `superslm_precache_unpinned` matches the branch: 23 green, and the same 4 `synthetic1p5` reds;
- `superslm_tests` has **315,967 checks and 1 failure**.

The failure is **cell 9.X2**:

    [9.X2] the project version is still 1.15.0 (J1's)

The cell is from before rev 1.3.6. It asserts that J1-X bumps the version past J1's 1.15.0. Under rev 1.3.6, J1 and
J1-X ship together as 1.15.0, so the release commit fails this check. The plan's 9.X2 text ("the version is bumped")
has the same stale reading. This needs a ruling before the release commit, and is not changed here.

## Branch coverage (`311c6b9a`)

The job now builds `superslm_precache_e1` and `superslm_precache_unpinned` and runs
`tools/ci/precache_coverage_cells.json` through `tools/ci/run_precache_suites.py`: 35 e1 cells, plus 8.13/E2 and
9.9/E5. A local clang-18 18.1.3 coverage build was checked. `libclang-rt-18-dev` was installed for the profile
runtime.

- **The selection:** 37 cells, 93,601 checks, 0 failures. It takes **75 s** (e1 61.6 s, unpinned 13.7 s) on a loaded
  4-core machine.
- **Its profiles merge and export with the new `-object` entries.** In a partial merge (the non-forced, SSE2 and
  AVX2 suites, with and without the precache profiles), `src/spx1_format.cpp` goes from 0 of 86 branches to 58 of
  86. `src/sslm_abi.cpp` goes from 59.1% to 67.2%, and `src/kv_pages.cpp` from 43.9% to 54.3%.
- **What was not run:** the full five-binary pipeline. Here each instrumented `superslm_tests` takes about 45 min,
  so the run was stopped. The floors are measured on the 1.15.0 tag run, as ruled.
