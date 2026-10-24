# F2 GPU twins on the RTX 2080 SUPER (home box, 2026-10-23/24)

Engine: SuperSLM `main` at `5599f7e4`, which carries the F2 code (the GPU prefix save and restore, unreleased until
1.16.0). Box: AMD Ryzen 9 3950X, NVIDIA GeForce RTX 2080 SUPER (vendor `10de`, device `1e81`), adapter 0. Built with
MSVC 14.34 and Windows SDK 10.0.22621.0, through `tests/paged-kv-gpu/build_paged_kv_gpu.bat`. Each piece ran alone
under the box's build lock, with no editor running. The logs are beside this file, and `pieces/` holds the driver.

Before this run, the card had the 9.10 captures (`tests/precache-gpu/captures/v1.13.1_10de_1e81`) and the release GPU
build on file. None of the pieces below had run on it.

| Piece | What | Result |
|---|---|---|
| build | engine library and shaders (CMake); `build_paged_kv_gpu.bat F2 pkvg_g4 pkvg_g5` | 9 linked, 0 red by link, 0 red by compile |
| f2 | the F2 cells with the D3D12 debug layer and GPU-based validation: `pcg_cell_format`, `_pool`, `_busy`, `_faults`, `_budget_snapshot`, `_restore_derived` (`--fixture-dir` = the pkv fixtures) | 6 / 6 PASS: 258 + 266 + 23 + 135 + 75 + 53 = **810 checks, 0 failures** |
| schema | `pcg_cell_schema` with `SUPERSLM_PRECACHE_E0_ARTIFACT_DIR` = the E0 debunker's artifacts (debug layer on) | **8 checks, 0 failures** |
| mutant | 11.5 on real hardware: `apply_gpu_mutant.py no-entry-guard` on a copy of `src/`; `pcg_cell_busy` built against it with `--gpu-dir` | **killed**: 21 checks, 3 failures, and the output names 3.4. The worktree's own `src/gpu` was not touched. |
| g5 | `pkvg_g5`, its 24 `[G5]` cells by exact id, debug layer on, `--gpu-ref` = the recorded GPU references | **24 cells, 0 red; 1,074 checks, 0 failures, 0 skips** |
| g4 | `pkvg_g4`, its `[G4]` cells except 1.9 (17), debug layer on, with the g1c adapters | **17 cells, 0 red; 1,418 checks, 0 failures, 0 skips** |
| g19 | cell 1.9 alone, debug layer off | **4,125 checks, 0 failures** (creates 135, adopts 279, prefills 47, resets 118, releases 125, prefix rebuilds 118) |

`pcg_cell_budget_snapshot` (6.8; GPU review V5, a prefix that stops at its budget refreshes its snapshot) is one of the
f2 piece's six cells: 75 checks, 0 failures.

Two runs did not count, and both are in the logs:
- The first mutant build failed to compile because the copied `gpu/` could not reach `../forward/`. The piece now copies
  the whole `src/`.
- The first `pkvg_g4` run hit the 25-minute bound inside 1.9 under GPU-based validation. On this card, 1.9 with the
  debug layer took 90 minutes at 1.13 (2026-10-13), and both earlier box crashes came under that load. So it runs alone
  with the debug layer off, as above. Earlier steps' cells (`[G2]`, `[G3]`) are not re-run: they passed on this card at
  their own releases.

The g1c debug-layer messages were 0 WARNING-or-worse in every cell that reports them. The three `HR FAIL` lines in the
f2 log come from `pcg_cell_faults`' injected allocation failures (E_OUTOFMEMORY), which the cell expects.

## Still owed: the AMD Radeon RX 7900 XTX (work machine)

- The same F2 cells on that card, with the debug layer.
- The cross-card leg: a blob saved on one card is refused `SSLM_RESTORE_PRODUCER_MISMATCH` on the other.
- `pkvg_g4`/`pkvg_g5` at their own steps.
