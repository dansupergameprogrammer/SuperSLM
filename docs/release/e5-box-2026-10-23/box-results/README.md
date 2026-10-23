# E5 home-box pieces, 2026-10-23 (partial: stopped at G1)

These pieces follow `../e5-box-runsheet-2026-10-23.md` as written at `569d4b67`. Each one builds the pinned commit
`ce82c898` in a detached worktree. The scratch root is `D:\_scratch\e5-1p14` in place of the sheet's
`D:\_scratch\e5`, per the coordinator's scratch rule; every other path and command is the sheet's (`pieces/hdr.ps1`
holds the header).

| Piece | Result | Key figures |
|---|---|---|
| E0 | PASS | full clone; src `ce82c898`; merge-base `1948e1f2`; v1131 `d697927f`; clean tree; pkv_def `994b58b4…` (`logs/E0-checks.log`) |
| E1 | PASS | MSVC 14.34 Release: configure and build of `superslm_precache_unpinned` both exit 0 |
| E2 | PASS | ctest 5/5: 4 stated ordinals match; locator test PASS; 7.22 reads 9 mention units and 19 label sites (as the cloud record says; the table at the top of the sheet says 17); version matches. The 7.17 dry run prints the same eight verdicts as the cloud, with `accept-one`'s expected "differs" note |
| E3 | PASS | `pkv_def: rule B = 128 … cap 4096, cap_pages 32, last page holds 128`; 9.9/E5 green, 136 checks |
| G1 | **FAIL (the batch call)** | detail below |
| G2–G4 | not run | stopped at the first red |

**G1.** Both cmake commands exit 0: `v1131\build\Release\superslm.lib` and 35 `shaders\*.cso` exist. The third
command is red:
- What happened: `build_paged_kv_gpu.bat` printed `'vswhere.exe' is not recognized`, then the full suite's section
  headers, then `PKVG BUILD: 0 linked`, and exited 0. There was no `LINKED (v1.13.1)` line, and no `--out` folder was
  written.
- Why: cmd's batch argument parser splits an unquoted argument at `=`. PowerShell passes `--out=D:\…` and
  `--capture910-tree=D:\…` without quotes, because they contain no spaces. The batch therefore sees `--out` and the
  path as two words, and neither matches a flag (`build_paged_kv_gpu.bat:52-66`). Both words are appended to
  TARGETS, so TARGETS is not `" capture910"`. The run takes the full-suite path with the default output folder
  (`src\build\pkvg`) and no `CAP910TREE`.
- G3's batch call has the same unquoted `--out=` form.
- Fix (the sheet's, for the test author): pass each `--name=value` argument quoted to cmd, for example
  `'"--out=' + "$E5\pkvg" + '"'`.
- The `vswhere.exe` line is a separate, smaller note: vswhere is not on PATH in this environment.
