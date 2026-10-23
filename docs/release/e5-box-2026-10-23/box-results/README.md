# E5 home-box pieces, 2026-10-23

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
| G1 | FAIL on the first try, then PASS after the run sheet fix (`9fce4b48`) | detail below |
| G2 | PASS | 9.10 capture on the RTX 2080 SUPER: 43 checks; `captures910\v1.13.1_10de_1e81\` holds `captures.txt` and five `.bin` files; header `writer=v1.13.1 commit=d697927f… adapter=v1.13.1_10de_1e81 fixture=994b58b4…` |
| G3 | PASS | release-tree GPU build; `[E5] pcg_cell_9_10: LINKED`; 1 linked, 0 red |
| G4 | PASS | 9.10 on the RTX 2080 SUPER: 53 checks, 0 failures |

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

**G1 resumed (`logs/G1b-v1131-capture-link.log`).** The sheet at `9fce4b48` quotes each `--name=value` argument.
Re-running only G1's batch call gives `LINKED (v1.13.1): …\pkvg\v1131\pcg_cell_9_10_capture.exe`. The `vswhere.exe`
line is still printed, and it is harmless: the batch file finds VS through `VsDevCmd.bat`'s fixed path.

**R3.** The G2 captures are committed as captured, under
`tests/precache-gpu/captures/v1.13.1_10de_1e81/`. The `.gitattributes` beside them (`* -text`) keeps them
byte-exact on every checkout; without it, Windows checkouts with `core.autocrlf=true` would rewrite `captures.txt`.
Their SHA-256s are in `logs/G2-9.10-capture-2080s.log`.

| File | SHA-256 |
|---|---|
| `captures.txt` | `a44229cb…dc10` |
| `slm5_dedicated_resting.bin` | `95454755…27ec` |
| `slm5_dedicated_mid_token.bin` | `3033593c…63ef` |
| `slm6_pooled_resting.bin` | `b5abb827…9880` |
| `slm6_pooled_mid_token.bin` | `be5f36d2…98b0` |
| `slm6_pooled_adopted_before_first_finish.bin` | `17b48510…16d9` |

The RX 7900 XTX half (W1–W4) is the work machine's. R1 and R2 are the cloud's.
