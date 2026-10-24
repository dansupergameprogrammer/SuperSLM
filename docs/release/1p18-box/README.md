# 1.18.0 box sitting: run sheet

Plan superslm-prefix-delta-1p18 §8 S3. One sitting of about 20 minutes, Dan nearby (home box: short runs only).
Run it on the merged release tree, after 1.16.0 and 1.17.0 have landed; the status ordinal may have moved then.

**P0. Worktree.** `git -C D:\SuperSLM worktree add D:\_scratch\delta-1p18\src <release-tree commit>` (scratch under
`D:\_scratch\delta-1p18`, per the disk rules). Never on `develop`/`main` directly.

**P1. Run.** `powershell -ExecutionPolicy Bypass -File D:\_scratch\delta-1p18\src\docs\release\1p18-box\run-1p18-box.ps1`

Steps, each resumable (`out\<step>.done`):

| Step | What | Pass |
|---|---|---|
| verbs | regenerate `c_abi_verbs.txt` into `out\` | exactly `sslm_prefix_assemble` and `sslm_prefix_save_delta` added |
| build-msvc | MSVC 14.34 Release: engine, precache E1 and unpinned targets, `superslm_tests` | builds |
| e6-msvc | every E6 cell + `check_spd1.py --self-test` on D15's blobs | all green |
| x2 | export check against the regenerated list | green |
| e1-msvc | the whole precache E1 target under `e4_box_env.ps1` (E2, E3, E6, D12) | green |
| build-clangcl, e6-clangcl | the same build and E6 under clang-cl | green |

**P2. Commit (box session, on the release branch, `[skip ci]`):** copy `out\c_abi_verbs.txt` and
`out\consumed_symbols.txt` over `tests\t2807-api-slot\`; set `build.bat`'s verb count to the new total; set cell
9.X2's expected line of `sslm_seq_jump_forward` to the line the `verbs` step reports (48 on 1.15's list); commit the
`out\status.txt` and logs under `docs\release\1p18-box\results\`.

D12 (the schema-bound delta cell) is not written yet: it needs the synthetic schema fixtures, so it is written with
the box session's fixtures before this sitting, or the e1-msvc step runs without it and the record says so.
