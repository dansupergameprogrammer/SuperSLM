# 1.15.0 box run sheet: exactness and the MSVC pin on the release tree

This sheet covers engine 1.15.0, schema skip-ahead (plan rev 1.3.6, the R row). B7 measured exactness and speed on
build `f03a5358` (`tools/schema_fast_decode_box/results/2026-10-21-exact/`). The release tree is
`claude/sfd-1p15`, which merges 1.14.0 into that work. This sheet re-checks the two things the merge could have
moved:

- **B7 exactness.** Exact mode must be byte-identical to off, at 1 task and at 4 tasks.
- **The MSVC GEMM placement.** `run-pinmap.ps1` checks it, and a one-thread 7-token prefill A/B against B7's build
  confirms it in milliseconds.

It does not re-time B7c to B7e. The CPU engine source is the B7 build's, apart from the version header, comments
and 1.14.0's GPU prefix fixes, so B7's timings stand.

**The tree.** The pieces run on the tip of `claude/sfd-1p15`. Its engine tree must equal `e82d7d40`: later commits
change only `docs/`, and the driver refuses any other tree. The version reads **1.14.0**. The bump to 1.15.0 is the
held patch `version-bump-1.15.0.patch` (beside this file), applied only as the release commit, after this sheet is
green.

**Time.** About 25 minutes in all, unattended after P0: the build 8, the pin check 1, B7b at 1 task 7, B7b at 4
tasks 5, and the A/B 4. Add 6 minutes if B7's bench is no longer on disk and the reference has to be built. It is
resumable: each step writes `D:\_scratch\sfd-1p15\out\<step>.done`, and re-running the driver skips the steps that
are done. To re-run one step, delete its `.done` file.

**Box state.** Nothing else should run: no editor, no build and no build lock. The driver refuses to start
otherwise. The A/B is a timing piece, so leave the box alone while it runs.

## P0. Worktree (2 min) — Dan

```powershell
cd D:\SuperSLM
git fetch origin
New-Item -ItemType Directory -Force D:\_scratch\sfd-1p15 | Out-Null
if (-not (Test-Path D:\_scratch\sfd-1p15\src)) { git worktree add --detach D:\_scratch\sfd-1p15\src origin/claude/sfd-1p15 }
git -C D:\_scratch\sfd-1p15\src rev-parse HEAD                       # note it in the record
git -C D:\_scratch\sfd-1p15\src diff --stat e82d7d40 HEAD -- . ':!docs' # must print nothing
git -C D:\_scratch\sfd-1p15\src status --porcelain                  # must print nothing
```

## P1. The driver (about 25 min) — Dan; needs P0

```powershell
powershell -ExecutionPolicy Bypass -File D:\_scratch\sfd-1p15\src\docs\release\1p15-box\run-1p15-box.ps1
```

The driver checks the tree, the model (`9ccf7e37…`), the population (`d846b9e3…`) and the frozen D2 table
(`4cc216d2…`, with any CRLF the checkout added taken out). It then runs five steps and writes everything under
`D:\_scratch\sfd-1p15\out`. `status.txt` is the log. The output ends with `end PASS`, and the exit code is 0.

| Step | What | Pass |
|---|---|---|
| `build` | An MSVC Release Ninja build of `superslm`, `sslm_schema_fast_decode_bench` and `sslm_decode_threading_bench` in `D:\_scratch\sfd-1p15\build-msvc` | exit 0 |
| `pinmap` | `tools/msvc_pin/run-pinmap.ps1` on that build | `GemmInt8AccumulateCols` 0, `DotRowAvx2` 16, `TiledGemmAvx2` 32 mod 64; `PIN OK` |
| `b7b-t1` | B7b at one thread. The 40 population prompts and B2's prompt each run off and three exact arms: the learned draft, the NULL draft and the always-wrong draft. One process per prompt | `41 prompts, 123 exact arms, 0 not identical to off`; B2's off digest `7e46330608d11704`, 28 tokens |
| `b7b-t4` | The same at 4 tasks | the same line |
| `abp` | The one-thread 7-token prefill (`pass --example --sizes 7 --no-one-row`): the release bench against B7's timed bench (sha256 `fcba88db…`; when it is missing, `aa0bc94c` is built fresh), 15 rounds after a warm-up, alternating order | release/ref median within 0.97 to 1.03, and release near 145 ms. The slow placement read about 185 ms (1.28x) |

**If `b7b-*` stops** with `an exact arm differs from off`, the release is blocked. Keep `out\b7b-*\` and send it
back: `summary.json` names the prompt and the arm. **If `pinmap` reports MOVED or MISSING**, or `abp` is outside
the band, the placement changed. Keep `out\pinmap\` and `out\abp\` and send them back. Do not re-pin on the box.

## Record

Send back these files:

- `D:\_scratch\sfd-1p15\out\status.txt`;
- `out\pinmap.log`;
- `out\b7b-t1\summary.json` and `out\b7b-t4\summary.json`;
- `out\abp\summary.json`.

They are committed under `docs/release/1p15-box/box-results/`. After that, the release commit applies
`version-bump-1.15.0.patch` with the date filled in (`@RELEASE_DATE@`, two places).

## The held version bump

`version-bump-1.15.0.patch` sets 1.15.0 in these files:

- `CMakeLists.txt`;
- `include/superslm/sslm_version.h`;
- `tests/install-consumer/CMakeLists.txt`.

It also turns the CHANGELOG's `[Unreleased]` entry into `## [1.15.0] - @RELEASE_DATE@`, dates the release note, and
updates the README's current-release paragraph. It applies with `git apply --check` on `e82d7d40`. On a scratch apply,
`tests/precache/check_version_values.py --expect 1.15.0` and `tests/ci/check_version_header.py` pass. It is not
committed on the branch (R14): until the release commit, dev builds carry 1.14.0, so no save is stamped with an
unreleased producer.
