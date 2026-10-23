# G2: 9.1/E2 on MSVC against a GNU cloud blob (one piece, ≤ 10 min) — Dan, home box

Plan rev 1.18.15 §15.35 G2. 9.1 restores a GNU-written `SPX1` blob on the MSVC Release build and requires the MSVC
save of the restored prefix to be byte-equal to the MSVC build's own blob. The GNU blob is checked against its captured
compiler job before it is restored (`e2_composition.cpp:73-106`).

**Bundle:** this folder at `5e57ebffae5095f6a93163120a1f656a958910bb`. It was captured by
`tests/precache/capture_compiler_job.py` at `ce82c898`, the E5 box tree, which reads project 1.13.1. The build was a
clean worktree with a fresh Ninja build, on Linux GCC 13.3. `.gitattributes` keeps every file byte-exact, because
the job hashes the logs and the CMake record.

**Build tree: the E1 tree is enough, with no rebuild.** `D:\_scratch\e5-1p14\build-msvc` is the MSVC 19.34 Release
tree of `ce82c898`, the same commit and project version (1.13.1) as the GNU blob. The blob's producer field must equal
the box build's, so this pairing is the one that restores. E1 built only `superslm_precache_unpinned`, and
`superslm_precache_e1` is `EXCLUDE_FROM_ALL`, so the piece's first step builds that one target in the same tree. The
library and the configure are reused.

A run at the 1.14.0 release commit would need both a fresh MSVC tree at `7585adac` and a GNU bundle captured
there. The engine source is the same apart from the version numbers.

```powershell
. D:\_scratch\e5-1p14\pieces\hdr.ps1
$P = '5e57ebffae5095f6a93163120a1f656a958910bb'
git -C $E5Src fetch origin claude/precache-1p14-post-m1
if (-not (Test-Path "$E5\g2src")) { git -C $E5Src worktree add --detach "$E5\g2src" $P }
git -C "$E5\g2src" rev-parse HEAD                      # must print: 5e57ebffae5095f6a93163120a1f656a958910bb
$G2 = "$E5\g2src\docs\release\e5-box-2026-10-23\g2-91"
Get-Content "$G2\SHA256SUMS.txt" | ForEach-Object {
  $h, $n = $_ -split '\s+', 2
  if ((Get-FileHash -Algorithm SHA256 (Join-Path $G2 $n)).Hash.ToLower() -ne $h) { "MISMATCH $n" } }   # must print nothing
E4-Run (E5-Log 'G2-91-build') ninja @('-C', "$E5\build-msvc", 'superslm_precache_e1')
$env:SUPERSLM_PAGED_KV_FIXTURE_DIR        = $FX
$env:SUPERSLM_PRECACHE_CLOUD_COMPILER     = 'GNU'
$env:SUPERSLM_PRECACHE_CROSS_COMPILER_DIR = $G2
$env:SUPERSLM_PRECACHE_CLOUD_BLOB         = "$G2\GNU.spx1"
$env:SUPERSLM_PRECACHE_RUNNER_EXE         = "$E5\build-msvc\superslm_precache_e1.exe"
E4-Run (E5-Log 'G2-91-run') "$E5\build-msvc\superslm_precache_e1.exe" @('=9.1/E2')
```

**Pass:**
- `rev-parse` prints the commit above, and the hash loop prints nothing.
- The build exits 0.
- The run prints `[E1] 9.1/E2 green (32 checks, 0 failed)` and `1 cells, 0 red; 32 checks, 0 failures`. The cloud
  self-check of this bundle against the GCC runner printed the same; the count should not differ on MSVC.

**Reading a red:**
- `9.1 cloud and box SPX1 bytes differ` is the finding 9.1 exists for: MSVC and GNU write different bytes. Stop and
  send both logs.
- `9.1 cloud blob lacks its verified producer binary, compiler job, or exact captured path` is a bundle or path
  problem: a hash mismatch above, or `CLOUD_BLOB` not equal to `CROSS_COMPILER_DIR\GNU.spx1`. It is not a
  cross-compiler result.
- `4.3 compiled runner executable is missing for provenance` means `RUNNER_EXE` is wrong.
- A restore refusal naming the producer means the build tree is not the 1.13.1 tree.

Resume: re-run the piece. **Receipt:** commit `logs\G2-91-build.log` and `logs\G2-91-run.log` under
`box-results/logs/`.
