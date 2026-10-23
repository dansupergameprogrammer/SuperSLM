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
