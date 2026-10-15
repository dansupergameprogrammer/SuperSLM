# Paged-KV plan rev 16.22 cell 9.3 [T2], the pkv_odd restore side: v1.11.0's reader restores the paged
# build's legacy save of persist saved at cap 4,100 and continues byte-equal to its own continuation.
#   1. superslm_pkv_r9 "=9.3/T2:write" writes the paged build's blob into -OutDir;
#   2. tests/paged-kv/pkv_r9_old_reader.cpp is built against the frozen v1.11.0 library
#      (tools/paged_kv_box_common.ps1 Initialize-PkvFrozenLibrary: a detached worktree of the tag, never
#      the tree under work);
#   3. the reader restores the blob and decodes the 8 tokens of v1.11.0's own "persist continuation8"
#      record (tests/paged-kv/reference/v1.11.0_pkv_odd.ref), and must produce exactly them.
# Exit 0 only when both steps pass.
#
# Usage: tests\paged-kv\r9_old_reader.ps1 -R9Exe <superslm_pkv_r9.exe> -FixtureDir <dir> [-WorkDir <dir>] [-OutDir <dir>]
param(
    [Parameter(Mandatory)] [string]$R9Exe,
    [Parameter(Mandatory)] [string]$FixtureDir,
    [string]$WorkDir = 'D:\_scratch\T-3189\ref',
    [string]$OutDir = 'D:\_scratch\T-3189\out93'
)
$ErrorActionPreference = 'Stop'
$Repo = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
. (Join-Path $Repo 'tools\paged_kv_box_common.ps1')
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
$env:SUPERSLM_PAGED_KV_FIXTURE_DIR = $FixtureDir
$env:SUPERSLM_PAGED_KV_R9_OUT = $OutDir
& $R9Exe '=9.3/T2:write'
$writeExit = $LASTEXITCODE
"9.3/T2:write exit=$writeExit"
$lib = Initialize-PkvFrozenLibrary -Tag 'v1.11.0' -WorkDir $WorkDir
$reader = Join-Path $WorkDir 'r9_old_reader-v1.11.0.exe'
Build-PkvExe -Out $reader -Sources @((Join-Path $PSScriptRoot 'pkv_r9_old_reader.cpp')) -Includes @((Join-Path $lib.Src 'include')) -Lib $lib.Lib
$line = Select-String -Path (Join-Path $Repo 'tests\paged-kv\reference\v1.11.0_pkv_odd.ref') -Pattern '^persist continuation8 ' | Select-Object -First 1
$tokens = ($line.Line -split 'tokens=')[1].Trim()
& $reader (Join-Path $FixtureDir 'pkv_odd.sslm') (Join-Path $OutDir 'pkv_odd_persist_saved_paged.ssb') $tokens
$readExit = $LASTEXITCODE
"old reader exit=$readExit"
if ($writeExit -ne 0 -or $readExit -ne 0) { exit 1 }
exit 0
