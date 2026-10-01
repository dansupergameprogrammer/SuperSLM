# Paged-KV plan (rev 16.2) §8 and step C6: the real-artifact reference on the Windows box.
#
# The C6 box cells (tests/paged-kv/c6_dim10_cohort.cpp) compare the paged build's tokens with
# tests/paged-kv/reference/v1.11.0_<artifact stem>.ref, the "cohort" scenario's records
# (persona<i>_1456 and persona<i>_1712: each persona's whole 1,200-token prompt prefilled into its
# own v1.11.0 block, no sharing, decoded to 1,456 and 1,712). This script writes those files, the
# box twin of tools/build_paged_kv_reference.sh:
#   1. builds the v1.11.0 library from the tag, in a throwaway worktree under -WorkDir, with MSVC
#      through the repository's CMake (the tree under work is never built here: the flat path
#      through the new code is never the reference);
#   2. builds tests/paged-kv/reference/pkv_reference.cpp against it;
#   3. runs `pkv_reference --only=cohort v1.11.0 <artifact> <out>` on each artifact.
# v1.8.1 is not built: no C6 cell reads a v1.8.1 reference.
#
# Resumable: an artifact whose .ref already exists in -OutDir is skipped, and a run writes
# <out>.partial and renames it only when pkv_reference exits 0, so a crash never leaves a .ref that
# looks finished. Delete a .ref to regenerate it.
#
# Usage (PowerShell, from anywhere):
#   tools\build_paged_kv_reference.ps1 -ArtifactDir D:\_scratch\pkv-c6\artifacts
#   tools\build_paged_kv_reference.ps1 -Artifacts D:\hf_cache\superslm_artifacts\example\qwen2.5-0.5b-instruct-cap4096-aex.sslm,D:\hf_cache\superslm_artifacts\qwen2.5-1.5b-instruct.sslm
# With neither, the box's usual artifact locations are used. Each artifact's file name must be the
# one the cells name (pkv_reference knows those two geometries and refuses any other model).
param(
    [string]$ArtifactDir = '',
    [string[]]$Artifacts = @(),
    [string]$WorkDir = 'D:\_scratch\pkv-c6\ref',
    [string]$OutDir = '',
    [string]$Tag = 'v1.11.0'
)
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'paged_kv_box_common.ps1')
$Repo = Get-PkvRepo
if (-not $OutDir) { $OutDir = Join-Path $Repo 'tests\paged-kv\reference' }
$Artifacts = @($Artifacts | ForEach-Object { $_ -split ',' } | Where-Object { $_ })  # powershell -File passes one string
if ($Artifacts.Count -eq 0) {
    if ($ArtifactDir) {
        $Artifacts = @((Join-Path $ArtifactDir $PkvArtifact05B), (Join-Path $ArtifactDir $PkvArtifact15B))
    } else {
        $Artifacts = @("D:\hf_cache\superslm_artifacts\example\$PkvArtifact05B", "D:\hf_cache\superslm_artifacts\$PkvArtifact15B")
    }
}
foreach ($a in $Artifacts) {
    if (-not (Test-Path -LiteralPath $a -PathType Leaf)) { throw "artifact not found: $a" }
}
New-Item -ItemType Directory -Force -Path $WorkDir, $OutDir, (Join-Path $WorkDir 'logs') | Out-Null
$WorkDir = (Resolve-Path -LiteralPath $WorkDir).Path
$OutDir = (Resolve-Path -LiteralPath $OutDir).Path

$todo = @()
foreach ($a in $Artifacts) {
    $stem = [IO.Path]::GetFileNameWithoutExtension($a)
    $ref = Join-Path $OutDir "${Tag}_$stem.ref"
    if (Test-Path -LiteralPath $ref) {
        Write-Output "SKIP $stem ($ref exists)"
    } else {
        $todo += [pscustomobject]@{ Artifact = (Resolve-Path -LiteralPath $a).Path; Stem = $stem; Ref = $ref }
    }
}
if ($todo.Count -eq 0) { Write-Output 'every reference already exists'; exit 0 }

$lib = Initialize-PkvFrozenLibrary -Tag $Tag -WorkDir $WorkDir
$exe = Join-Path $WorkDir "pkv_reference-$Tag.exe"
# Rebuilt every run (seconds): the generator source may have changed since the last run.
Build-PkvExe -Out $exe -Sources @((Join-Path $Repo 'tests\paged-kv\reference\pkv_reference.cpp')) `
    -Includes @((Join-Path $lib.Src 'include')) -Lib $lib.Lib

$failed = 0
foreach ($t in $todo) {
    $partial = "$($t.Ref).partial"
    Remove-Item -LiteralPath $partial -ErrorAction SilentlyContinue
    $log = Join-Path $WorkDir "logs\$($t.Stem).log"
    Write-Output "RUN $($t.Stem): pkv_reference --only=cohort $Tag $($t.Artifact) (log $log)"
    $t0 = Get-Date
    $code = Invoke-PkvLogged -Exe $exe -Arguments @('--only=cohort', $Tag, $t.Artifact, $partial) -Log $log
    $secs = [int]((Get-Date) - $t0).TotalSeconds
    if ($code -ne 0) {
        Write-Output "FAIL $($t.Stem): exit $code after $secs s; see $log and $log.err"
        Get-Content -LiteralPath "$log.err" -Tail 20 -ErrorAction SilentlyContinue
        $failed++
        continue
    }
    # The header's digest is the artifact's (the cells check it against the file they load).
    $header = Get-Content -LiteralPath $partial -TotalCount 1
    $sha = Get-PkvSha256 -Path $t.Artifact
    if ($header -notmatch "sha256=$sha`$") { Write-Output "FAIL $($t.Stem): header '$header' does not carry the artifact's sha256 $sha"; $failed++; continue }
    $records = @(Get-Content -LiteralPath $partial | Where-Object { $_ -like 'cohort persona*' }).Count
    if ($records -ne 20) { Write-Output "FAIL $($t.Stem): $records cohort records, expected 20"; $failed++; continue }
    Move-Item -LiteralPath $partial -Destination $t.Ref -Force
    Write-Output "DONE $($t.Stem): $($t.Ref) ($records records, $secs s)"
}
if ($failed) { Write-Output "$failed reference(s) failed"; exit 1 }
Write-Output "references written to $OutDir"
