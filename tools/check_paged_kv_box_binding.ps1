# Exercise only Get-LegRecord, without entering a timing cell or build shell.
param([string]$Runner = (Join-Path $PSScriptRoot 'run_paged_kv_c6_box.ps1'),
      [string]$BuildDir = 'D:\_scratch\T-3114\build-c6-opt')
$ErrorActionPreference = 'Stop'
$source = Get-Content -LiteralPath $Runner -Raw
$start = $source.IndexOf('function Get-LegRecord($leg) {', [StringComparison]::Ordinal)
$end = $source.IndexOf('function Test-LegCurrent', $start, [StringComparison]::Ordinal)
if ($start -lt 0 -or $end -le $start) { throw 'Get-LegRecord boundary missing' }
Invoke-Expression $source.Substring($start, $end - $start)

function Get-PkvSha256([string]$Path) { return 'live-byte-sha' }
function Get-LegInputs($leg) { return [ordered]@{} }
function Test-PkvSourceBinding { return $false }
function New-PkvLegRecord {
    param($Leg, $Rests, $Head, $Binary, $BinarySha256, $SourceManifestSha256,
          $Inputs, $Commissioned, $HookState)
    return @{ Commissioned = @($Commissioned); SourceManifestSha256 = "$SourceManifestSha256" }
}
$exe = @{ pkv_c6 = (Join-Path $BuildDir 'superslm_pkv_c6.exe') }
if (-not (Test-Path -LiteralPath "$($exe.pkv_c6).source-set.json")) { throw 'C6 source manifest missing' }
$Commissioned = @('timing74')
$PkvC6Instruments = @{ timing74 = @{ Registry = 'PKV-TIMING-74-C6-R168' } }
$Head = 'test-head'
$Repo = 'D:\SuperSLM\.worktrees\T-3114'
$leg = [pscustomobject]@{ Name = '7.4/C6'; Bin = 'pkv_c6'; Rests = @('timing74') }
$record = Get-LegRecord $leg
if ($record.Commissioned.Count -ne 0 -or $record.SourceManifestSha256 -ne 'live-byte-sha') {
    throw 'source mismatch still conferred standing or manifest identity was absent'
}
Write-Output 'ok source-mismatched C6 leg quarantined with manifest identity'
