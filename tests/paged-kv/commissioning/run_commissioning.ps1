# Paged-KV plan (rev 16.2) §8: the debunker's commissioning constructions on the box.
#
# Enters the Visual Studio developer shell (tools\paged_kv_box_common.ps1), then runs
# run_commissioning.py beside this file, which builds the pristine and mutant scratch trees under
# -Scratch and grades the named sets by the instruments' own verdicts. The exit status is the
# runner's, which is what instrument-commission.ps1 reads (non-zero = REJECTED); see README.md.
#
# An infrastructure failure here (no developer shell, no Python, no fixtures) must never read as a
# rejection, so it exits 0 for a must-reject or no-result selection (the registry then reads DEAD,
# never a false COMMISSIONED) and 2 for a must-accept selection.
#
# Usage:
#   pwsh -NoProfile -File tests\paged-kv\commissioning\run_commissioning.ps1 -Set oracle-reject
#       [-Scratch D:\_scratch\pkv-commission] [-Fixtures DIR] [-ArtifactDir DIR] [-Python python]
#       [-SubresFactor 0.01] [-Max74Ratio 2.0] [-Construction ID] [-Rebuild] [-BuildOnly] [-List]
param(
    [string[]]$Set = @(),
    [string[]]$Construction = @(),
    [string]$Scratch = 'D:\_scratch\pkv-commission',
    [string]$Fixtures = '',
    [string]$ArtifactDir = 'D:\_scratch\pkv-c6\artifacts',
    [string]$Python = 'python',
    [string]$SubresFactor = '0.01',
    [string]$Max74Ratio = '2.0',
    [switch]$Rebuild,
    [switch]$BuildOnly,
    [switch]$List
)
$ErrorActionPreference = 'Stop'
# Lists may arrive as one comma-joined string (powershell -File): split them here.
$Set = @($Set | ForEach-Object { $_ -split ',' } | Where-Object { $_ })
$Construction = @($Construction | ForEach-Object { $_ -split ',' } | Where-Object { $_ })
$rejectOnly = (($Set + $Construction).Count -gt 0) -and
    (@($Set | Where-Object { $_ -notmatch '-(reject|noresult)(-box|-cloud)?$' }).Count -eq 0) -and
    (@($Construction | Where-Object { $_ -notmatch '\.(R|N)\d+$' }).Count -eq 0)
$infraExit = if ($rejectOnly) { 0 } else { 2 }

try {
    . (Join-Path $PSScriptRoot '..\..\..\tools\paged_kv_box_common.ps1')
    $runner = Join-Path $PSScriptRoot 'run_commissioning.py'
    if ($List) {
        & $Python $runner --list
        exit $LASTEXITCODE
    }
    Enter-PkvDevShell
    New-Item -ItemType Directory -Force -Path $Scratch | Out-Null
    if (-not $Fixtures) { $Fixtures = Join-Path $Scratch 'fixtures' }
    if (-not (Test-Path -LiteralPath (Join-Path $Fixtures 'pkv_def.sslm'))) {
        $gen = Join-Path (Get-PkvRepo) 'tools\gen_paged_kv_fixture.py'
        Write-Output "generating the pkv fixtures into $Fixtures"
        & $Python $gen $Fixtures | Out-Host
        if ($LASTEXITCODE -ne 0) { throw "gen_paged_kv_fixture.py exited $LASTEXITCODE" }
    }
    $pyArgs = @($runner, '--scratch', $Scratch, '--fixtures', $Fixtures, '--subres-factor', $SubresFactor,
        '--max74-ratio', $Max74Ratio)
    if ($ArtifactDir) { $pyArgs += @('--artifacts', $ArtifactDir) }
    foreach ($s in $Set) { $pyArgs += @('--set', $s) }
    foreach ($c in $Construction) { $pyArgs += @('--construction', $c) }
    if ($Rebuild) { $pyArgs += '--rebuild' }
    if ($BuildOnly) { $pyArgs += '--build-only' }
    & $Python @pyArgs
    $code = $LASTEXITCODE
} catch {
    Write-Output "INFRASTRUCTURE FAILURE: $($_.Exception.Message)"
    exit $infraExit
}
exit $code
