# Paged-KV plan (rev 16.2) step C6: the box run, one leg at a time, resumable.
#
# Legs, in order (each is one process, with its own log and a done-marker):
#   correctness  10.1/C6 10.2/C6 10.4/C6 10.6/C6 10.8/C6   superslm_pkv_c6 on the real artifacts,
#                against tests/paged-kv/reference/v1.11.0_<stem>.ref (tools/build_paged_kv_reference.ps1)
#   timing       7.4/C6                                     superslm_pkv_c6, caps 4096 and 32768
#                7.9-v1.11.0                                c6_dim7_throughput.cpp built against the frozen
#                                                           v1.11.0 library, SUPERSLM_PAGED_KV_79_OUT=<file>
#                7.9/C6                                     superslm_pkv_c6, SUPERSLM_PAGED_KV_79_BASELINE=<file>
#                7.9-plainlib (with -Also79PlainLibrary)    the same harness file built against the tree
#                                                           under work's production `superslm` library (no
#                                                           test seams), graded against the same baseline
# Every leg selects its cell by exact id ("=10.1/C6"), so no cloud twin runs here.
#
# Resume: a leg that finished (any verdict) writes <Scratch>\c6\done\<leg>.done and is skipped on
# the next run; a crash or a reboot leaves no marker, so the rerun starts at the first unfinished
# leg. -Fresh deletes the selected legs' markers first. A timing leg whose idle check fails before it
# starts is not run; one whose idle check fails after it ends writes no marker (rerun it).
#
# Quarantine (§8): until an instrument is commissioned, its readings are recorded, never acted on or
# headlined. The cells assert the timing verdicts only when SUPERSLM_PAGED_KV_TIMING_COMMISSIONED=1
# (7.9) and SUPERSLM_PAGED_KV_74_MAX_RATIO (7.4) are set; this script sets them only with
# -Commissioned timing (and -Max74Ratio for 7.4). Pass -Commissioned only after the debunker has
# commissioned the instrument through the records tree's instrument-commission.ps1 (-Status reads
# COMMISSIONED). The summary marks each leg's verdict quarantined or standing by the instruments it
# rests on: oracle (the byte-equality oracle: 10.1, 10.2, 10.6), admission (10.4, 10.6, 10.8),
# timing (7.4, 7.9).
#
# Idle checks for timing legs (§8: the same checks as the lifecycle-cost measurement): before and
# after each timing leg, no D:\_ssu_build_lock and none of UnrealEditor, UnrealEditor-Cmd, cl, link,
# dotnet, UnrealBuildTool running; machine-wide CPU load is sampled and logged with them.
#
# Usage:
#   tools\run_paged_kv_c6_box.ps1 -BuildDir D:\_scratch\pkv-c6\build [-ArtifactDir DIR]
#       [-ScratchDir D:\_scratch\pkv-c6] [-CorrectnessOnly | -TimingOnly] [-Only 10.1/C6,7.4/C6]
#       [-Commissioned oracle,admission,timing] [-Max74Ratio 1.5] [-Also79PlainLibrary] [-Fresh] [-NoBuild]
# -BuildDir is a CMake build directory of THIS checkout (configured here with Ninja, Release, when it
# has no CMakeCache.txt); superslm_pkv_c6 is (re)built in it unless -NoBuild. Exit 0 only when every
# selected leg is GREEN.
param(
    [Parameter(Mandatory = $true)][string]$BuildDir,
    [string]$ArtifactDir = '',
    [string]$Model05B = 'D:\hf_cache\superslm_artifacts\example\qwen2.5-0.5b-instruct-cap4096-aex.sslm',
    [string]$Model15B = 'D:\hf_cache\superslm_artifacts\qwen2.5-1.5b-instruct.sslm',
    [string]$ScratchDir = 'D:\_scratch\pkv-c6',
    [switch]$TimingOnly,
    [switch]$CorrectnessOnly,
    [string[]]$Only = @(),
    [string[]]$Commissioned = @(),
    [string]$Max74Ratio = '',
    [switch]$Also79PlainLibrary,
    [switch]$Fresh,
    [switch]$NoBuild
)
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'paged_kv_box_common.ps1')
if ($TimingOnly -and $CorrectnessOnly) { throw '-TimingOnly and -CorrectnessOnly exclude each other' }
# Lists may arrive as one comma-joined string (powershell -File): split them here.
$Commissioned = @($Commissioned | ForEach-Object { $_ -split ',' } | Where-Object { $_ })
foreach ($c in $Commissioned) { if (@('oracle', 'admission', 'timing') -notcontains $c) { throw "-Commissioned takes oracle, admission, timing; not '$c'" } }
$Repo = Get-PkvRepo
$RefDir = Join-Path $Repo 'tests\paged-kv\reference'
$C6 = Join-Path $ScratchDir 'c6'
$Logs = Join-Path $C6 'logs'
$Done = Join-Path $C6 'done'
New-Item -ItemType Directory -Force -Path $ScratchDir, $C6, $Logs, $Done | Out-Null
$ScratchDir = (Resolve-Path -LiteralPath $ScratchDir).Path
$Baseline79 = Join-Path $C6 '79_v1.11.0.txt'

# ---- the legs ---------------------------------------------------------------------------------
$legs = @(
    [pscustomobject]@{ Name = '10.1/C6'; Kind = 'correctness'; Bin = 'pkv_c6'; Cell = '10.1/C6'; Rests = @('oracle') },
    [pscustomobject]@{ Name = '10.2/C6'; Kind = 'correctness'; Bin = 'pkv_c6'; Cell = '10.2/C6'; Rests = @('oracle') },
    [pscustomobject]@{ Name = '10.4/C6'; Kind = 'correctness'; Bin = 'pkv_c6'; Cell = '10.4/C6'; Rests = @('admission') },
    [pscustomobject]@{ Name = '10.6/C6'; Kind = 'correctness'; Bin = 'pkv_c6'; Cell = '10.6/C6'; Rests = @('admission', 'oracle') },
    [pscustomobject]@{ Name = '10.8/C6'; Kind = 'correctness'; Bin = 'pkv_c6'; Cell = '10.8/C6'; Rests = @('admission') },
    [pscustomobject]@{ Name = '7.4/C6'; Kind = 'timing'; Bin = 'pkv_c6'; Cell = '7.4/C6'; Rests = @('timing') },
    [pscustomobject]@{ Name = '7.9-v1.11.0'; Kind = 'timing'; Bin = 'h79_v1110'; Cell = '7.9/C6'; Rests = @('timing') },
    [pscustomobject]@{ Name = '7.9/C6'; Kind = 'timing'; Bin = 'pkv_c6'; Cell = '7.9/C6'; Rests = @('timing') }
)
if ($Also79PlainLibrary) {
    $legs += [pscustomobject]@{ Name = '7.9-plainlib'; Kind = 'timing'; Bin = 'h79_plain'; Cell = '7.9/C6'; Rests = @('timing') }
}
if ($TimingOnly) { $legs = @($legs | Where-Object { $_.Kind -eq 'timing' }) }
if ($CorrectnessOnly) { $legs = @($legs | Where-Object { $_.Kind -eq 'correctness' }) }
if ($Only.Count -gt 0) {
    $names = @($Only | ForEach-Object { $_ -split ',' } | Where-Object { $_ })
    foreach ($n in $names) { if (-not ($legs | Where-Object { $_.Name -eq $n })) { throw "-Only names no selected leg: $n" } }
    $legs = @($legs | Where-Object { $names -contains $_.Name })
}
function Get-MarkerPath($leg) { return Join-Path $Done (($leg.Name -replace '[/\\:]', '_') + '.done') }
function Get-LogPath($leg) { return Join-Path $Logs (($leg.Name -replace '[/\\:]', '_') + '.log') }
if ($Fresh) {
    foreach ($leg in $legs) { Remove-Item -LiteralPath (Get-MarkerPath $leg) -ErrorAction SilentlyContinue }
    if ($legs | Where-Object { $_.Name -eq '7.9-v1.11.0' }) { Remove-Item -LiteralPath $Baseline79 -ErrorAction SilentlyContinue }
}
$pending = @($legs | Where-Object { -not (Test-Path -LiteralPath (Get-MarkerPath $_)) })

# ---- artifacts: one directory holding both, under the names the cells read ----------------------
if ($ArtifactDir) {
    $ArtDir = (Resolve-Path -LiteralPath $ArtifactDir).Path
} else {
    $ArtDir = Join-Path $ScratchDir 'artifacts'
    New-Item -ItemType Directory -Force -Path $ArtDir | Out-Null
    foreach ($pair in @(@($Model05B, $PkvArtifact05B), @($Model15B, $PkvArtifact15B))) {
        $from = $pair[0]
        $to = Join-Path $ArtDir $pair[1]
        if (-not (Test-Path -LiteralPath $from -PathType Leaf)) { throw "artifact not found: $from" }
        if (-not (Test-Path -LiteralPath $to)) {
            try { New-Item -ItemType HardLink -Path $to -Target $from | Out-Null }
            catch { Write-Output "hard link refused ($($_.Exception.Message)); copying $from"; Copy-Item -LiteralPath $from -Destination $to }
        }
        if ((Get-Item -LiteralPath $to).Length -ne (Get-Item -LiteralPath $from).Length) { throw "$to differs in size from ${from}: delete it and rerun" }
    }
}
foreach ($f in @($PkvArtifact05B, $PkvArtifact15B)) {
    if (-not (Test-Path -LiteralPath (Join-Path $ArtDir $f) -PathType Leaf)) { throw "artifact not found: $(Join-Path $ArtDir $f)" }
}
# The correctness cells read the box's R0 references; without them they fail on a missing file.
if ($pending | Where-Object { $_.Kind -eq 'correctness' }) {
    foreach ($f in @($PkvArtifact05B, $PkvArtifact15B)) {
        $ref = Join-Path $RefDir ('v1.11.0_' + [IO.Path]::GetFileNameWithoutExtension($f) + '.ref')
        if (-not (Test-Path -LiteralPath $ref)) { throw "missing ${ref}: run tools\build_paged_kv_reference.ps1 -ArtifactDir $ArtDir first" }
    }
}

# ---- builds (all before any leg, so no compiler runs during a timing leg) ----------------------
$exe = @{}
$needs = @{}
foreach ($leg in $pending) { $needs[$leg.Bin] = $true }
if ($needs.Count -gt 0) { Enter-PkvDevShell }
if ($needs['pkv_c6'] -or $needs['h79_plain']) {
    if (-not (Test-Path -LiteralPath (Join-Path $BuildDir 'CMakeCache.txt'))) {
        if ($NoBuild) { throw "$BuildDir is not a configured build directory" }
        & cmake -S $Repo -B $BuildDir -G Ninja -DCMAKE_BUILD_TYPE=Release -DSUPERSLM_BUILD_GPU=OFF | Out-Host
        if ($LASTEXITCODE -ne 0) { throw 'configure failed' }
    }
    $BuildDir = (Resolve-Path -LiteralPath $BuildDir).Path
    $targets = @()
    if ($needs['pkv_c6']) { $targets += 'superslm_pkv_c6' }
    if ($needs['h79_plain']) { $targets += 'superslm' }
    if (-not $NoBuild) {
        & cmake --build $BuildDir --target @targets | Out-Host
        if ($LASTEXITCODE -ne 0) { throw "build of $($targets -join ', ') failed" }
    }
    $exe['pkv_c6'] = @((Join-Path $BuildDir 'superslm_pkv_c6.exe'), (Join-Path $BuildDir 'Release\superslm_pkv_c6.exe')) |
        Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1
    if ($needs['pkv_c6'] -and -not $exe['pkv_c6']) { throw "superslm_pkv_c6.exe not found under $BuildDir" }
}
if ($needs['h79_v1110'] -or $needs['h79_plain']) {
    # c6_dim7_throughput.cpp calls legacy verbs only, so the same file builds against both libraries
    # (§8: one harness, both binaries, same machine). Headers come from the library's own tree.
    $refHeader = Join-Path $C6 'pkv_refdir.h'
    Set-Content -LiteralPath $refHeader -Encoding ascii -Value ('#define PKV_REFERENCE_DIR "' + ($RefDir -replace '\\', '/') + '"')
    $harness = @((Join-Path $Repo 'tests\paged-kv\pkv_main.cpp'), (Join-Path $Repo 'tests\paged-kv\c6_dim7_throughput.cpp'))
    if ($needs['h79_v1110']) {
        $lib = Initialize-PkvFrozenLibrary -Tag 'v1.11.0' -WorkDir (Join-Path $ScratchDir 'ref')
        $exe['h79_v1110'] = Join-Path $C6 'pkv79_v1.11.0.exe'
        Build-PkvExe -Out $exe['h79_v1110'] -Sources $harness -Lib $lib.Lib -ForceInclude $refHeader `
            -Includes @((Join-Path $lib.Src 'include'), (Join-Path $Repo 'tests\paged-kv'))
    }
    if ($needs['h79_plain']) {
        $plainLib = Join-Path $BuildDir 'superslm.lib'
        if (-not (Test-Path -LiteralPath $plainLib)) { throw "missing $plainLib" }
        $exe['h79_plain'] = Join-Path $C6 'pkv79_plainlib.exe'
        Build-PkvExe -Out $exe['h79_plain'] -Sources $harness -Lib $plainLib -ForceInclude $refHeader `
            -Includes @((Join-Path $Repo 'include'), (Join-Path $Repo 'tests\paged-kv'))
    }
}

# ---- idle check -------------------------------------------------------------------------------
function Get-IdleReport {
    $offenders = @()
    if (Test-Path -LiteralPath 'D:\_ssu_build_lock') { $offenders += 'D:\_ssu_build_lock present' }
    $procs = @(Get-Process -Name 'UnrealEditor', 'UnrealEditor-Cmd', 'cl', 'link', 'dotnet', 'UnrealBuildTool' -ErrorAction SilentlyContinue)
    foreach ($p in $procs) { $offenders += "process $($p.ProcessName) (pid $($p.Id))" }
    $load = @()
    for ($i = 0; $i -lt 3; ++$i) {
        $load += [int](Get-CimInstance Win32_Processor | Measure-Object -Property LoadPercentage -Average).Average
        Start-Sleep -Seconds 1
    }
    return [pscustomobject]@{ Idle = ($offenders.Count -eq 0); Offenders = $offenders; Load = ($load -join ',') }
}

# ---- run --------------------------------------------------------------------------------------
$commissionedTiming = $Commissioned -contains 'timing'
if ($commissionedTiming -and -not $Max74Ratio) { Write-Output 'NOTE: timing commissioned but no -Max74Ratio: 7.4 stays quarantined (the cell asserts only with a ratio)' }
$envNames = @('SUPERSLM_PAGED_KV_REAL_ARTIFACT_DIR', 'SUPERSLM_PAGED_KV_79_OUT', 'SUPERSLM_PAGED_KV_79_BASELINE',
    'SUPERSLM_PAGED_KV_TIMING_COMMISSIONED', 'SUPERSLM_PAGED_KV_74_MAX_RATIO', 'SUPERSLM_PAGED_KV_FIXTURE_DIR')
$results = @{}
$stopTiming = $false
foreach ($leg in $legs) {
    $marker = Get-MarkerPath $leg
    if (Test-Path -LiteralPath $marker) {
        Write-Output "SKIP $($leg.Name): done earlier ($marker)"
        continue
    }
    if ($leg.Kind -eq 'timing' -and $stopTiming) { $results[$leg.Name] = 'NOT-RUN (earlier timing leg not idle)'; continue }
    if ($leg.Name -eq '7.9/C6' -or $leg.Name -eq '7.9-plainlib') {
        if (-not (Test-Path -LiteralPath $Baseline79)) { $results[$leg.Name] = 'NOT-RUN (no 7.9-v1.11.0 baseline)'; continue }
    }
    foreach ($n in $envNames) { Remove-Item -Path "Env:\$n" -ErrorAction SilentlyContinue }
    $env:SUPERSLM_PAGED_KV_REAL_ARTIFACT_DIR = $ArtDir
    if ($leg.Name -eq '7.9-v1.11.0') { $env:SUPERSLM_PAGED_KV_79_OUT = "$Baseline79.partial" }
    if ($leg.Name -eq '7.9/C6' -or $leg.Name -eq '7.9-plainlib') { $env:SUPERSLM_PAGED_KV_79_BASELINE = $Baseline79 }
    if ($commissionedTiming) {
        $env:SUPERSLM_PAGED_KV_TIMING_COMMISSIONED = '1'
        if ($Max74Ratio) { $env:SUPERSLM_PAGED_KV_74_MAX_RATIO = $Max74Ratio }
    }
    $log = Get-LogPath $leg
    $before = $null
    if ($leg.Kind -eq 'timing') {
        $before = Get-IdleReport
        if (-not $before.Idle) {
            Write-Output "NOT IDLE before $($leg.Name): $($before.Offenders -join '; ') (load $($before.Load)%). Timing legs stop here; rerun when the box is quiet."
            $results[$leg.Name] = 'NOT-IDLE'
            $stopTiming = $true
            continue
        }
    }
    Write-Output ("RUN  {0}: {1} ={2} (log {3})" -f $leg.Name, $exe[$leg.Bin], $leg.Cell, $log)
    $t0 = Get-Date
    $code = Invoke-PkvLogged -Exe $exe[$leg.Bin] -Arguments @("=$($leg.Cell)") -Log $log -WorkingDirectory $C6
    $secs = [int]((Get-Date) - $t0).TotalSeconds
    $after = $null
    $idleNote = ''
    if ($leg.Kind -eq 'timing') {
        $after = Get-IdleReport
        $idleNote = "idle before: yes (load $($before.Load)%); idle after: $(if ($after.Idle) { 'yes' } else { 'NO: ' + ($after.Offenders -join '; ') }) (load $($after.Load)%)"
        Add-Content -LiteralPath $log -Value "IDLE CHECK $idleNote"
    }
    $tail = @(Get-Content -LiteralPath $log -ErrorAction SilentlyContinue)
    $last = $tail | Where-Object { $_ -match '^\d+ cells, \d+ red; \d+ checks, \d+ failures$' } | Select-Object -Last 1
    $verdict = switch ($code) { 0 { 'GREEN' } 1 { 'RED' } default { "CRASH($code)" } }
    if ($last -match '^0 cells') { $verdict = 'NO-CELL' }
    if ($leg.Kind -eq 'timing' -and -not $after.Idle) {
        Write-Output "NOT IDLE after $($leg.Name): $($after.Offenders -join '; '). No marker written; rerun this leg."
        $results[$leg.Name] = "$verdict (NOT IDLE after: rerun)"
        $stopTiming = $true
        continue
    }
    if ($leg.Name -eq '7.9-v1.11.0' -and $code -eq 0) { Move-Item -LiteralPath "$Baseline79.partial" -Destination $Baseline79 -Force }
    $line = '{0} exit={1} {2} ({3} s) {4} {5}' -f $verdict, $code, $last, $secs, (Get-Date -Format s), $idleNote
    Set-Content -LiteralPath $marker -Encoding utf8 -Value $line
    Write-Output "DONE $($leg.Name): $line"
}
foreach ($n in $envNames) { Remove-Item -Path "Env:\$n" -ErrorAction SilentlyContinue }

# ---- summary ----------------------------------------------------------------------------------
$rows = @()
$allGreen = $true
foreach ($leg in $legs) {
    $marker = Get-MarkerPath $leg
    $state = if (Test-Path -LiteralPath $marker) { (Get-Content -LiteralPath $marker -TotalCount 1) } elseif ($results.ContainsKey($leg.Name)) { $results[$leg.Name] } else { 'NOT-RUN' }
    $verdict = ($state -split ' ')[0]
    if ($verdict -ne 'GREEN') { $allGreen = $false }
    $missing = @($leg.Rests | Where-Object { $Commissioned -notcontains $_ })
    if ($leg.Name -eq '7.4/C6' -and -not $Max74Ratio) { $missing += 'ratio' }
    $standing = if ($missing.Count -eq 0) { 'standing' } else { 'quarantined (' + ($missing -join ', ') + ')' }
    $rows += [pscustomobject]@{ Leg = $leg.Name; Kind = $leg.Kind; Verdict = $verdict; Standing = $standing; Detail = $state }
}
$summary = @('C6 box run summary ' + (Get-Date -Format s) + " (repo $Repo, build $BuildDir, artifacts $ArtDir)")
$summary += ($rows | Format-Table -AutoSize -Wrap Leg, Kind, Verdict, Standing, Detail | Out-String -Width 220).TrimEnd()
$summary += ''
$summary += 'Readings (timing figures and diagnostics, from the logs):'
foreach ($leg in $legs) {
    $log = Get-LogPath $leg
    if (Test-Path -LiteralPath $log) {
        Get-Content -LiteralPath $log | Where-Object { $_ -match '^(7\.4|7\.9|10\.6) |^FAIL |^IDLE CHECK' } | Select-Object -First 40 |
            ForEach-Object { $summary += "  [$($leg.Name)] $_" }
    }
}
if (Test-Path -LiteralPath $Baseline79) { $summary += "7.9 baseline ($Baseline79):"; Get-Content -LiteralPath $Baseline79 | ForEach-Object { $summary += "  $_" } }
$summary | Set-Content -LiteralPath (Join-Path $C6 'summary.txt') -Encoding utf8
$summary | Write-Output
if (-not $allGreen) { exit 1 }
