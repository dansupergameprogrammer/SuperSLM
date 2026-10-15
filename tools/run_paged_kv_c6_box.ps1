# C6 box runner for correctness, 7.4, and the lifecycle baseline/production leg.
#
# 7.9 and 7.9b use tools/run_paged_kv_pairs.py with superslm_pkv_pair, linked
# to the production superslm library. This runner never supplies a decode-speed
# verdict. Each selected leg runs by exact cell id and writes a provenance-bound
# marker only after the process returns. Timing legs record pre/post load and
# offenders. Instrument readings remain quarantined until their registry
# receipts are live.
#
# Usage: tools/run_paged_kv_c6_box.ps1 -BuildDir DIR [-ArtifactDir DIR]
#        [-CorrectnessOnly | -TimingOnly] [-Only lifecycle/C6,7.4/C6]
#        [-Fresh] [-NoBuild]
param(
    [Parameter(Mandatory = $true)][string]$BuildDir,
    [string]$ArtifactDir = '',
    [string]$Model05B = 'D:\hf_cache\superslm_artifacts\example\qwen2.5-0.5b-instruct-cap4096-aex.sslm',
    [string]$Model05BCap32768 = 'D:\hf_cache\superslm_artifacts\example\qwen2.5-0.5b-instruct-cap32768-aex.sslm',
    [string]$Model15B = 'D:\hf_cache\superslm_artifacts\qwen2.5-1.5b-instruct.sslm',
    [string]$ScratchDir = 'D:\_scratch\pkv-c6',
    [switch]$TimingOnly,
    [switch]$CorrectnessOnly,
    [string[]]$Only = @(),
    [string[]]$Commissioned = @(),
    [switch]$Fresh,
    [switch]$NoBuild
)
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'paged_kv_box_common.ps1')
. (Join-Path $PSScriptRoot 'paged_kv_c6_provenance.ps1')
if ($TimingOnly -and $CorrectnessOnly) { throw '-TimingOnly and -CorrectnessOnly exclude each other' }
# The caller's legacy -Commissioned hint confers no standing. The receipt
# verifier alone supplies the commissioned IDs recorded for this invocation.
$verification = @(powershell -NoProfile -File 'D:\Wizard\Claude\Tooling\instrument-commission.ps1' -Verify 2>&1 | ForEach-Object { "$_" })
$Commissioned = Get-PkvRunnerCommissionedIds -Requested $Commissioned -VerifyLines $verification
$Repo = Get-PkvRepo
$RefDir = Join-Path $Repo 'tests\paged-kv\reference'
$C6 = Join-Path $ScratchDir 'c6'
$Logs = Join-Path $C6 'logs'
$Done = Join-Path $C6 'done'
New-Item -ItemType Directory -Force -Path $ScratchDir, $C6, $Logs, $Done | Out-Null
$ScratchDir = (Resolve-Path -LiteralPath $ScratchDir).Path
$BaselineLife = Join-Path $C6 'lifecycle_v1.11.0.txt'
$Artifact05BCap32768 = 'qwen2.5-0.5b-instruct-cap32768-aex.sslm'

# ---- the legs ---------------------------------------------------------------------------------
$legs = @(
    [pscustomobject]@{ Name = '10.1/C6'; Kind = 'correctness'; Bin = 'pkv_c6'; Cell = '10.1/C6'; Rests = @('oracle') },
    [pscustomobject]@{ Name = '10.2/C6'; Kind = 'correctness'; Bin = 'pkv_c6'; Cell = '10.2/C6'; Rests = @('oracle') },
    [pscustomobject]@{ Name = '10.4/C6'; Kind = 'correctness'; Bin = 'pkv_c6'; Cell = '10.4/C6'; Rests = @('admission') },
    [pscustomobject]@{ Name = '10.6/C6'; Kind = 'correctness'; Bin = 'pkv_c6'; Cell = '10.6/C6'; Rests = @('admission', 'oracle') },
    [pscustomobject]@{ Name = '10.8/C6'; Kind = 'correctness'; Bin = 'pkv_c6'; Cell = '10.8/C6'; Rests = @('admission') },
    [pscustomobject]@{ Name = '7.4/C6'; Kind = 'timing'; Bin = 'pkv_c6'; Cell = '7.4/C6'; Rests = @('timing74') },
    [pscustomobject]@{ Name = 'lifecycle/C6'; Kind = 'timing'; Bin = 'pkv_life_prod'; Cell = 'lifecycle/C6'; Rests = @('lifecycle') }
)
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
}

# ---- artifacts: one directory holding both, under the names the cells read ----------------------
if ($ArtifactDir) {
    $ArtDir = (Resolve-Path -LiteralPath $ArtifactDir).Path
} else {
    $ArtDir = Join-Path $ScratchDir 'artifacts'
    New-Item -ItemType Directory -Force -Path $ArtDir | Out-Null
    $copies = @(@($Model05B, $PkvArtifact05B), @($Model15B, $PkvArtifact15B))
    if ($legs | Where-Object { $_.Name -eq '7.4/C6' }) { $copies += ,@($Model05BCap32768, $Artifact05BCap32768) }
    foreach ($pair in $copies) {
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
if (($legs | Where-Object { $_.Name -eq '7.4/C6' }) -and
    -not (Test-Path -LiteralPath (Join-Path $ArtDir $Artifact05BCap32768) -PathType Leaf)) {
    throw "7.4 requires the same-model cap-32768 artifact: $(Join-Path $ArtDir $Artifact05BCap32768)"
}
# The correctness cells read the box's R0 references; without them they fail on a missing file.
function Get-RefPath($f) { return Join-Path $RefDir ('v1.11.0_' + [IO.Path]::GetFileNameWithoutExtension($f) + '.ref') }
if ($legs | Where-Object { $_.Kind -eq 'correctness' }) {
    foreach ($f in @($PkvArtifact05B, $PkvArtifact15B)) {
        $ref = Get-RefPath $f
        if (-not (Test-Path -LiteralPath $ref)) { throw "missing ${ref}: run tools\build_paged_kv_reference.ps1 -ArtifactDir $ArtDir first" }
    }
}

# ---- provenance: what each leg's record holds (tools/paged_kv_c6_provenance.ps1) ----------------
$Head = & git -C $Repo rev-parse HEAD
if ($LASTEXITCODE -ne 0 -or -not $Head) { throw "git rev-parse HEAD failed in $Repo" }
$Head = "$Head".Trim()
$shaCache = @{}  # artifacts and references only: a baseline can be rewritten by an earlier leg of this run
function Get-CachedSha($path) {
    if (-not $shaCache.ContainsKey($path)) { $shaCache[$path] = Get-PkvSha256 $path }
    return $shaCache[$path]
}
function Get-ShaOrMissing($path) { if (Test-Path -LiteralPath $path -PathType Leaf) { return Get-PkvSha256 $path } else { return 'missing' } }
function Get-LegInputs($leg) {
    $in = [ordered]@{}
    foreach ($f in @($PkvArtifact05B, $PkvArtifact15B)) { $in["artifact:$f"] = Get-CachedSha (Join-Path $ArtDir $f) }
    if ($leg.Name -eq '7.4/C6') { $in["artifact:$Artifact05BCap32768"] = Get-CachedSha (Join-Path $ArtDir $Artifact05BCap32768) }
    if ($leg.Kind -eq 'correctness') {
        foreach ($f in @($PkvArtifact05B, $PkvArtifact15B)) { $r = Get-RefPath $f; $in["reference:$(Split-Path -Leaf $r)"] = Get-CachedSha $r }
    }
    if ($leg.Name -eq 'lifecycle/C6') { $in["baseline:$(Split-Path -Leaf $BaselineLife)"] = Get-ShaOrMissing $BaselineLife }
    return $in
}
# The record this invocation would write for $leg, with the binary as it stands now.
function Get-LegRecord($leg) {
    $bin = $exe[$leg.Bin]
    $sha = if ($bin -and (Test-Path -LiteralPath $bin -PathType Leaf)) { Get-PkvSha256 $bin } else { 'missing' }
    $manifestPath = "$bin.source-set.json"
    $manifestSha = if ($bin -and (Test-Path -LiteralPath $manifestPath -PathType Leaf)) { Get-PkvSha256 $manifestPath } else { 'missing' }
    $bound = @($Commissioned)
    foreach ($kind in @('timing74', 'lifecycle')) {
        if ($leg.Rests -contains $kind -and $bound -contains $kind) {
            $id = $PkvC6Instruments[$kind].Registry
            if (-not (Test-PkvSourceBinding -Binary $bin -RegistryId $id -Repo $Repo)) {
                Write-Host "SOURCE-QUARANTINE $($leg.Name): receipt $id does not bind the invoked source set"
                $bound = @($bound | Where-Object { $_ -ne $kind })
            }
        }
    }
    $hook = ''
    $prefix = if ($leg.Name -eq '7.4/C6') { 'SUPERSLM_PAGED_KV_74_' }
              elseif ($leg.Name -like 'lifecycle*') { 'SUPERSLM_PAGED_KV_LIFE_' } else { '' }
    if ($prefix) {
        $hook = (@(Get-ChildItem Env: | Where-Object { $_.Name.StartsWith($prefix) -and
            $_.Name -ne 'SUPERSLM_PAGED_KV_LIFE_COMMISSIONED' } | Sort-Object Name |
            ForEach-Object { "$($_.Name)=$($_.Value)" }) -join ';')
        if ($hook) { $bound = @($bound | Where-Object { $_ -notin @('timing74', 'lifecycle') }) }
    }
    return New-PkvLegRecord -Leg $leg.Name -Rests $leg.Rests -Head $Head -Binary "$bin" -BinarySha256 $sha `
        -SourceManifestSha256 $manifestSha -Inputs (Get-LegInputs $leg) -Commissioned $bound -HookState $hook
}
function Test-LegCurrent($leg) { return Test-PkvLegRecordCurrent (Read-PkvLegRecord (Get-MarkerPath $leg)) (Get-LegRecord $leg) }

# ---- builds (all before any leg, so no compiler runs during a timing leg) ----------------------
# Every selected leg's binary is needed, current or not: its sha256 is part of the record a resume
# compares. superslm_pkv_c6 is built incrementally (an unchanged tree relinks nothing, so its sha256
# holds); a harness .exe is rebuilt only when a leg that uses it is not current, since every cl run
# writes a new binary.
$exe = @{}
$needs = @{}
foreach ($leg in $legs) { $needs[$leg.Bin] = $true }
if ($needs.Count -gt 0) { Enter-PkvDevShell }
if ($needs['pkv_c6'] -or $needs['pkv_life_prod']) {
    if (-not (Test-Path -LiteralPath (Join-Path $BuildDir 'CMakeCache.txt'))) {
        if ($NoBuild) { throw "$BuildDir is not a configured build directory" }
        & cmake -S $Repo -B $BuildDir -G Ninja -DCMAKE_BUILD_TYPE=Release -DSUPERSLM_BUILD_GPU=OFF | Out-Host
        if ($LASTEXITCODE -ne 0) { throw 'configure failed' }
    }
    $BuildDir = (Resolve-Path -LiteralPath $BuildDir).Path
    $targets = @()
    if ($needs['pkv_c6']) { $targets += 'superslm_pkv_c6' }
    if ($needs['pkv_life_prod']) { $targets += 'superslm_pkv_lifecycle_prod' }
    if (-not $NoBuild) {
        & cmake --build $BuildDir --target @targets | Out-Host
        if ($LASTEXITCODE -ne 0) { throw "build of $($targets -join ', ') failed" }
    }
    $exe['pkv_c6'] = @((Join-Path $BuildDir 'superslm_pkv_c6.exe'), (Join-Path $BuildDir 'Release\superslm_pkv_c6.exe')) |
        Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1
    if ($needs['pkv_c6'] -and -not $exe['pkv_c6']) { throw "superslm_pkv_c6.exe not found under $BuildDir" }
    $exe['pkv_life_prod'] = @((Join-Path $BuildDir 'superslm_pkv_lifecycle_prod.exe'), (Join-Path $BuildDir 'Release\superslm_pkv_lifecycle_prod.exe')) |
        Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1
    if ($needs['pkv_life_prod'] -and -not $exe['pkv_life_prod']) { throw "superslm_pkv_lifecycle_prod.exe not found under $BuildDir" }
}
# ---- idle check -------------------------------------------------------------------------------
function Get-IdleReport {
    $offenders = @()
    if (Test-Path -LiteralPath 'D:\_ssu_build_lock') { $offenders += 'D:\_ssu_build_lock present' }
    $procs = @(Get-Process -Name 'UnrealEditor', 'UnrealEditor-Cmd', 'cl', 'link', 'dotnet',
        'UnrealBuildTool', 'smoke_adapter_driver', 'superslm_pkv_*' -ErrorAction SilentlyContinue)
    foreach ($p in $procs) { $offenders += "process $($p.ProcessName) (pid $($p.Id))" }
    $load = @()
    for ($i = 0; $i -lt 3; ++$i) {
        $load += [int](Get-CimInstance Win32_Processor | Measure-Object -Property LoadPercentage -Average).Average
        Start-Sleep -Seconds 1
    }
    return [pscustomobject]@{ Idle = ($offenders.Count -eq 0); Offenders = $offenders; Load = ($load -join ',') }
}

# ---- run --------------------------------------------------------------------------------------
$envNames = @('SUPERSLM_PAGED_KV_REAL_ARTIFACT_DIR',
    'SUPERSLM_PAGED_KV_LIFE_OUT', 'SUPERSLM_PAGED_KV_LIFE_BASELINE',
    'SUPERSLM_PAGED_KV_LIFE_COMMISSIONED', 'SUPERSLM_PAGED_KV_FIXTURE_DIR',
    'SUPERSLM_PAGED_KV_74_FLOOR_CONTROL_TICKS', 'SUPERSLM_PAGED_KV_74_ORDER_BIAS_TICKS',
    'SUPERSLM_PAGED_KV_74_CAP32768_FILE', 'SUPERSLM_PAGED_KV_74_ADOPT_MUTANT',
    'SUPERSLM_PAGED_KV_74_RESET_MUTANT')
$results = @{}
foreach ($leg in $legs) {
    $marker = Get-MarkerPath $leg
    $record = Get-LegRecord $leg
    $stored = Read-PkvLegRecord $marker
    if (Test-PkvLegRecordCurrent $stored $record) {
        Write-Output "SKIP $($leg.Name): done earlier with this run's provenance ($marker)"
        continue
    }
    if (Test-Path -LiteralPath $marker) {
        $why = if ($stored) { 'differs in ' + ((Get-PkvLegRecordDiff $stored $record) -join ', ') } else { 'holds no provenance record' }
        Write-Output "STALE $($leg.Name): its marker $why; rerunning"
    }
    if ($leg.Name -eq 'lifecycle/C6') {
        if (-not (Test-Path -LiteralPath $BaselineLife)) { $results[$leg.Name] = 'NOT-RUN (no lifecycle-v1.11.0 baseline)'; continue }
    }
    foreach ($n in $envNames) { Remove-Item -Path "Env:\$n" -ErrorAction SilentlyContinue }
    $env:SUPERSLM_PAGED_KV_REAL_ARTIFACT_DIR = $ArtDir
    if ($leg.Name -eq 'lifecycle/C6') { $env:SUPERSLM_PAGED_KV_LIFE_BASELINE = $BaselineLife }
    # The assertion variables come from the record itself, so the record says exactly what was set.
    $assertEnv = Get-PkvRecordEnv $record
    foreach ($k in $assertEnv.Keys) {
        if ($envNames -notcontains $k) { throw "assertion variable $k is not cleared between legs" }
        Set-Item -Path "Env:\$k" -Value $assertEnv[$k]
    }
    # Direct 7.4 acquires the same lock inside its process. Other timing legs
    # retain the runner lock so both direct and orchestrated paths serialize.
    $timingLock = if ($leg.Kind -eq 'timing' -and $leg.Name -ne '7.4/C6') { Enter-PkvTimingLock } else { $null }
    try {
    $log = Get-LogPath $leg
    $before = $null
    if ($leg.Kind -eq 'timing') {
        $before = Get-IdleReport
        $record['LoadBefore'] = "load=$($before.Load); offenders=$($before.Offenders -join ',')"
    }
    Write-Output ("RUN  {0}: {1} ={2} (log {3})" -f $leg.Name, $exe[$leg.Bin], $leg.Cell, $log)
    $t0 = Get-Date
    $code = Invoke-PkvLogged -Exe $exe[$leg.Bin] -Arguments @("=$($leg.Cell)") -Log $log -WorkingDirectory $C6
    $secs = [int]((Get-Date) - $t0).TotalSeconds
    foreach ($n in $envNames) { Remove-Item -Path "Env:\$n" -ErrorAction SilentlyContinue }
    $postRecord = Get-LegRecord $leg
    if (-not (Test-PkvLegRecordCurrent $record $postRecord)) {
        $record['Commissioned'] = ''
        $record['HookState'] = ((@($record['HookState'], 'post-run provenance changed') |
            Where-Object { $_ }) -join ';')
        Add-Content -LiteralPath $log -Value 'SOURCE-QUARANTINE: deciding input changed during leg'
    }
    $after = $null
    $idleNote = ''
    if ($leg.Kind -eq 'timing') {
        $after = Get-IdleReport
        $record['LoadAfter'] = "load=$($after.Load); offenders=$($after.Offenders -join ',')"
        $idleNote = "before: $($record['LoadBefore']); after: $($record['LoadAfter'])"
        Add-Content -LiteralPath $log -Value "IDLE CHECK $idleNote"
    }
    $tail = @(Get-Content -LiteralPath $log -ErrorAction SilentlyContinue)
    $last = $tail | Where-Object { $_ -match '^\d+ cells, \d+ red; \d+ checks, \d+ failures$' } | Select-Object -Last 1
    $verdict = switch ($code) { 0 { 'GREEN' } 1 { 'RED' } default { "CRASH($code)" } }
    if ($last -match '^0 cells') { $verdict = 'NO-CELL' }
    $line = '{0} exit={1} {2} ({3} s) {4} {5}' -f $verdict, $code, $last, $secs, (Get-Date -Format s), $idleNote
    $record.Outcome = $line
    Write-PkvLegRecord $marker $record
    Write-Output "DONE $($leg.Name): $line"
    } finally { if ($timingLock) { Exit-PkvTimingLock $timingLock } }
}
foreach ($n in $envNames) { Remove-Item -Path "Env:\$n" -ErrorAction SilentlyContinue }

# ---- summary ----------------------------------------------------------------------------------
$rows = @()
$allGreen = $true
foreach ($leg in $legs) {
    # A leg this run tried and did not finish reports that, not an older marker; a finished leg reports
    # its stored record. Standing comes from that record alone, never from this run's arguments.
    $record = $null
    if ($results.ContainsKey($leg.Name)) { $state = $results[$leg.Name] }
    else {
        $record = Read-PkvLegRecord (Get-MarkerPath $leg)
        $state = if ($record) { "$($record.Outcome)" } else { 'NOT-RUN' }
    }
    $verdict = ($state -split ' ')[0]
    if ($verdict -ne 'GREEN') { $allGreen = $false }
    $standing = Get-PkvLegStanding -Rests $leg.Rests -Record $record
    $rows += [pscustomobject]@{ Leg = $leg.Name; Kind = $leg.Kind; Verdict = $verdict; Standing = $standing; Detail = $state }
}
$summary = @('C6 box run summary ' + (Get-Date -Format s) + " (repo $Repo, build $BuildDir, artifacts $ArtDir)")
$summary += ($rows | Format-Table -AutoSize -Wrap Leg, Kind, Verdict, Standing, Detail | Out-String -Width 220).TrimEnd()
$summary += ''
$summary += 'Readings (timing figures and diagnostics, from the logs):'
foreach ($leg in $legs) {
    $log = Get-LogPath $leg
    if (Test-Path -LiteralPath $log) {
        Get-Content -LiteralPath $log | Where-Object { $_ -match '^(7\.4|7\.9|10\.6|lifecycle) |^FAIL |^IDLE CHECK' } | Select-Object -First 60 |
            ForEach-Object { $summary += "  [$($leg.Name)] $_" }
    }
}
if (Test-Path -LiteralPath $BaselineLife) { $summary += "lifecycle baseline ($BaselineLife):"; Get-Content -LiteralPath $BaselineLife | ForEach-Object { $summary += "  $_" } }
$summary | Set-Content -LiteralPath (Join-Path $C6 'summary.txt') -Encoding utf8
$summary | Write-Output
if (-not $allGreen) { exit 1 }
