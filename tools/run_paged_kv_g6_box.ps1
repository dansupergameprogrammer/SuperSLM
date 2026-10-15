# Paged-KV plan step G6 box runner: the GPU cohort and admission cells (10.3, 10.5, 10.7) and the GPU decode timing
# (§10 R1 as the plan of record states it at rev 16.30: the 0.5B cohort row at width 1,712 and the 1.5B row at width
# 16,384 (rev 16.16), each under cell 7.9's per-process rule (rev 16.18, §20.4), bar no more than 5% slower), on each
# selected GPU, resumably. Windows PowerShell 5.1 compatible; safe to launch from pwsh 7 (see "Launching" below).
#
# Usage: powershell -NoProfile -ExecutionPolicy Bypass -File <repo>\tools\run_paged_kv_g6_box.ps1
#            [-Scratch D:\_scratch\pkv-gpu-113] [-Model05 PATH] [-Model15 PATH] [-FixtureDir DIR] [-BinDir DIR]
#            [-Adapters 0] [-ExpectAdapter 'RTX 2080 SUPER'] [-Rows 0.5B,1.5B] [-SlowArm A|B]
#            [-CorrectnessOnly | -TimingOnly] [-Only 10.3/box@0,R1/1.5B@0] [-Fresh] [-Rebuild | -NoBuild]
#            [-QuietWaitSeconds 900] [-MaxLoadPercent 30] [-PauseFile D:\_scratch\TE-444\u3\PAUSE] [-RegistryId ID]
#            [-Pairs 104] [-Width15 16384]
#
# Launching. Every parameter is a string, an int or a switch, so `-File` binds each one as written: a list is one
# comma-separated string (`-Adapters 0,1`, `-Rows 0.5B,1.5B`, `-Only a,b`), never a PowerShell array, which `-File`
# cannot pass. A switch is written bare (`-TimingOnly`), never `-TimingOnly:$true`. `-File` takes the script's absolute
# path and must be the last powershell.exe option: everything after it goes to the script. Launched from pwsh 7,
# powershell.exe inherits pwsh's PSModulePath, and Windows PowerShell 5.1 would then autoload PowerShell 7's
# CimCmdlets and Microsoft.PowerShell.* first and fail (Get-CimInstance, Get-FileHash); the first statement below
# rebuilds PSModulePath as 5.1's own before anything autoloads. From pwsh 7, also clear it in the launching command:
#     $env:PSModulePath = $null; & "$env:SystemRoot\System32\WindowsPowerShell\v1.0\powershell.exe" -NoProfile `
#         -ExecutionPolicy Bypass -File D:\SuperSLM\tools\run_paged_kv_g6_box.ps1 -TimingOnly -Adapters 0
#
# Legs, per adapter index N in -Adapters (SSLM_GPU_ADAPTER_INDEX=N):
#   10.3/fixture@N 10.3/box@N 10.5/fixture@N 10.5/box@N 10.7/fixture@N 10.7/box@N   (correctness)
#       pkvg_suite.exe --step=G6 =<cell>/<leg>, debug layer ON (pkvg_suite sets SSLM_GPU_ENABLE_DEBUG_LAYER=1:
#       these are acceptance runs). fixture = pkv_def under -FixtureDir; box = the real 0.5B (-Model05). Both read
#       the R0 records in tests\paged-kv\reference.
#   R1/0.5B@N R1/1.5B@N                                                                (timing, -Rows)
#       pkvg_g6_decode_timing.exe, debug layer OFF. One row = five processes on one boot, each its own OS process
#       (`process --process=K`), each timing 104 pairs after 4 warm-up tokens per arm; then `grade` applies the row
#       rule. The runner reads LastBootUpTime before and after every process: a process whose boot changed is
#       discarded (its record deleted) and rerun; the harness itself discards every record of a row that is on
#       another boot, so a row is never mixed across boots. A row that reads NO RESULT gets one rerun of five fresh
#       processes (attempt2), whose verdict is its own and never pooled; a second NO RESULT goes to the planner.
#       R1/1.5B first makes the row's two blobs once (`prepare`: the 16,380-token prompt prefilled into a pooled
#       holder and a dedicated cap-size holder, saved as SLM6 and SLM5, under <Scratch>\timing-r1\blobs), which every
#       process then restores. Records: <Scratch>\timing-r1\<arm>\<row>\adapterN\attempt<k>\ (run.txt, process_K.log,
#       result.txt), <arm> = plan, or slow-arm-A / slow-arm-B under -SlowArm. After the timing legs, R1 over both rows
#       is written per adapter to <Scratch>\timing-r1\<arm>\r1_adapterN.txt (`verdict`: PASS when both rows read
#       PASS, FAIL when either reads FAIL, NO RESULT otherwise).
#       The rev 16.2 harness's records (one process, 508 pairs a repeat; <Scratch>\timing, done\R1_N.done) are left as
#       they were: under the per-process rule no pair of them is a process of a row, so none is reused.
# Every correctness leg runs before any timing leg, and every build before any leg, so no compiler or GPU suite runs
# while a timing leg does.
#
# Mechanism runs. -Pairs (default the plan's 104) and -Width15 (default the plan's 16,384) exist for smoke runs on tiny
# fixtures: any other value stamps every row MECHANISM_ONLY (a reading, never a verdict) and keeps its records apart
# (<arm> gains -mech-p<Pairs>-w<Width15>).
#
# Commissioning arm. -SlowArm A sets SUPERSLM_PKVG_G6_SLOW_ARM=A for the timing legs: the paged arm busy-waits 17.65% of
# each timed call, s near +0.1765, so each row must read FAIL (the must-reject, at the 1.5B row as at the 0.5B); -SlowArm
# B slows the legacy arm and each row must read PASS (a must-accept). The rows' status is then MUTANT_CONSTRUCTION; their
# records and markers are kept apart from the plan's, and the harness's `grade`/`verdict` exit 1 on a mutant's FAIL or
# NO RESULT, 0 on its PASS. -SlowArm selects the timing legs only. The 1.5B blobs are shared with the plan's rows.
#
# Resume: each leg writes <Scratch>\done\<leg>.done only after it finishes, holding its provenance (HEAD, the
# executables' and inputs' SHA-256, the adapter, the commissioning state, the slow arm) and its outcome. A leg whose
# marker matches the provenance this run would record is skipped; -Fresh deletes the selected legs' markers (and, for
# timing legs, their row records; never the blobs, which the harness checks against their manifest) first. Inside a
# timing leg every completed process is kept by the harness and every made blob by `prepare`, so an interrupted leg
# resumes at the next process. Logs: <Scratch>\logs\<leg>[.<step>].log and .log.err; summary: <Scratch>\summary.txt.
#
# Timing discipline. Before the first timing leg the runner creates -PauseFile (the other box worker's pause marker)
# and checks -ExpectAdapter against each timing adapter's DXGI description (R1 is read on the RTX 2080 SUPER; pass
# -ExpectAdapter '' to time another card). Before each process it waits up to -QuietWaitSeconds for a quiet box and
# refuses the process (NOT-QUIET, the leg left resumable) unless the box is quiet: no build lock
# (D:\_ssu_build_lock), none of the C6 runners' busy processes or another GPU suite process, and CPU load at most
# -MaxLoadPercent over three one-second samples. The harness refuses a process on the same process list (exit 40) and
# discards one that ends on a busy box. The pause marker is deleted at the end of the timing phase and on any failure
# or interruption (finally); a marker already present when the runner started is someone else's and is left as found.
#
# Commissioning (§8): the harness's readings are quarantined until its instrument is commissioned. The runner passes
# --commissioned only when D:\Wizard\Claude\Tooling\instrument-commission.ps1 -Verify lists "COMMISSIONED  <RegistryId>"
# (default PKV-TIMING-G6-R1; its receipt binds the instrument's bytes, so the rev 16.2 harness's receipt reads STALE on
# this one and this harness's readings stay QUARANTINED until it is commissioned on its own bytes).
#
# Clock on the RTX 2080 SUPER (underived: from the published 1.5B GPU rates, about 63 tok/s decode and batched prefill
# at about the decode rate, and about 8.4 ms a 0.5B token): R1/0.5B about 1 minute a process (the cohort's 3,600
# prefill tokens, 512 decode steps per arm), 5 minutes a row; R1/1.5B's prepare about 10 minutes once (two 16,380-token
# prefills), then about 40 s a process (load, restore, 108 decode steps per arm), 4 minutes a row. One adapter, both
# rows, no rerun: about 20 minutes, plus the quiet checks; each NO RESULT rerun adds that row's minutes again.
param(
    [string]$Scratch = 'D:\_scratch\pkv-gpu-113',
    [string]$Model05 = 'D:\hf_cache\superslm_artifacts\example\qwen2.5-0.5b-instruct-cap4096-aex.sslm',
    [string]$Model15 = 'D:\_artifacts\SuperSLM\t3121\artifacts\qwen2.5-1.5b-instruct.sslm',
    [string]$FixtureDir = '',
    [string]$BinDir = '',
    [string]$Adapters = '0,1',
    [string]$ExpectAdapter = 'RTX 2080 SUPER',
    [string]$Rows = '0.5B,1.5B',
    [string]$SlowArm = '',
    [switch]$CorrectnessOnly,
    [switch]$TimingOnly,
    [string[]]$Only = @(),
    [switch]$Fresh,
    [switch]$Rebuild,
    [switch]$NoBuild,
    [int]$QuietWaitSeconds = 900,
    [int]$MaxLoadPercent = 30,
    [string]$PauseFile = 'D:\_scratch\TE-444\u3\PAUSE',
    [string]$RegistryId = 'PKV-TIMING-G6-R1',
    [int]$Pairs = 104,
    [int]$Width15 = 16384
)
# Windows PowerShell 5.1 under a pwsh 7 parent: rebuild PSModulePath as 5.1's own before any module autoloads. Only .NET
# calls and Core-snap-in language here: a cmdlet from Microsoft.PowerShell.Management or .Utility (Join-Path,
# Select-Object, New-Object) would itself autoload from the inherited path, which is the failure being avoided.
if ($PSVersionTable.PSVersion.Major -lt 6) {
    $candidates = @([IO.Path]::Combine([Environment]::GetFolderPath('MyDocuments'), 'WindowsPowerShell\Modules'),
        [IO.Path]::Combine($env:ProgramFiles, 'WindowsPowerShell\Modules'), [IO.Path]::Combine($PSHOME, 'Modules'))
    $machineModules = [Environment]::GetEnvironmentVariable('PSModulePath', 'Machine')
    if ($machineModules) { $candidates += $machineModules.Split(';') }
    $keep = [System.Collections.Generic.List[string]]::new()
    foreach ($c in $candidates) {
        if ($c -and $c -notmatch '\\PowerShell\\7|\\Documents\\PowerShell\\Modules' -and -not $keep.Contains($c)) { $keep.Add($c) }
    }
    $env:PSModulePath = [string]::Join(';', $keep)
}
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'paged_kv_box_common.ps1')
if ($TimingOnly -and $CorrectnessOnly) { throw '-TimingOnly and -CorrectnessOnly exclude each other' }
if ($SlowArm -and $SlowArm -cne 'A' -and $SlowArm -cne 'B') { throw "-SlowArm takes A or B, not '$SlowArm'" }
if ($SlowArm -and $CorrectnessOnly) { throw '-SlowArm selects the timing legs; it excludes -CorrectnessOnly' }
$Repo = Get-PkvRepo
$RefDir = Join-Path $Repo 'tests\paged-kv\reference'
if (-not $BinDir) { $BinDir = Join-Path $Repo 'build\pkvg\bin' }
$Logs = Join-Path $Scratch 'logs'
$Done = Join-Path $Scratch 'done'
$TimingR1 = Join-Path $Scratch 'timing-r1'
New-Item -ItemType Directory -Force -Path $Scratch, $Logs, $Done, $TimingR1 | Out-Null
$Scratch = (Resolve-Path -LiteralPath $Scratch).Path
$TimingR1 = (Resolve-Path -LiteralPath $TimingR1).Path
$Arm = if ($SlowArm) { "slow-arm-$SlowArm" } else { 'plan' }
$Mechanism = ($Pairs -ne 104 -or $Width15 -ne 16384)
if ($Pairs -lt 1 -or $Width15 -lt 8) { throw '-Pairs >= 1 and -Width15 >= 8' }
if ($Mechanism) { $Arm += "-mech-p$Pairs-w$Width15" }
$mechArgs = @()
if ($Pairs -ne 104) { $mechArgs += "--pairs=$Pairs" }
$ArmRoot = Join-Path $TimingR1 $Arm
$TimingExe = 'pkvg_g6_decode_timing.exe'
# -Adapters is a comma list (a string, so `powershell -File ... -Adapters 0,1` binds under Windows PowerShell 5.1).
$adapterList = @($Adapters -split ',' | ForEach-Object { $_.Trim() } | Where-Object { $_ } | ForEach-Object {
    if ($_ -notmatch '^\d+$') { throw "-Adapters takes adapter indices: '$_'" }; [int]$_ })
if ($adapterList.Count -eq 0) { throw 'no adapter index in -Adapters' }
$rowList = @($Rows -split ',' | ForEach-Object { $_.Trim().ToUpperInvariant() } | Where-Object { $_ } | ForEach-Object {
    if ($_ -ne '0.5B' -and $_ -ne '1.5B') { throw "-Rows takes 0.5B and 1.5B: '$_'" }; $_ } | Select-Object -Unique)
if ($rowList.Count -eq 0) { throw 'no row in -Rows' }

# ---- the legs ------------------------------------------------------------------------------------
$legs = @()
foreach ($a in $adapterList) {
    foreach ($cell in @('10.3', '10.5', '10.7')) {
        foreach ($kind in @('fixture', 'box')) {
            $legs += [pscustomobject]@{ Name = "$cell/$kind@$a"; Kind = 'correctness'; Adapter = $a; Job = "$cell/$kind"; Leg = $kind
                Exe = "g1e_cell_$($cell -replace '\.', '_').exe"; Row = '' }
        }
    }
}
foreach ($a in $adapterList) {
    foreach ($r in $rowList) {
        $legs += [pscustomobject]@{ Name = "R1/$r@$a"; Kind = 'timing'; Adapter = $a; Job = ''; Leg = 'box'; Exe = $TimingExe
            Row = $r.ToLowerInvariant() }
    }
}
if ($TimingOnly -or $SlowArm) { $legs = @($legs | Where-Object { $_.Kind -eq 'timing' }) }
if ($CorrectnessOnly) { $legs = @($legs | Where-Object { $_.Kind -eq 'correctness' }) }
if ($Only.Count -gt 0) {
    $names = @($Only | ForEach-Object { $_ -split ',' } | ForEach-Object { $_.Trim() } | Where-Object { $_ })
    foreach ($n in $names) { if (-not ($legs | Where-Object { $_.Name -eq $n })) { throw "-Only names no selected leg: $n" } }
    $legs = @($legs | Where-Object { $names -contains $_.Name })
}
if ($legs.Count -eq 0) { throw 'no leg selected' }
function Get-SafeName($leg) {
    $s = ($leg.Name -replace '[/\\:@]', '_')
    if ($leg.Kind -eq 'timing' -and $SlowArm) { $s += ".slow$SlowArm" }
    return $s
}
function Get-MarkerPath($leg) { return (Join-Path $Done ((Get-SafeName $leg) + '.done')) }
function Get-LogPath($leg, $step = '') {
    $n = Get-SafeName $leg
    if ($step) { $n += ".$step" }
    return (Join-Path $Logs ($n + '.log'))
}
function Get-RowRoot($leg) { return (Join-Path (Join-Path $ArmRoot $leg.Row) ('adapter' + $leg.Adapter)) }
function Get-ModelFor($leg) { if ($leg.Row -eq '1.5b') { return $Model15 } else { return $Model05 } }
function Get-BlobDir {
    $n = [IO.Path]::GetFileNameWithoutExtension($Model15) + "-w$Width15"
    if ($Pairs -ne 104) { $n += "-p$Pairs" }
    return (Join-Path (Join-Path $TimingR1 'blobs') $n)
}
if ($Fresh) {
    foreach ($leg in $legs) {
        Remove-Item -LiteralPath (Get-MarkerPath $leg) -ErrorAction SilentlyContinue
        if ($leg.Kind -eq 'timing') { Remove-Item -LiteralPath (Get-RowRoot $leg) -Recurse -Force -ErrorAction SilentlyContinue }
    }
}

# ---- inputs ----------------------------------------------------------------------------------------
$need05 = [bool]($legs | Where-Object { $_.Leg -eq 'box' -and $_.Row -ne '1.5b' })
$need15 = [bool]($legs | Where-Object { $_.Row -eq '1.5b' })
$needFixture = [bool]($legs | Where-Object { $_.Leg -eq 'fixture' })
if ($need05 -and -not (Test-Path -LiteralPath $Model05 -PathType Leaf)) { throw "the 0.5B artifact is not at $Model05 (-Model05)" }
if ($need15 -and -not (Test-Path -LiteralPath $Model15 -PathType Leaf)) { throw "the 1.5B artifact is not at $Model15 (-Model15)" }
foreach ($p in @($Model05, $Model15, $Scratch, $BinDir)) { if ($p -match '\s') { throw "path '$p' contains a space: move it" } }
if ($needFixture) {
    if (-not $FixtureDir) { $FixtureDir = $env:SUPERSLM_PAGED_KV_FIXTURE_DIR }
    if (-not $FixtureDir) { $FixtureDir = Join-Path $Scratch 'fixtures' }
    New-Item -ItemType Directory -Force -Path $FixtureDir | Out-Null
    $FixtureDir = (Resolve-Path -LiteralPath $FixtureDir).Path
    if (-not (Test-Path -LiteralPath (Join-Path $FixtureDir 'pkv_def.sslm'))) {
        Write-Output "generating the paged-KV fixtures into $FixtureDir"
        & python (Join-Path $Repo 'tools\gen_paged_kv_fixture.py') $FixtureDir | Out-Host
        if ($LASTEXITCODE -ne 0 -or -not (Test-Path -LiteralPath (Join-Path $FixtureDir 'pkv_def.sslm'))) { throw 'fixture generation failed' }
    }
}

# ---- build (all before any leg) ------------------------------------------------------------------------
# Built only when a G6 executable is missing, or with -Rebuild. Every leg's provenance, and the timing harness's run
# identity, hold the executables' SHA-256, and build_paged_kv_gpu.bat relinks every executable it builds, so a rebuild
# on every invocation would make every finished leg stale and refuse every timing resume. After a deliberate
# -Rebuild, correctness legs rerun and a timing leg needs -Fresh (its records belong to the old binary).
$missing = @(@('pkvg_suite.exe') + @($legs | ForEach-Object { $_.Exe }) | Select-Object -Unique |
    Where-Object { -not (Test-Path -LiteralPath (Join-Path $BinDir $_) -PathType Leaf) })
if (-not $NoBuild -and ($Rebuild -or $missing.Count -gt 0)) {
    $outDir = Split-Path -Parent $BinDir
    $buildLog = Join-Path $Logs 'build.log'
    Write-Output "BUILD tests\paged-kv-gpu\build_paged_kv_gpu.bat --out=$outDir g6 (log $buildLog)"
    & cmd /c "`"$(Join-Path $Repo 'tests\paged-kv-gpu\build_paged_kv_gpu.bat')`" `"--out=$outDir`" g6 > `"$buildLog`" 2>&1"
    if ($LASTEXITCODE -ne 0) { Get-Content -LiteralPath $buildLog | Select-Object -Last 30 | Out-Host; throw 'the G6 build failed' }
    Get-Content -LiteralPath $buildLog | Where-Object { $_ -match '^\[G6\]|PKVG BUILD|RED BY' } | Out-Host
}
$suite = Join-Path $BinDir 'pkvg_suite.exe'
foreach ($leg in $legs) {
    $e = Join-Path $BinDir $leg.Exe
    if (-not (Test-Path -LiteralPath $e -PathType Leaf)) { throw "$e is not built (build_paged_kv_gpu.bat g6)" }
}
if (($legs | Where-Object { $_.Kind -eq 'correctness' }) -and -not (Test-Path -LiteralPath $suite -PathType Leaf)) { throw "$suite is not built" }
$timingExePath = Join-Path $BinDir $TimingExe

# ---- commissioning ----------------------------------------------------------------------------------
$commissioned = $false
$verifier = 'D:\Wizard\Claude\Tooling\instrument-commission.ps1'
if (($legs | Where-Object { $_.Kind -eq 'timing' }) -and (Test-Path -LiteralPath $verifier)) {
    # Under Windows PowerShell 5.1 a native command's stderr merged with 2>&1 becomes a terminating error when
    # $ErrorActionPreference is Stop; the verifier's stderr must not stop the run, only fail to commission.
    $saved = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try { $verification = @(powershell -NoProfile -File $verifier -Verify 2>&1 | ForEach-Object { "$_" }) }
    catch { $verification = @() }
    finally { $ErrorActionPreference = $saved }
    $commissioned = [bool]($verification | Where-Object { $_ -eq "COMMISSIONED  $RegistryId" })
}

# ---- provenance and markers ------------------------------------------------------------------------------
$Head = (& git -C $Repo rev-parse HEAD)
if ($LASTEXITCODE -ne 0 -or -not $Head) { throw "git rev-parse HEAD failed in $Repo" }
$Head = "$Head".Trim()
$shaCache = @{}
function Get-CachedSha($path) {
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { return 'missing' }
    if (-not $shaCache.ContainsKey($path)) { $shaCache[$path] = Get-PkvSha256 $path }
    return $shaCache[$path]
}
function Get-LegRecord($leg) {
    $r = [ordered]@{}
    $r['Leg'] = $leg.Name
    $r['Head'] = $Head
    $r['Adapter'] = "$($leg.Adapter)"
    $r['Exe'] = Get-CachedSha (Join-Path $BinDir $leg.Exe)
    if ($leg.Kind -eq 'correctness') {
        $r['Suite'] = Get-CachedSha $suite
        if ($leg.Leg -eq 'box') { $r['Model05'] = Get-CachedSha $Model05 }
        else { $r['Fixture'] = Get-CachedSha (Join-Path $FixtureDir 'pkv_def.sslm') }
        $stem = if ($leg.Leg -eq 'box') { [IO.Path]::GetFileNameWithoutExtension($Model05) } else { 'pkv_def' }
        $r['Reference'] = Get-CachedSha (Join-Path $RefDir ('v1.11.0_' + $stem + '.ref'))
    } else {
        $r['Row'] = $leg.Row
        $r['Model'] = Get-CachedSha (Get-ModelFor $leg)
        $r['Rule'] = 'per-process: 5 processes x 104 pairs, 4 warm-up, one rerun on NO RESULT'
        $r['SlowArm'] = if ($SlowArm) { $SlowArm } else { 'none' }
        $r['Pairs'] = "$Pairs"
        if ($leg.Row -eq '1.5b') { $r['Width'] = "$Width15" }
        $r['Commissioned'] = "$commissioned"
    }
    return $r
}
function Read-LegRecord($path) {
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { return $null }
    $r = [ordered]@{}
    foreach ($line in (Get-Content -LiteralPath $path)) {
        $at = $line.IndexOf('=')
        if ($at -gt 0) { $r[$line.Substring(0, $at)] = $line.Substring($at + 1) }
    }
    return $r
}
function Write-LegRecord($path, $record) {
    $tmp = "$path.partial"
    ($record.Keys | ForEach-Object { "$_=$($record[$_])" }) | Set-Content -LiteralPath $tmp -Encoding ascii
    Move-Item -LiteralPath $tmp -Destination $path -Force
}
function Test-LegCurrent($stored, $record) {
    if (-not $stored) { return $false }
    foreach ($k in $record.Keys) { if ("$($stored[$k])" -ne "$($record[$k])") { return $false } }
    return $true
}

# ---- the quiet check (the C6 runners' offenders, and the GPU's own contenders), and the boot ---------------
function Get-QuietReport {
    $offenders = @()
    if (Test-Path -LiteralPath 'D:\_ssu_build_lock') { $offenders += 'D:\_ssu_build_lock present' }
    $procs = @(Get-Process -Name 'UnrealEditor', 'UnrealEditor-Cmd', 'cl', 'link', 'dotnet', 'UnrealBuildTool', 'ninja',
        'cmake', 'msbuild', 'cc1plus', 'smoke_adapter_driver', 'superslm_*', 'pkvg_*', 'g1d_cell_*', 'g1e_cell_*' -ErrorAction SilentlyContinue)
    foreach ($p in $procs) { $offenders += "process $($p.ProcessName) (pid $($p.Id))" }
    $load = @()
    for ($i = 0; $i -lt 3; ++$i) {
        $load += [int](Get-CimInstance Win32_Processor | Measure-Object -Property LoadPercentage -Average).Average
        Start-Sleep -Seconds 1
    }
    $avg = [int](($load | Measure-Object -Average).Average)
    if ($avg -gt $MaxLoadPercent) { $offenders += "CPU load $avg% above $MaxLoadPercent% (samples $($load -join ','))" }
    return [pscustomobject]@{ Quiet = ($offenders.Count -eq 0); Offenders = $offenders; Load = ($load -join ',') }
}
function Wait-Quiet($what) {
    $deadline = (Get-Date).AddSeconds($QuietWaitSeconds)
    $q = Get-QuietReport
    while (-not $q.Quiet -and (Get-Date) -lt $deadline) {
        Write-Host "waiting for a quiet box before ${what}: $($q.Offenders -join '; ')"
        Start-Sleep -Seconds 30
        $q = Get-QuietReport
    }
    Add-Content -LiteralPath (Join-Path $Logs 'quiet.log') -Value ("{0} {1}: load {2}; {3}" -f (Get-Date -Format s), $what, $q.Load, ($q.Offenders -join '; '))
    return $q
}
# The plan's boot identity (cell 7.9: LastBootUpTime, read before and after each process), as one token.
function Get-BootId {
    return ((Get-CimInstance Win32_OperatingSystem).LastBootUpTime.ToUniversalTime().ToString('yyyy-MM-ddTHH:mm:ssZ'))
}

# ---- run ---------------------------------------------------------------------------------------------
$results = @{}
function Invoke-Timing($leg, $step, [string[]]$argv) {
    foreach ($a in $argv) { if ($a -match '\s') { throw "argument '$a' contains a space: move the path" } }
    $log = Get-LogPath $leg $step
    Write-Host ("RUN  {0} {1}: {2} {3}" -f $leg.Name, $step, $timingExePath, ($argv -join ' '))
    $code = Invoke-PkvLogged -Exe $timingExePath -Arguments $argv -Log $log -WorkingDirectory $Repo
    $tail = @(Get-Content -LiteralPath $log -ErrorAction SilentlyContinue)
    return [pscustomobject]@{ Code = $code; Tail = $tail; Log = $log }
}
function Get-InfraLine($tail) { return (($tail | Where-Object { $_ -match '^INFRASTRUCTURE:|^RED:' } | Select-Object -Last 1) -join '') }

# One row: prepare (1.5B), then attempt 1 and, on NO RESULT, attempt 2; each attempt five processes and a grade.
# Returns the leg's outcome string, or $null when the leg is not finished (resumable: no marker is written).
function Invoke-Row($leg) {
    $model = Get-ModelFor $leg
    $rowArgs = @("--row=$($leg.Row)", "--model=$model") + $mechArgs
    if ($leg.Row -eq '1.5b') {
        $blobs = Get-BlobDir
        $rowArgs += "--blobs=$blobs"
        if ($Width15 -ne 16384) { $rowArgs += "--width=$Width15" }
        $p = Invoke-Timing $leg 'prepare' (@('prepare') + $rowArgs)
        if ($p.Code -eq 1) { return "RED (prepare) $(Get-InfraLine $p.Tail)" }
        if ($p.Code -ne 0) { $results[$leg.Name] = "NOT DONE: prepare exit $($p.Code) $(Get-InfraLine $p.Tail)"; return $null }
    }
    $attemptDir = $null
    $grade = $null
    foreach ($attempt in 1, 2) {
        $attemptDir = Join-Path (Get-RowRoot $leg) "attempt$attempt"
        # A row already complete (graded, or all five processes on file) is graded again, never re-entered: a process
        # invocation on another boot would discard its records.
        $grade = $null
        if (Test-Path -LiteralPath (Join-Path $attemptDir 'run.txt')) {
            $g = Invoke-Timing $leg "a$attempt.grade" @('grade', "--out=$attemptDir")
            if ($g.Code -ne 40) { $grade = $g }
        }
        if (-not $grade) {
            for ($k = 1; $k -le 5; ++$k) {
                $tries = 0
                while ($true) {
                    $tries += 1
                    if ($tries -gt 3) { $results[$leg.Name] = "NOT DONE: attempt $attempt process $k failed 3 times"; return $null }
                    $q = Wait-Quiet "$($leg.Name) attempt $attempt process $k"
                    if (-not $q.Quiet) { $results[$leg.Name] = "NOT-QUIET (attempt $attempt process $k not run): $($q.Offenders -join '; ')"; return $null }
                    $bootBefore = Get-BootId
                    $argv = @('process') + $rowArgs + @("--out=$attemptDir", "--process=$k", "--boot=$bootBefore")
                    if ($commissioned) { $argv += '--commissioned' }
                    $r = Invoke-Timing $leg "a$attempt.p$k" $argv
                    $bootAfter = Get-BootId
                    if ($bootAfter -ne $bootBefore) {
                        Remove-Item -LiteralPath (Join-Path $attemptDir "process_$k.log") -Force -ErrorAction SilentlyContinue
                        Write-Host "DISCARDED $($leg.Name) attempt $attempt process ${k}: the boot changed ($bootBefore -> $bootAfter); rerun"
                        continue
                    }
                    if ($r.Code -eq 0) { break }
                    if ($r.Code -eq 1) { return "RED attempt $attempt process ${k}: $(Get-InfraLine $r.Tail)" }
                    if ($r.Code -eq 40) { $results[$leg.Name] = "NOT DONE: attempt $attempt process ${k}: $(Get-InfraLine $r.Tail)"; return $null }
                    Write-Host "CRASH $($leg.Name) attempt $attempt process ${k}: exit $($r.Code); rerun whole"
                }
            }
            $grade = Invoke-Timing $leg "a$attempt.grade" @('grade', "--out=$attemptDir")
            if ($grade.Code -eq 40) { $results[$leg.Name] = "NOT DONE: grade: $(Get-InfraLine $grade.Tail)"; return $null }
        }
        $line = $grade.Tail | Where-Object { $_ -match '^G6 R1 row \S+: process_medians=' } | Select-Object -Last 1
        $rule = if ($line -match ' rule=(PASS|FAIL|NO RESULT) ') { $Matches[1] } else { '' }
        $status = if ($line -match ' status=(\S+)') { $Matches[1] } else { '' }
        if (-not $rule -or -not $status) { $results[$leg.Name] = "NOT DONE: the grade printed no row line ($($grade.Log))"; return $null }
        if ($rule -eq 'NO RESULT' -and $attempt -eq 1) {
            Write-Host "NO RESULT $($leg.Name) attempt 1: one rerun of five fresh processes (attempt 2), its verdict its own"
            continue
        }
        break
    }
    $tag = switch ($status) {
        'STANDING' { 'ROW-' + ($rule -replace ' ', '-') }
        'MUTANT_CONSTRUCTION' { "MUTANT-$SlowArm-" + ($rule -replace ' ', '-') }
        default { "$status-READING(" + ($rule -replace ' ', '-') + ')' }
    }
    $note = if ($rule -eq 'NO RESULT' -and $attempt -eq 2) { ' second NO RESULT: to the planner with the load records' } else { '' }
    return "$tag attempt=$attempt $($line -replace '^G6 R1 ', '')$note"
}

function Invoke-Leg($leg) {
    $marker = Get-MarkerPath $leg
    $record = Get-LegRecord $leg
    $stored = Read-LegRecord $marker
    if (Test-LegCurrent $stored $record) {
        Write-Output "SKIP $($leg.Name): done earlier with this run's provenance ($($stored['Outcome']))"
        return
    }
    if ($stored) { Write-Output "STALE $($leg.Name): its marker's provenance differs; rerunning" }
    Remove-Item -Path 'Env:\SSLM_GPU_ENABLE_DEBUG_LAYER' -ErrorAction SilentlyContinue
    $env:SSLM_GPU_ADAPTER_INDEX = "$($leg.Adapter)"
    $t0 = Get-Date
    if ($leg.Kind -eq 'correctness') {
        $log = Get-LogPath $leg
        $argv = @("--bin=$BinDir", '--step=G6', "--pkv-reference=$RefDir")
        if ($leg.Leg -eq 'box') { $argv += "--model05=$Model05" } else { $argv += "--pkv-fixtures=$FixtureDir" }
        $argv += "=$($leg.Job)"
        foreach ($a in $argv) { if ($a -match '\s') { throw "argument '$a' contains a space: move the path" } }
        Write-Output ("RUN  {0}: {1} {2} (log {3})" -f $leg.Name, $suite, ($argv -join ' '), $log)
        $code = Invoke-PkvLogged -Exe $suite -Arguments $argv -Log $log -WorkingDirectory $Repo
        $tail = @(Get-Content -LiteralPath $log -ErrorAction SilentlyContinue)
        $last = $tail | Where-Object { $_ -match '^pkvg_suite at G6:' } | Select-Object -Last 1
        $verdict = switch ($code) { 0 { 'GREEN' } 1 { 'RED' } default { "CRASH($code)" } }
        if ($verdict -like 'CRASH*') {
            $results[$leg.Name] = "$verdict ($([int]((Get-Date) - $t0).TotalSeconds) s)"
            Write-Output "NOT DONE $($leg.Name): $($results[$leg.Name])"
            return
        }
        $outcome = "$verdict exit=$code $last"
    } else {
        if ($SlowArm) { $env:SUPERSLM_PKVG_G6_SLOW_ARM = $SlowArm } else { Remove-Item -Path 'Env:\SUPERSLM_PKVG_G6_SLOW_ARM' -ErrorAction SilentlyContinue }
        try { $outcome = Invoke-Row $leg }
        finally { Remove-Item -Path 'Env:\SUPERSLM_PKVG_G6_SLOW_ARM' -ErrorAction SilentlyContinue }
        if (-not $outcome) { Write-Output "NOT DONE $($leg.Name): $($results[$leg.Name])"; return }
    }
    $record['Outcome'] = ('{0} ({1} s) {2}' -f $outcome, [int]((Get-Date) - $t0).TotalSeconds, (Get-Date -Format s))
    Write-LegRecord $marker $record
    $results.Remove($leg.Name)
    Write-Output "DONE $($leg.Name): $($record['Outcome'])"
}

$correctness = @($legs | Where-Object { $_.Kind -eq 'correctness' })
$timing = @($legs | Where-Object { $_.Kind -eq 'timing' })
$timingAll = $timing
foreach ($leg in $correctness) { Invoke-Leg $leg }

# Timing legs already done with this run's provenance are reported without pausing anyone.
$pending = @($timing | Where-Object { -not (Test-LegCurrent (Read-LegRecord (Get-MarkerPath $_)) (Get-LegRecord $_)) })
foreach ($leg in $timing) { if ($pending -notcontains $leg) { Invoke-Leg $leg } }
$timing = $pending
if ($timing.Count -gt 0) {
    # R1 is read on the RTX 2080 SUPER: each timing adapter's DXGI description is checked before anything runs.
    $wrong = @()
    foreach ($a in @($timing | ForEach-Object { $_.Adapter } | Select-Object -Unique)) {
        $env:SSLM_GPU_ADAPTER_INDEX = "$a"
        $desc = "$(& $timingExePath adapter)".Trim()
        Write-Output "adapter ${a}: $desc"
        if ($ExpectAdapter -and $desc -notlike "*$ExpectAdapter*") { $wrong += "adapter $a is '$desc', not '$ExpectAdapter'" }
    }
    Remove-Item -Path 'Env:\SSLM_GPU_ADAPTER_INDEX' -ErrorAction SilentlyContinue
    if ($wrong.Count -gt 0) { throw ("timing refused: " + ($wrong -join '; ') + " (-Adapters, or -ExpectAdapter '' to time another card)") }
    $pauseDir = Split-Path -Parent $PauseFile
    $pauseCreated = $false
    try {
        if (Test-Path -LiteralPath $PauseFile) {
            Write-Output "PAUSE marker $PauseFile was already present: left as found"
        } else {
            New-Item -ItemType Directory -Force -Path $pauseDir | Out-Null
            Set-Content -LiteralPath $PauseFile -Value ("G6 GPU timing, pid $PID, started " + (Get-Date -Format s)) -Encoding ascii
            $pauseCreated = $true
            Write-Output "PAUSE marker created: $PauseFile"
        }
        foreach ($leg in $timing) { Invoke-Leg $leg }
    } finally {
        if ($pauseCreated) {
            Remove-Item -LiteralPath $PauseFile -Force -ErrorAction SilentlyContinue
            Write-Output "PAUSE marker removed: $PauseFile"
        }
    }
}
Remove-Item -Path 'Env:\SSLM_GPU_ADAPTER_INDEX' -ErrorAction SilentlyContinue

# ---- R1 over both rows, per adapter ---------------------------------------------------------------------
# The decisive attempt of a row is attempt 2 when it was graded (a NO RESULT's rerun), else attempt 1.
$r1Lines = @()
foreach ($a in @($timingAll | ForEach-Object { $_.Adapter } | Select-Object -Unique)) {
    $dirs = @()
    foreach ($r in @('0.5b', '1.5b')) {
        $root = Join-Path (Join-Path $ArmRoot $r) "adapter$a"
        foreach ($attempt in 2, 1) {
            $d = Join-Path $root "attempt$attempt"
            if (Test-Path -LiteralPath (Join-Path $d 'result.txt')) { $dirs += $d; break }
        }
    }
    if ($dirs.Count -eq 0) { continue }
    $res = Join-Path $ArmRoot "r1_adapter$a.txt"
    $saved = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try { $out = @(& $timingExePath verdict "--rows=$($dirs -join ',')" "--result=$res" 2>&1 | ForEach-Object { "$_" }) }
    finally { $ErrorActionPreference = $saved }
    $r1Lines += @($out | Where-Object { $_ -match '^G6 R1' } | ForEach-Object { "  [adapter $a] $_" })
}

# ---- summary ---------------------------------------------------------------------------------------------
$rows = @()
$allDone = $true
foreach ($leg in $legs) {
    if ($results.ContainsKey($leg.Name)) { $state = $results[$leg.Name] }
    else {
        $stored = Read-LegRecord (Get-MarkerPath $leg)
        $state = if ($stored) { "$($stored['Outcome'])" } else { 'NOT-RUN' }
    }
    $verdict = ($state -split ' ')[0]
    # Done: a green cell, a standing row PASS, a mutant's reading, or a complete quarantined reading (recorded; it
    # decides nothing).
    if ($verdict -ne 'GREEN' -and $verdict -ne 'ROW-PASS' -and $verdict -notlike 'MUTANT-*' -and $verdict -notlike '*-READING*') { $allDone = $false }
    $rows += [pscustomobject]@{ Leg = $leg.Name; Kind = $leg.Kind; Verdict = $verdict; Detail = $state }
}
$summary = @('G6 box run summary ' + (Get-Date -Format s) + " (repo $Repo at $Head, bin $BinDir, scratch $Scratch)")
$summary += ("timing instrument {0}: {1}; arm {2}" -f $RegistryId, $(if ($commissioned) { 'COMMISSIONED (readings stand)' } else { 'not commissioned: readings QUARANTINED' }), $Arm)
$summary += ($rows | Format-Table -AutoSize -Wrap Leg, Kind, Verdict, Detail | Out-String -Width 240).TrimEnd()
$summary += ''
$summary += 'R1 rows (per process and per row, from each attempt''s result.txt):'
foreach ($leg in $timingAll) {
    foreach ($attempt in 1, 2) {
        $res = Join-Path (Join-Path (Get-RowRoot $leg) "attempt$attempt") 'result.txt'
        if (Test-Path -LiteralPath $res) {
            Get-Content -LiteralPath $res | Where-Object { $_ -match '^G6 R1 |^adapter=|^commissioned=|^model=' } |
                ForEach-Object { $summary += "  [$($leg.Name) attempt $attempt] $_" }
        }
    }
}
if ($r1Lines.Count -gt 0) { $summary += ''; $summary += 'R1 over both rows:'; $summary += $r1Lines }
$summary | Set-Content -LiteralPath (Join-Path $Scratch 'summary.txt') -Encoding utf8
$summary | Write-Output
# Exit 1 while a leg is unfinished, red, or a standing row FAIL or NO RESULT; 0 when every selected leg is done (a mutant
# arm's rows included: their verdicts are the harness's `grade` and `verdict` exits, and the summary's lines).
if (-not $allDone) { exit 1 }
exit 0
