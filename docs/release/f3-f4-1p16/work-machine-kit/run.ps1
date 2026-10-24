# F3 certification on the RX 7900 XTX (work machine). Runs from the drive; writes results\<computer>\ beside this file.
# Windows PowerShell 5.1. Resumable: a step with a .done marker is skipped (-Force reruns everything).
# Each GPU step is bounded at 15 minutes and starts at least 5 minutes after the previous one ended (rest.txt).
# -Vendor is for testing the kit on another card; -CheckOnly lists the adapters and stops.
param([switch]$Force, [switch]$CheckOnly, [string]$Vendor = '1002')
$ErrorActionPreference = 'Stop'
$Kit = Split-Path -Parent $MyInvocation.MyCommand.Path
$Bin = Join-Path $Kit 'bin'
$Res = Join-Path $Kit ('results\' + $env:COMPUTERNAME)
$Blobs46 = Join-Path $Res 'blobs46'
$R10x = Join-Path $Res 'r10x'
foreach ($d in @($Res, $Blobs46, $R10x)) { New-Item -ItemType Directory -Force $d | Out-Null }
$KitLog = Join-Path $Res 'kit.log'
$StepMinutes = 15
$RestMinutes = 5
function Say([string]$t) { $l = '[' + (Get-Date -Format 'HH:mm:ss') + '] ' + $t; Write-Host $l; Add-Content -LiteralPath $KitLog -Value $l }
function Done([string]$n) { (-not $Force) -and (Test-Path (Join-Path $Res "$n.done")) }
function Mark([string]$n, [string]$v) { Set-Content -LiteralPath (Join-Path $Res "$n.done") -Value "$v $(Get-Date -Format s)" }

# The rest between GPU steps: rest.txt holds when the last one ended (it survives a crash or a rerun).
function Rest {
    $f = Join-Path $Res 'rest.txt'
    if (-not (Test-Path $f)) { return }
    $wait = ([datetime]::Parse((Get-Content -LiteralPath $f | Select-Object -First 1))).AddMinutes($RestMinutes) - (Get-Date)
    if ($wait.TotalSeconds -gt 0) {
        Say ('resting {0:N0} s before the next step (5 minutes between GPU steps)' -f $wait.TotalSeconds)
        Start-Sleep -Seconds ([int][math]::Ceiling($wait.TotalSeconds))
    }
}
function RestStamp { Set-Content -LiteralPath (Join-Path $Res 'rest.txt') -Value (Get-Date -Format 'o') }

# Runs one executable from bin\ with a time bound: the smaller of $Minutes and what is left of the step's 15 minutes.
# Its output goes to results\<step>.log, then "exit N" (124: killed at the bound; 125: no time left to start it).
$script:Deadline = $null
function Cell([string]$Step, [string]$Exe, [string[]]$Arguments, [int]$Minutes) {
    $log = Join-Path $Res "$Step.log"
    $ms = [int64]$Minutes * 60000
    $left = [int64](($script:Deadline - (Get-Date)).TotalMilliseconds)
    if ($left -lt $ms) { $ms = $left }
    if ($ms -le 0) { Add-Content -LiteralPath $log -Value "SKIPPED $Exe : the step's 15 minutes have run out"; return 125 }
    $o = "$log.out"; $e = "$log.err"
    $quoted = @($Arguments | ForEach-Object { '"' + $_ + '"' })
    Add-Content -LiteralPath $log -Value ('[' + (Get-Date -Format s) + '] ' + $Exe + ' ' + ($Arguments -join ' '))
    $sp = @{ FilePath = (Join-Path $Bin $Exe); WorkingDirectory = $Bin; NoNewWindow = $true; PassThru = $true
             RedirectStandardOutput = $o; RedirectStandardError = $e }
    if ($quoted.Count -gt 0) { $sp.ArgumentList = $quoted }  # 5.1 refuses an empty -ArgumentList
    $p = Start-Process @sp
    $null = $p.Handle
    $code = 0
    if (-not $p.WaitForExit([int]$ms)) {
        try { $p.Kill() } catch {}
        Add-Content -LiteralPath $log -Value ('TIMEOUT after {0:N1} min: killed' -f ($ms / 60000))
        $code = 124
    } else { $code = $p.ExitCode }
    foreach ($f in @($o, $e)) { if (Test-Path $f) { Get-Content -LiteralPath $f | Add-Content -LiteralPath $log; Remove-Item -LiteralPath $f } }
    Add-Content -LiteralPath $log -Value ('[' + (Get-Date -Format s) + '] exit ' + $code)
    return $code
}

Say "F3 certification kit (SuperSLM branch claude/f3-f4-cells-1p16); drive folder $Kit"
# --- the RX 7900 XTX: vendor 1002 with at least 4 GB (the Ryzen's integrated Radeon is vendor 1002 too, ~0.5 GB) ---
$adapters = @(& (Join-Path $Bin 'adapters.exe'))
Say 'D3D12 hardware adapters (EnumAdapters1 order):'
foreach ($l in $adapters) { Say "    $l" }
$xtx = $adapters | Where-Object { $_ -match ('^(\d+) ' + $Vendor + ' ([0-9a-f]{4}) (\d+) ') -and [int64]$Matches[3] -ge 4096 } | Select-Object -First 1
if (-not $xtx) { Say "NO ADAPTER of vendor $Vendor with at least 4 GB: this kit needs the RX 7900 XTX. Nothing was run."; exit 2 }
$null = $xtx -match ('^(\d+) ' + $Vendor + ' ([0-9a-f]{4}) ')
$Index = $Matches[1]; $Device = $Matches[2]
$Card = "$($Vendor)_$Device"
$env:SSLM_GPU_ADAPTER_INDEX = $Index
Say "using adapter $Index (vendor $Vendor, device $Device)"
# No debug layer for any step: the F2 cells passed under it on this card on 2026-10-24, and GPU-based validation would
# put the real-model steps far past their 15 minutes (1.9 took 90 min under it on the 2080 SUPER).
$env:SSLM_GPU_ENABLE_DEBUG_LAYER = $null
if ($CheckOnly) { Say 'check only: stopping here'; exit 0 }

$FX = Join-Path $Kit 'fixtures'
$P = Join-Path $Kit 'prompts'
$M05 = Join-Path $Kit 'models\qwen2.5-0.5b-instruct-dg.sslm'
$M15 = Join-Path $Kit 'models\qwen2.5-1.5b-instruct-dg-g5.sslm'
$Theirs = Join-Path $Kit 'blobs'   # the RTX 2080 SUPER's 4.6 saves

# The steps, in order: a name and what it runs. RunStep returns the step's failures (none: PASS).
$Steps = @(
    @{ N = 'f2set' },
    @{ N = 'real15'; Tag = '1p5'; M = $M15; Ids = 'world.1p5.ids' },
    @{ N = 'real05'; Tag = '0p5'; M = $M05; Ids = 'world.ids' },
    @{ N = 'r10x' },
    @{ N = 't-acc-15'; Role = 'accept'; Exe = 'pcg_cell_10_4_timing.exe' },
    @{ N = 't-rej-15'; Role = 'reject'; Exe = 'pcg_cell_10_4_timing_restore_prefills.exe' },
    @{ N = 't-grade-15'; Role = 'grade'; Exe = 'pcg_cell_10_4_timing.exe' })
function RunStep($s) {
    $bad = @()
    switch -Wildcard ($s.N) {
        'f2set' {
            foreach ($c in 'pcg_cell_format', 'pcg_cell_pool', 'pcg_cell_busy', 'pcg_cell_faults', 'pcg_cell_budget_snapshot', 'pcg_cell_restore_derived') {
                $x = Cell 'f2set' "$c.exe" @("--fixture-dir=$FX") 5; if ($x -ne 0) { $bad += "$c exit $x" }
            }
            $env:SUPERSLM_PRECACHE_E0_ARTIFACT_DIR = Join-Path $Kit 'e0'
            $x = Cell 'f2set' 'pcg_cell_schema.exe' @() 3; if ($x -ne 0) { $bad += "pcg_cell_schema exit $x" }
            $x = Cell 'f2set' 'pcg_cell_9_10.exe' @("--fixture-dir=$FX", ('--captures=' + (Join-Path $Kit "captures\v1.13.1_$Card"))) 5
            if ($x -ne 0) { $bad += "pcg_cell_9_10 exit $x" }
            $x = Cell 'f2set' 'pcg_cell_6_5.exe' @("--fixture-dir=$FX") 5; if ($x -ne 0) { $bad += "pcg_cell_6_5 pkv_def exit $x" }
        }
        'real*' {
            $ids = Join-Path $P $s.Ids
            $other = Join-Path $Theirs "gpu-$($s.Tag)-10de_1e81.spx1"
            # 4.6 (a) here, (b) against the 2080 SUPER's blob, (c) the 2080 SUPER's blob restored here through the seam.
            $x = Cell $s.N 'pcg_cell_4_6.exe' @("--model=$($s.M)", "--tag=$($s.Tag)", "--ids=$ids", "--out=$Blobs46", "--other=$other", "--foreign=$other") 10
            if ($x -ne 0) { $bad += "pcg_cell_4_6 $($s.Tag) exit $x (a 4.6 difference is a finding, not a broken kit)" }
            $x = Cell $s.N 'pcg_cell_6_5.exe' @("--model=$($s.M)", "--tag=$($s.Tag)", "--world=$ids", ('--persona=' + (Join-Path $P 'persona00.ids'))) 10
            if ($x -ne 0) { $bad += "pcg_cell_6_5 $($s.Tag) exit $x" }
        }
        'r10x' {
            foreach ($cell in '10.1', '10.2') {
                foreach ($phase in 'capture', 'restore') {
                    $x = Cell 'r10x' 'pcg_cell_10_4_real.exe' @("--cell=$cell", "--phase=$phase", "--model=$M05", "--prompts=$P", "--blobs=$R10x") 8
                    if ($x -ne 0) { $bad += "pcg_cell_10_4_real $cell $phase exit $x"; break }
                }
            }
        }
        't-*' {
            # The instrument decides nothing (grade_f3.py does, on the dev box): exit 0 means the run completed and
            # every reading resolved.
            $x = Cell $s.N $s.Exe @("--model=$M15", '--tag=1p5', ('--ids=' + (Join-Path $P 'world.1p5.ids')), "--role=$($s.Role)") 15
            if ($x -ne 0) { $bad += "pcg_cell_10_4_timing $($s.Role) exit $x" }
        }
    }
    $bad
}

foreach ($s in $Steps) {
    $name = $s.N
    if (Done $name) { Say "$name : already done ($(Get-Content -LiteralPath (Join-Path $Res "$name.done")))"; continue }
    Rest
    Say "$name : starting (at most $StepMinutes min)"
    $stepLog = Join-Path $Res "$name.log"
    if (Test-Path $stepLog) { Remove-Item -LiteralPath $stepLog }   # a rerun's log holds only the rerun
    $script:Deadline = (Get-Date).AddMinutes($StepMinutes)
    $t0 = Get-Date
    $bad = @()
    try { $bad = @(RunStep $s) } catch { $bad = @("ERROR $($_.Exception.Message)") }
    RestStamp
    $mins = ((Get-Date) - $t0).TotalMinutes
    if ($mins -gt $StepMinutes) { $bad += ('over the 15-minute bound ({0:N1} min)' -f $mins) }
    $v = $(if ($bad.Count -eq 0) { 'PASS' } else { 'FAIL ' + ($bad -join '; ') })
    Mark $name $v
    Say ('{0} : {1} ({2:N1} min)' -f $name, $v, $mins)
}

$Summary = @()
foreach ($n in @($Steps | ForEach-Object { $_.N })) {
    $d = Join-Path $Res "$n.done"
    $l = Join-Path $Res "$n.log"
    $suites = $(if (Test-Path $l) { (Get-Content -LiteralPath $l | Where-Object { $_ -match 'checks=\d+|^10\.4 reading' }) -join ' | ' } else { '' })
    $Summary += "$n : $(if (Test-Path $d) { Get-Content -LiteralPath $d } else { 'not run' }) :: $suites"
}
$Summary += "adapter $Index ($Vendor`:$Device); debug layer off"
Set-Content -LiteralPath (Join-Path $Res 'summary.txt') -Value $Summary
foreach ($l in $Summary) { Say $l }
Say 'KIT RUN FINISHED: unplug the drive and bring it back to the dev box'
