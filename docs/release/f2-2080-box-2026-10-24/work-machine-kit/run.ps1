# F2 GPU twins on the RX 7900 XTX (work machine). Runs from the drive; writes results\<computer>\ beside this file.
# Windows PowerShell 5.1. Resumable: a step with a .done marker is skipped (-Force reruns everything).
# -Vendor and -Smoke are for testing the kit on another card: -Smoke runs one cell of g5 and of g4 and skips g19.
param([switch]$Force, [switch]$CheckOnly, [string]$Vendor = '1002', [switch]$Smoke)
$ErrorActionPreference = 'Stop'
$Kit = Split-Path -Parent $MyInvocation.MyCommand.Path
$Bin = Join-Path $Kit 'bin'
$Res = Join-Path $Kit ('results\' + $env:COMPUTERNAME)
New-Item -ItemType Directory -Force $Res | Out-Null
$KitLog = Join-Path $Res 'kit.log'
function Say([string]$t) { $l = '[' + (Get-Date -Format 'HH:mm:ss') + '] ' + $t; Write-Host $l; Add-Content -LiteralPath $KitLog -Value $l }
function Done([string]$n) { (-not $Force) -and (Test-Path (Join-Path $Res "$n.done")) }
function Mark([string]$n, [string]$v) { Set-Content -LiteralPath (Join-Path $Res "$n.done") -Value "$v $(Get-Date -Format s)" }

# Runs one executable from bin\ with a time bound; its output goes to results\<step>.log, then "exit N".
function Cell([string]$Step, [string]$Exe, [string[]]$Arguments, [int]$Minutes) {
    $log = Join-Path $Res "$Step.log"
    $o = "$log.out"; $e = "$log.err"
    $quoted = @($Arguments | ForEach-Object { '"' + $_ + '"' })
    Add-Content -LiteralPath $log -Value ('[' + (Get-Date -Format s) + '] ' + $Exe + ' ' + ($Arguments -join ' '))
    $sp = @{ FilePath = (Join-Path $Bin $Exe); WorkingDirectory = $Bin; NoNewWindow = $true; PassThru = $true
             RedirectStandardOutput = $o; RedirectStandardError = $e }
    if ($quoted.Count -gt 0) { $sp.ArgumentList = $quoted }  # 5.1 refuses an empty -ArgumentList
    $p = Start-Process @sp
    $null = $p.Handle
    $code = 0
    if (-not $p.WaitForExit($Minutes * 60000)) {
        try { $p.Kill() } catch {}
        Add-Content -LiteralPath $log -Value "TIMEOUT after $Minutes min: killed"
        $code = 124
    } else { $code = $p.ExitCode }
    foreach ($f in @($o, $e)) { if (Test-Path $f) { Get-Content -LiteralPath $f | Add-Content -LiteralPath $log; Remove-Item -LiteralPath $f } }
    Add-Content -LiteralPath $log -Value ('[' + (Get-Date -Format s) + '] exit ' + $code)
    return $code
}

Say "F2 GPU twins kit (SuperSLM main 5599f7e4); drive folder $Kit"
# --- the RX 7900 XTX: vendor 1002 with at least 4 GB (the Ryzen's integrated GPU is vendor 1002 too) ---
$adapters = @(& (Join-Path $Bin 'adapters.exe'))
Say 'D3D12 hardware adapters (EnumAdapters1 order):'
foreach ($l in $adapters) { Say "    $l" }
$xtx = $adapters | Where-Object { $_ -match ('^(\d+) ' + $Vendor + ' ([0-9a-f]{4}) (\d+) ') -and [int64]$Matches[3] -ge 4096 } | Select-Object -First 1
if (-not $xtx) { Say 'NO AMD ADAPTER with at least 4 GB: this kit needs the RX 7900 XTX. Nothing was run.'; exit 2 }
$null = $xtx -match ('^(\d+) ' + $Vendor + ' ([0-9a-f]{4}) ')
$Index = $Matches[1]; $Device = $Matches[2]
$env:SSLM_GPU_ADAPTER_INDEX = $Index
Say "using adapter $Index (vendor 1002, device $Device)"
$Layer = Test-Path (Join-Path $env:windir 'System32\d3d12SDKLayers.dll')
if ($Layer) { Say 'D3D12 debug layer: present (Graphics Tools); the cells run with it, as on the 2080 SUPER' }
else { Say 'D3D12 debug layer: NOT installed (Settings > Optional features > Graphics Tools); the cells run without it, and the summary says so' }
if ($CheckOnly) { Say 'check only: stopping here'; exit 0 }

$FX = Join-Path $Kit 'fixtures'
$Summary = @()
function DebugEnv([bool]$on) { if ($on -and $Layer) { $env:SSLM_GPU_ENABLE_DEBUG_LAYER = '1' } else { $env:SSLM_GPU_ENABLE_DEBUG_LAYER = $null } }
function Verdict([string]$Step) { $l = Join-Path $Res "$Step.log"; if (-not (Test-Path $l)) { return "" }; (Get-Content -LiteralPath $l | Where-Object { $_ -match 'checks=\d+|cells, \d+ red' }) -join '; ' }

# 1. the F2 cells (debug layer on)
if (-not (Done 'f2')) {
    DebugEnv $true
    $bad = 0
    foreach ($c in 'pcg_cell_format', 'pcg_cell_pool', 'pcg_cell_busy', 'pcg_cell_faults', 'pcg_cell_budget_snapshot', 'pcg_cell_restore_derived') {
        Say "f2: $c"
        if ((Cell 'f2' "$c.exe" @("--fixture-dir=$FX") 20) -ne 0) { $bad++ }
    }
    Mark 'f2' ($(if ($bad -eq 0) { 'PASS' } else { "FAIL $bad cell(s)" }))
}
# 2. pcg_cell_schema with the E0 artifacts (debug layer on)
if (-not (Done 'schema')) {
    DebugEnv $true
    $env:SUPERSLM_PRECACHE_E0_ARTIFACT_DIR = Join-Path $Kit 'e0'
    Say 'schema: pcg_cell_schema'
    $c = Cell 'schema' 'pcg_cell_schema.exe' @() 10
    Mark 'schema' ($(if ($c -eq 0) { 'PASS' } else { "FAIL exit $c" }))
}
# 3. the cross-card leg: the 2080 SUPER's blob is refused here, and this card's blob is saved for the 2080 SUPER
if (-not (Done 'cross')) {
    DebugEnv $true
    $own = Join-Path $Res "blob-$($Vendor)_$Device.spx1"
    Say 'cross: restore the RTX 2080 SUPER blob (must be refused SSLM_RESTORE_PRODUCER_MISMATCH), save this card''s'
    $c = Cell 'cross' 'pcg_cross_card.exe' @("--fixture-dir=$FX", ('--restore=' + (Join-Path $Kit 'blobs\blob-10de_1e81.spx1')), "--save=$own") 10
    Mark 'cross' ($(if ($c -eq 0) { 'PASS' } else { "FAIL exit $c" }))
}
# 4-6. pkvg_g5 and pkvg_g4 at their own steps (debug layer on), then 1.9 alone (debug layer off: about 90 min under
# GPU-based validation on the 2080 SUPER, and both of its box crashes came under that load)
$common = @("--pkv-fixtures=$FX", ('--pkv-reference=' + (Join-Path $Kit 'paged-kv-reference')),
    ('--model05=' + (Join-Path $Kit 'models\qwen2.5-0.5b-instruct-dg.sslm')), ('--model15=' + (Join-Path $Kit 'models\qwen2.5-1.5b-instruct-dg-g5.sslm')),
    ('--adapter=' + (Join-Path $Kit 'adapters\pkv_def_zero_adapter.sslm')), ('--adapter-other-base=' + (Join-Path $Kit 'adapters\pkv_qk_zero_adapter.sslm')))
foreach ($s in @(@{ Step = 'g5'; Exe = 'pkvg_g5.exe'; Tag = 'G5'; Debug = $true }, @{ Step = 'g4'; Exe = 'pkvg_g4.exe'; Tag = 'G4'; Debug = $true },
                 @{ Step = 'g19'; Exe = 'pkvg_g4.exe'; Tag = 'G4'; Debug = $false })) {
    if (Done $s.Step) { continue }
    DebugEnv $s.Debug
    Push-Location $Bin
    $ids = @(& (Join-Path $Bin $s.Exe) --list 2>&1 | ForEach-Object { "$_" } | Where-Object { $_ -match ('^\[' + $s.Tag + ' ') } |
        ForEach-Object { '=' + (($_ -replace '^\[[^\]]+\]\s+', '').Trim()) })
    Pop-Location
    if ($s.Step -eq 'g4') { $ids = @($ids | Where-Object { $_ -ne '=1.9' }) }
    if ($s.Step -eq 'g19') { $ids = @('=1.9') }
    if ($Smoke) { if ($s.Step -eq 'g19') { continue }; $ids = @($ids | Select-Object -First 1) }
    Say "$($s.Step): $($s.Exe), $($ids.Count) [$($s.Tag)] cell(s)"
    $c = Cell $s.Step $s.Exe ($common + $ids) 30
    Mark $s.Step ($(if ($c -eq 0) { 'PASS' } else { "FAIL exit $c" }))
}

foreach ($n in 'f2', 'schema', 'cross', 'g5', 'g4', 'g19') {
    $d = Join-Path $Res "$n.done"
    $Summary += "$n : $(if (Test-Path $d) { Get-Content -LiteralPath $d } else { 'not run' }) :: $(Verdict $n)"
}
$Summary += "adapter $Index ($($Vendor):$Device); debug layer $(if ($Layer) { 'on' } else { 'NOT INSTALLED: off' })"
Set-Content -LiteralPath (Join-Path $Res 'summary.txt') -Value $Summary
foreach ($l in $Summary) { Say $l }
Say 'KIT RUN FINISHED: unplug the drive and bring it back to the dev box'
