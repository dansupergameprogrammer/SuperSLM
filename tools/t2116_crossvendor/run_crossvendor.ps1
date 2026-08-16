# T-2116 -- SuperSLM 1.0 GPU core cross-vendor certification.
#
# ASCII-only (no em-dashes, no smart quotes, no unicode anywhere in this file). A prior
# package (G-1, Claude/Laplace/gpu-determinism) shipped a script with an em-dash that
# Windows PowerShell 5.1 mis-decoded, producing a false FAIL (D-SLM2858). This script is
# plain 7-bit ASCII throughout and is run-tested under BOTH powershell.exe 5.1 and pwsh 7
# before it ships.
#
# WHAT THIS DOES. Runs the same battery of determinism, slice-invariance, and throughput
# tools this project already used to certify the GPU core on its own reference card (an
# NVIDIA RTX 2080 SUPER), against every D3D12 hardware adapter present on THIS machine,
# and compares the determinism-relevant output against the pins recorded on the reference
# card (pins\pins.txt). Throughput (tok/s) is measured and reported, labeled by adapter
# and cell, but is never gated pass/fail -- it is expected to differ across vendors.
#
# NO BUILD STEP. Every .exe under bin\ is prebuilt. This script only runs them.
#
# USAGE.
#   powershell -File run_crossvendor.ps1
#     Enumerates every D3D12 hardware adapter on this machine (skips software/WARP
#     adapters), runs the full battery on each USABLE one, and requires ALL of them to
#     pass for a zero exit code. A hardware adapter that enumerates but FAILS device
#     creation (USABLE=0) is a certification FAILURE, not a silent drop -- see
#     -AcknowledgeUnusableAdapters below if that is expected (e.g. a known-dead iGPU).
#
#   powershell -File run_crossvendor.ps1 -OnlyAdapterNameLike "*Radeon*"
#     Restricts the run to adapters whose DXGI description matches the given wildcard
#     pattern (case-insensitive). Use this if the machine also enumerates an integrated
#     GPU you do not want to certify -- e.g. a machine with a 7900 XTX plus an iGPU: only
#     the discrete card needs to pass for this ticket's own certification claim. If the
#     excluded adapter is USABLE=0, it is simply not matched by the filter and never
#     enters either list -- no acknowledgement flag needed. This flag is the correct
#     tool for "certify only this card"; it is NOT a way to make a USABLE=0 match go
#     away -- an unusable adapter that DOES match this filter still fails loudly unless
#     -AcknowledgeUnusableAdapters is also given.
#
#   powershell -File run_crossvendor.ps1 -OnlyAdapterIndex 1
#     Restricts the run to one adapter by its raw DXGI enumeration index (see the
#     "adapter enumeration" section this script prints at the top of every run).
#
#   powershell -File run_crossvendor.ps1 -AcknowledgeUnusableAdapters
#     Downgrades a USABLE=0 hardware adapter (that survived the -Only* filters, or all
#     of them if no filter is given) from a certification FAILURE to an explicit,
#     printed SKIP. Use this only when you already know a specific card on this
#     machine is not being certified this trip. The skip is still named in the console
#     output and in SUMMARY.txt -- it is never silently absent.
#
# EXIT CODE. 0 only if: at least one adapter was actually certified, every certified
# adapter passed every hard-pinned check, and no unacknowledged USABLE=0 hardware
# adapter was dropped from the run. Any missing .exe, missing .sslm artifact, or
# missing pins file is a loud, immediate FAIL -- never a silently-skipped cell and
# never a certification of a stale or absent binary.

param(
    [int]$OnlyAdapterIndex = -1,
    [string]$OnlyAdapterNameLike = "",
    [switch]$AcknowledgeUnusableAdapters
)

# NOTE: deliberately NOT "Stop". Every battery tool below can legitimately write to
# stderr (its own FAIL lines on a real failure, e.g. a bad adapter index during
# enumeration) -- with ErrorActionPreference=Stop, PowerShell treats ANY native-command
# stderr line as a terminating error and aborts the whole script before this script's own
# exit-code/output checks ever run. Every failure this script cares about is detected
# explicitly below via $LASTEXITCODE and output pattern matching, not via PowerShell's
# error stream, so "Continue" is the correct and safe setting here.
$ErrorActionPreference = "Continue"
$root = $PSScriptRoot
Set-Location $root

$bin = Join-Path $root "bin"
$art = Join-Path $root "artifacts"
$pinsFile = Join-Path $root "pins\pins.txt"
$stamp = Get-Date -Format "yyyyMMdd-HHmmss"
$resultsDir = Join-Path $root ("results\" + $stamp)
New-Item -ItemType Directory -Force -Path $resultsDir | Out-Null

$overallOk = $true
$failReasons = New-Object System.Collections.Generic.List[string]

function Fail-Loud($msg) {
    Write-Host "FAIL: $msg" -ForegroundColor Red
    $script:overallOk = $false
    $script:failReasons.Add($msg)
}

# ---------------------------------------------------------------------------
# Preflight: every exe, every artifact, the pins file. Missing = loud FAIL,
# never a silently-skipped cell.
# ---------------------------------------------------------------------------
$requiredExes = @(
    "t2116_list_adapters.exe",
    "t2039_c5_harness.exe",
    "t2100_gpu_throughput.exe",
    "t2113_b1_context_smoke.exe",
    "t2113_b2_model_smoke.exe",
    "t2113_b35_embed_smoke.exe",
    "t2113_b3_sequence_smoke.exe",
    "t2113_b5_async_smoke.exe",
    "t2113_b6_adapter_smoke.exe",
    "t2113_b6b_adapter_delta_smoke.exe",
    "t2113_b7_batch_smoke.exe",
    "t2113_b8_thread_smoke.exe"
)
foreach ($e in $requiredExes) {
    $p = Join-Path $bin $e
    if (-not (Test-Path $p)) { Fail-Loud "missing binary: $p" }
}
if (-not (Test-Path (Join-Path $bin "shaders"))) { Fail-Loud "missing shaders directory: $bin\shaders" }

$m15 = Join-Path $art "qwen2.5-1.5b-instruct.sslm"
$m05 = Join-Path $art "qwen2.5-0.5b-instruct.sslm"
$adp = Join-Path $art "qwen2.5-1.5b-shopkeeper-lora-v2-t2102-runtime.sslm"
foreach ($f in @($m15, $m05, $adp)) {
    if (-not (Test-Path $f)) { Fail-Loud "missing artifact: $f" }
}
if (-not (Test-Path $pinsFile)) { Fail-Loud "missing pins file: $pinsFile" }

if (-not $overallOk) {
    Write-Host ""
    Write-Host "CROSS-VENDOR GATE: FAIL -- package is incomplete, cannot certify anything." -ForegroundColor Red
    exit 1
}

# ---------------------------------------------------------------------------
# Load pins. Format: "TOOL=x EXIT=0 CHECKS=11 FAILURES=0 ..." one line per cell.
# ---------------------------------------------------------------------------
$pins = @{}
foreach ($line in (Get-Content $pinsFile)) {
    $line = $line.Trim()
    if ($line.Length -eq 0 -or $line.StartsWith("#")) { continue }
    $fields = @{}
    foreach ($tok in ($line -split '\s+')) {
        $kv = $tok -split '=', 2
        if ($kv.Length -eq 2) { $fields[$kv[0]] = $kv[1] }
    }
    if ($fields.ContainsKey("TOOL")) { $pins[$fields["TOOL"]] = $fields }
}

# ---------------------------------------------------------------------------
# Adapter enumeration: the dedicated t2116_list_adapters.exe (raw DXGI, no
# superslm_gpu dependency -- see that tool's own header comment for why probing the
# battery tools themselves cannot reliably tell "no more adapters" apart from "this
# adapter exists but is broken"). Its output lines look like:
#   INDEX=0 SOFTWARE=0 USABLE=1 VENDOR=0x1002 DEVICEID=0x744c DEDICATED_VIDEO_MB=24188 NAME=AMD Radeon RX 7900 XTX
#   INDEX=1 SOFTWARE=0 USABLE=0 VENDOR=... NAME=Some Broken Adapter
#   INDEX=2 SOFTWARE=1 NAME=Microsoft Basic Render Driver
#
# Software (WARP) adapters are never certification targets -- excluded entirely, not
# a result. A HARDWARE adapter that fails device creation (USABLE=0) is NOT the same
# thing: it is a real card the machine has, and dropping it from the run silently
# turns "this card could not be certified" into "this card was never asked" -- a gate
# that reads CROSS-VENDOR GATE: PASS while the card that mattered was never run. So a
# USABLE=0 hardware adapter that survives the -Only* filters is a certification
# FAILURE by default (loud, and it flips the final exit code), never a silent drop.
# -AcknowledgeUnusableAdapters downgrades that to an explicit, printed SKIP -- for the
# case where the machine legitimately carries a card you know is not being certified
# this trip (e.g. a known-dead iGPU) -- but even then it is named in the output and in
# SUMMARY.txt, never absent from it.
# ---------------------------------------------------------------------------
Write-Host "=== adapter enumeration ===" -ForegroundColor Cyan
$usableHW = New-Object System.Collections.Generic.List[hashtable]
$unusableHW = New-Object System.Collections.Generic.List[hashtable]
$listOut = & (Join-Path $bin "t2116_list_adapters.exe") 2>&1 | Out-String
foreach ($line in ($listOut -split "`n")) {
    $line = $line.Trim()
    if ($line.Length -eq 0) { continue }
    Write-Host "  $line"
    if ($line -match "^INDEX=(\d+) SOFTWARE=0 USABLE=1 .*NAME=(.+)$") {
        $usableHW.Add(@{ Index = [int]$Matches[1]; Name = $Matches[2].Trim() })
    }
    elseif ($line -match "^INDEX=(\d+) SOFTWARE=0 USABLE=0 .*NAME=(.+)$") {
        $unusableHW.Add(@{ Index = [int]$Matches[1]; Name = $Matches[2].Trim() })
    }
    # SOFTWARE=1 lines are printed above (raw enumeration) and intentionally not
    # collected into either list -- never a certification target, never a failure.
}

function Matches-Filter($a) {
    if ($OnlyAdapterIndex -ge 0 -and $a.Index -ne $OnlyAdapterIndex) { return $false }
    if ($OnlyAdapterNameLike -ne "" -and ($a.Name -notlike $OnlyAdapterNameLike)) { return $false }
    return $true
}

$selected = New-Object System.Collections.Generic.List[hashtable]
foreach ($a in $usableHW) { if (Matches-Filter $a) { $selected.Add($a) } }

$unusableSelected = New-Object System.Collections.Generic.List[hashtable]
foreach ($a in $unusableHW) { if (Matches-Filter $a) { $unusableSelected.Add($a) } }

$unusableSummaryLines = New-Object System.Collections.Generic.List[string]
foreach ($u in $unusableSelected) {
    if ($AcknowledgeUnusableAdapters) {
        $msg = "SKIP (acknowledged): adapter [$($u.Index)] $($u.Name) enumerated as hardware but FAILED device creation (USABLE=0) -- excluded from certification by -AcknowledgeUnusableAdapters, not certified this run"
        Write-Host $msg -ForegroundColor Yellow
        $unusableSummaryLines.Add($msg)
    } else {
        Fail-Loud "adapter [$($u.Index)] $($u.Name) enumerated as HARDWARE but FAILED device creation (USABLE=0) -- this is a certification FAILURE for that adapter, not a skip. If this card is not meant to be certified this run (e.g. a known-dead iGPU), re-run with -AcknowledgeUnusableAdapters to exclude it explicitly and certify the remaining adapter(s)."
        $unusableSummaryLines.Add("FAIL: adapter [$($u.Index)] $($u.Name) -- hardware, USABLE=0, not acknowledged")
    }
}

if ($selected.Count -eq 0 -and $unusableSelected.Count -eq 0) {
    Fail-Loud "no adapter matched the selection filter (OnlyAdapterIndex=$OnlyAdapterIndex OnlyAdapterNameLike='$OnlyAdapterNameLike') -- nothing to certify"
    exit 1
}
if ($selected.Count -eq 0) {
    # Every adapter that matched the filter was unusable hardware. Whether that is
    # fatal was already decided above (Fail-Loud if not acknowledged); either way
    # there is no adapter left to run the battery on, so nothing further to do.
    if (-not $overallOk) {
        Write-Host ""
        Write-Host "CROSS-VENDOR GATE: FAIL -- no usable adapter to certify; see the FAIL lines above." -ForegroundColor Red
        exit 1
    } else {
        Write-Host ""
        Write-Host "CROSS-VENDOR GATE: FAIL -- every matched adapter was acknowledged-unusable; nothing was certified." -ForegroundColor Red
        exit 1
    }
}

Write-Host ""
Write-Host ("Running the full battery on " + $selected.Count + " adapter(s).") -ForegroundColor Cyan

# Best-effort driver-version lookup, matched by name substring. Informational
# only -- never gates pass/fail.
$videoControllers = @()
try { $videoControllers = Get-CimInstance Win32_VideoController } catch { $videoControllers = @() }

# ---------------------------------------------------------------------------
# The battery. Each cell: key (matches pins TOOL=), exe, args, whether it is
# hard-pinned (checks/failures/exit) or has its own special parse (c5, t2100).
# ---------------------------------------------------------------------------
$battery = @(
    @{ Key = "c5";  Exe = "t2039_c5_harness.exe";              Args = @($m15, "0") }
    @{ Key = "b1";  Exe = "t2113_b1_context_smoke.exe";        Args = @() }
    @{ Key = "b2";  Exe = "t2113_b2_model_smoke.exe";          Args = @($m15, $m05) }
    @{ Key = "b3";  Exe = "t2113_b3_sequence_smoke.exe";       Args = @($m15, "24") }
    @{ Key = "b35"; Exe = "t2113_b35_embed_smoke.exe";         Args = @($m15) }
    @{ Key = "b5";  Exe = "t2113_b5_async_smoke.exe";          Args = @($m15) }
    @{ Key = "b6";  Exe = "t2113_b6_adapter_smoke.exe";        Args = @($m15, $m05, $adp) }
    @{ Key = "b6b"; Exe = "t2113_b6b_adapter_delta_smoke.exe"; Args = @($m15, $adp) }
    @{ Key = "b7";  Exe = "t2113_b7_batch_smoke.exe";          Args = @($m15, $adp) }
    @{ Key = "b8";  Exe = "t2113_b8_thread_smoke.exe";         Args = @($m15) }
    @{ Key = "t2100"; Exe = "t2100_gpu_throughput.exe";        Args = @($m15, "64", "0") }
)

$summaryLines = New-Object System.Collections.Generic.List[string]
$summaryLines.Add("T-2116 cross-vendor certification results")
$summaryLines.Add("machine: " + $env:COMPUTERNAME)
$summaryLines.Add("timestamp: " + $stamp)
$summaryLines.Add("")
foreach ($l in $unusableSummaryLines) { $summaryLines.Add($l) }
if ($unusableSummaryLines.Count -gt 0) { $summaryLines.Add("") }

$anyAdapterFailed = $false

foreach ($a in $selected) {
    $idx = $a.Index
    $name = $a.Name
    $safeName = ($name -replace '[^a-zA-Z0-9]+', '_')
    $adapterDir = Join-Path $resultsDir "adapter_${idx}_${safeName}"
    New-Item -ItemType Directory -Force -Path $adapterDir | Out-Null

    $driver = "unknown"
    foreach ($vc in $videoControllers) {
        if ($vc.Name -and ($name -like ("*" + $vc.Name + "*") -or $vc.Name -like ("*" + $name + "*"))) {
            $driver = $vc.DriverVersion
            break
        }
    }

    Write-Host ""
    Write-Host ("=== adapter [$idx] $name (driver: $driver) ===") -ForegroundColor Yellow
    $summaryLines.Add("adapter [$idx] $name  driver=$driver")

    $env:SSLM_GPU_ADAPTER_INDEX = "$idx"
    $adapterOk = $true

    foreach ($cell in $battery) {
        $exePath = Join-Path $bin $cell.Exe
        $logPath = Join-Path $adapterDir ($cell.Key + ".log")
        $sw = [System.Diagnostics.Stopwatch]::StartNew()
        $out = & $exePath @($cell.Args) 2>&1 | Out-String
        $sw.Stop()
        $ec = $LASTEXITCODE
        Set-Content -Path $logPath -Value $out -Encoding ASCII

        $pin = $pins[$cell.Key]
        $cellOk = $true
        $detail = ""

        # S3: the whole point of SSLM_GPU_ADAPTER_INDEX is that the adapter the
        # script asked for is the adapter that actually ran. Every "# adapter: NAME"
        # line this cell printed (a tool may create more than one context, e.g. b1's
        # two independent contexts) must name the requested adapter -- if the label
        # is absent, or names a different adapter, that is a certification failure
        # for this cell regardless of what its own checks/failures counters say,
        # because "which GPU actually ran" is exactly the thing that would be wrong.
        $labelMatches = [regex]::Matches($out, "# adapter: (.+)")
        if ($labelMatches.Count -eq 0) {
            $cellOk = $false
            $detail += " no '# adapter:' label printed (expected '$name')"
        } else {
            foreach ($lm in $labelMatches) {
                $labelName = $lm.Groups[1].Value.Trim()
                if ($labelName -ne $name) {
                    $cellOk = $false
                    $detail += " adapter label MISMATCH: printed '$labelName', requested '$name'"
                    break
                }
            }
        }

        if ($cell.Key -eq "c5") {
            $bitIdentical = ($out -match "RESULT: BIT-IDENTICAL")
            $argmaxCpu = ""; $argmaxGpu = ""
            if ($out -match "argmax token: CPU=(\d+) GPU=(\d+)") {
                $argmaxCpu = $Matches[1]; $argmaxGpu = $Matches[2]
            }
            if ($ec -ne [int]$pin["EXIT"]) { $cellOk = $false; $detail += " exit=$ec(want $($pin['EXIT']))" }
            if (-not $bitIdentical) { $cellOk = $false; $detail += " RESULT!=BIT-IDENTICAL" }
            if ($argmaxCpu -ne $pin["ARGMAX_CPU"] -or $argmaxGpu -ne $pin["ARGMAX_GPU"]) {
                $cellOk = $false
                $detail += " argmax CPU=$argmaxCpu GPU=$argmaxGpu (want CPU=$($pin['ARGMAX_CPU']) GPU=$($pin['ARGMAX_GPU']))"
            }
            if ($cellOk) { $detail = "argmax CPU=$argmaxCpu GPU=$argmaxGpu, BIT-IDENTICAL" }
        }
        elseif ($cell.Key -eq "t2100") {
            $equal = "UNKNOWN"
            if ($out -match "per-step CPU/GPU equality over \d+ timed steps: (\S+)") { $equal = $Matches[1] }
            if ($ec -ne [int]$pin["EXIT"]) { $cellOk = $false; $detail += " exit=$ec(want $($pin['EXIT']))" }
            if ($equal -ne $pin["EQUALITY"]) { $cellOk = $false; $detail += " equality=$equal (want $($pin['EQUALITY']))" }
            $gpuToks = ""
            if ($out -match "GPU \(RunLayerLoopGpu\)\s+\S+\s+s/token\s+(\S+)\s+tok/s") { $gpuToks = $Matches[1] }
            $cpuToks = ""
            if ($out -match "CPU \(RunLayerLoop\)\s+\S+\s+s/token\s+(\S+)\s+tok/s") { $cpuToks = $Matches[1] }
            $detail += "  equality=$equal  GPU=$gpuToks tok/s  CPU(host)=$cpuToks tok/s"
            $summaryLines.Add("  [t2100] cell=dispatch-path,adapter=$name,surface=decode_step: GPU=$gpuToks tok/s CPU(host)=$cpuToks tok/s (throughput, unpinned)")
        }
        else {
            $checks = ""; $failures = ""
            if ($out -match "checks=(\d+)\s+failures=(\d+)") {
                $checks = $Matches[1]; $failures = $Matches[2]
            }
            if ($ec -ne [int]$pin["EXIT"]) { $cellOk = $false; $detail += " exit=$ec(want $($pin['EXIT']))" }
            if ($checks -ne $pin["CHECKS"]) { $cellOk = $false; $detail += " checks=$checks(want $($pin['CHECKS']))" }
            if ($failures -ne $pin["FAILURES"]) { $cellOk = $false; $detail += " failures=$failures(want $($pin['FAILURES']))" }
            if ($cellOk) { $detail = "checks=$checks failures=$failures" }

            # Throughput lines this cell happens to print, captured and labeled but
            # never gated -- every "tok/s" line in this cell's own output.
            foreach ($tl in ($out -split "`n")) {
                if ($tl -match "tok/s") {
                    $summaryLines.Add("  [$($cell.Key)] cell=$($cell.Key),adapter=$name,surface=throughput: " + $tl.Trim() + " (throughput, unpinned)")
                }
            }
        }

        if (-not $cellOk) { $adapterOk = $false }
        $status = if ($cellOk) { "PASS" } else { "FAIL" }
        $color = if ($cellOk) { "Green" } else { "Red" }
        Write-Host ("  [{0,-5}] {1,-4} {2}s  {3}" -f $cell.Key, $status, [math]::Round($sw.Elapsed.TotalSeconds,1), $detail) -ForegroundColor $color
        $summaryLines.Add("  [$($cell.Key)] $status  $detail")
    }

    Remove-Item Env:\SSLM_GPU_ADAPTER_INDEX -ErrorAction SilentlyContinue

    $verdict = if ($adapterOk) { "PASS" } else { "FAIL" }
    $verdictColor = if ($adapterOk) { "Green" } else { "Red" }
    Write-Host ("ADAPTER [$idx] $name : $verdict") -ForegroundColor $verdictColor
    $summaryLines.Add("ADAPTER [$idx] $name : $verdict")
    $summaryLines.Add("")
    if (-not $adapterOk) { $anyAdapterFailed = $true }
}

$summaryPath = Join-Path $resultsDir "SUMMARY.txt"
Set-Content -Path $summaryPath -Value ($summaryLines -join "`r`n") -Encoding ASCII

Write-Host ""
Write-Host "Results written to: $summaryPath"

# Final verdict combines three things, none of which alone is sufficient: every
# selected (usable, filter-matched) adapter's own battery passed ($anyAdapterFailed),
# no unacknowledged unusable hardware adapter was dropped from the run ($overallOk,
# set false by Fail-Loud above), and at least one adapter was actually certified
# ($selected.Count -gt 0, already enforced before the battery loop runs at all).
if ($anyAdapterFailed -or -not $overallOk) {
    Write-Host "CROSS-VENDOR GATE: FAIL -- see the FAIL lines above and $summaryPath" -ForegroundColor Red
    exit 1
} else {
    Write-Host "CROSS-VENDOR GATE: PASS -- every certified adapter matched the RTX 2080 SUPER reference pins on every determinism check, and every hardware adapter this machine enumerated was either certified or explicitly acknowledged as skipped." -ForegroundColor Green
    exit 0
}
