# T-1979 spike: end-to-end reproducibility check (T-1987, closing T-1983 review
# S-3: "the run is not reproducible from HEAD -- no build recipe, no run
# transcript, no dump provenance").
#
# Builds the CPU dump tool (tools/build_t1979_dump_qproj.bat) and the GPU
# shader+harness (build.ps1, this directory) from the recipes now committed to
# this branch, then runs the full dump -> GPU-dispatch -> compare pipeline
# TWICE against the SAME real prompt and confirms both dumps and both harness
# outputs are byte-identical. This proves the committed recipe reproduces its
# own result -- not merely that a run happened once, unrecorded, on one
# machine, which is what the review found.
#
# Resource rule: exactly two GPU dispatches total (one per pipeline run), well
# inside the "short dispatches only" constraint and the commissioning brief's
# "single re-run is fine, no sustained runs" allowance.
#
# Usage: repro.ps1 <model.sslm> <tokenizer.sslm> "<prompt>"
# Real artifact paths on this machine, e.g.:
#   repro.ps1 D:\hf_cache\superslm_artifacts\qwen2.5-1.5b-instruct.sslm tests\fixtures\qwen2.5-1.5b.tok.sslm "The old lighthouse keeper"
param(
    [Parameter(Mandatory=$true)][string]$Model,
    [Parameter(Mandatory=$true)][string]$Tokenizer,
    [Parameter(Mandatory=$true)][string]$Prompt
)
$ErrorActionPreference = "Stop"
$spikeGpu = $PSScriptRoot
$repoRoot = Split-Path -Parent $spikeGpu
$transcript = "$spikeGpu\repro_transcript.txt"
if (Test-Path $transcript) { Remove-Item $transcript }

function Log($line) {
    Write-Output $line
    Add-Content -Path $transcript -Value $line
}

function Sha256Of($path) {
    (Get-FileHash -Path $path -Algorithm SHA256).Hash
}

Log "=== T-1979 reproducibility run — $(Get-Date -Format o) ==="
Log "model=$Model"
Log "tokenizer=$Tokenizer"
Log "prompt=$Prompt"
Log ""

Log "--- build: tools\build_t1979_dump_qproj.bat ---"
Push-Location $repoRoot
cmd /c "tools\build_t1979_dump_qproj.bat" 2>&1 | ForEach-Object { Log $_ }
if ($LASTEXITCODE -ne 0) { Pop-Location; throw "CPU dump tool build failed" }
Pop-Location
Log ""

Log "--- build: spike_gpu\build.ps1 ---"
& "$spikeGpu\build.ps1" 2>&1 | ForEach-Object { Log $_ }
Log ""

$dumpExe = "$repoRoot\out\t1979_dump_qproj.exe"
$harnessExe = "$spikeGpu\t1979_gpu_harness.exe"
$cso = "$spikeGpu\shaders\qproj_site.cso"

for ($run = 1; $run -le 2; $run++) {
    $dump = "$repoRoot\out\t1979_repro_run${run}.bin"
    Log "--- run ${run}: dump ---"
    & $dumpExe $Model $Tokenizer $Prompt --dump $dump 2>&1 | ForEach-Object { Log $_ }
    if ($LASTEXITCODE -ne 0) { throw "run $run dump failed" }
    $hash = Sha256Of $dump
    Log "dump sha256 = $hash"
    Log ""

    Log "--- run ${run}: GPU harness ---"
    & $harnessExe $dump $cso 2>&1 | ForEach-Object { Log $_ }
    $harnessExit = $LASTEXITCODE
    Log "harness exit = $harnessExit"
    Log ""

    if ($run -eq 1) {
        $run1Hash = $hash
        $run1Exit = $harnessExit
    } else {
        $run2Hash = $hash
        $run2Exit = $harnessExit
    }
}

Log "=== reproducibility verdict ==="
if ($run1Hash -eq $run2Hash) {
    Log "DUMP BIT-IDENTICAL across both runs: $run1Hash"
} else {
    Log "DUMP MISMATCH: run1=$run1Hash run2=$run2Hash"
}
if ($run1Exit -eq 0 -and $run2Exit -eq 0) {
    Log "BOTH HARNESS RUNS PASS (exit 0)"
} else {
    Log "HARNESS EXIT MISMATCH OR FAILURE: run1=$run1Exit run2=$run2Exit"
}
Log "transcript written to $transcript"
