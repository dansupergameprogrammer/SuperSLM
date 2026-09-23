# TE-399 (Curie) -- runs cell_gpu_pipeline_stdout_empty.exe (build.bat) twice, as two separate
# processes (harness::GetDevice() is a per-process magic static -- a second in-process open would
# under-count the defect, see the cell's own header comment), once with SSLM_GPU_ADAPTER_INDEX
# unset and once set to a valid index, and reports both legs' pass/fail and byte counts. This is
# the property D-SLM7753 rules into SuperSLM v1.7.1: the engine library writes nothing to stdout.
#
# Usage: run_cell.ps1 -Model D:\path\to\model.sslm [-AdapterIndex 0] [-ExeDir D:\_te399\build]
param(
	[Parameter(Mandatory = $true)][string]$Model,
	[int]$AdapterIndex = 0,
	[string]$ExeDir = "D:\_te399\build"
)
$ErrorActionPreference = "Stop"
$exe = Join-Path $ExeDir "te399_stdout_pipeline.exe"
if (-not (Test-Path $exe)) { throw "not built: $exe (run build.bat first)" }

# Stage shaders next to the executable (E13: the harness resolves shaders relative to the exe's
# own directory by default) -- same as Claude/Loki/te393-u3-strike-probe/build.bat's own staging
# step.
$shadersSrc = Join-Path $ExeDir "gpu-shaders-staged"
$shadersDst = Join-Path $ExeDir "shaders"
if (Test-Path $shadersSrc) {
	New-Item -ItemType Directory -Force -Path $shadersDst | Out-Null
	Copy-Item (Join-Path $shadersSrc "*.cso") $shadersDst -Force
}

function Run-Leg([string]$mode) {
	$env:SSLM_GPU_ADAPTER_INDEX = $null
	Remove-Item Env:\SSLM_GPU_ADAPTER_INDEX -ErrorAction SilentlyContinue
	if ($mode -eq "set") { $env:SSLM_GPU_ADAPTER_INDEX = "$AdapterIndex" }
	$out = & $exe "--model=$Model" "--mode=$mode" "--adapter-index=$AdapterIndex" 2>&1 | Out-String
	$ec = $LASTEXITCODE
	Remove-Item Env:\SSLM_GPU_ADAPTER_INDEX -ErrorAction SilentlyContinue
	[pscustomobject]@{ Mode = $mode; ExitCode = $ec; Output = $out }
}

$unsetResult = Run-Leg "unset"
$setResult = Run-Leg "set"

Write-Host "=== mode=unset (SSLM_GPU_ADAPTER_INDEX not set) ==="
Write-Host $unsetResult.Output
Write-Host "exit=$($unsetResult.ExitCode)"
Write-Host ""
Write-Host "=== mode=set (SSLM_GPU_ADAPTER_INDEX=$AdapterIndex) ==="
Write-Host $setResult.Output
Write-Host "exit=$($setResult.ExitCode)"
Write-Host ""

$skipped = ($unsetResult.ExitCode -eq 3) -or ($setResult.ExitCode -eq 3)
if ($skipped) {
	Write-Host "CENSUS: SKIPPED -- no usable D3D12 hardware adapter on this machine"
	exit 3
}

$unsetPass = $unsetResult.ExitCode -eq 0
$setPass = $setResult.ExitCode -eq 0
Write-Host "CENSUS: mode=unset $(if ($unsetPass) {'PASS'} else {'FAIL'}); mode=set $(if ($setPass) {'PASS'} else {'FAIL'})"
if ($unsetPass -and -not $setPass) {
	Write-Host "RED, as expected before SuperSLM v1.7.1: the set leg's public-API pipeline wrote nonzero bytes to stdout."
} elseif ($unsetPass -and $setPass) {
	Write-Host "GREEN: both legs wrote zero bytes to stdout."
} else {
	Write-Host "UNEXPECTED: the unset (control) leg itself failed -- see its output above."
}
exit ($(if ($unsetPass -and $setPass) { 0 } else { 1 }))
