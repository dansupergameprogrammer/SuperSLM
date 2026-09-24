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
# TE-402 S1 (Claude/Poirot/fcbc6a7-slm171-stdout-fix.md): "Stop" plus a bare `2>&1` on the exe made
# this script abort under Windows PowerShell 5.1 the moment the fixed `set` leg wrote its label to
# stderr -- 5.1 wraps each native stderr line from `2>&1` in a terminating NativeCommandError, so
# the script threw before reaching the exit-code grading below and a genuinely green suite read as
# a hard failure. `run_crossvendor.ps1:59-66` already carries this exact hazard and uses
# "Continue"; matched here. `ForEach-Object { "$_" }` stringifies each captured record (native
# stdout/stderr line or ErrorRecord alike) before `Out-String` renders it, the same shape Poirot's
# own remedy probe (`Claude/Poirot/fcbc6a7-slm171-stdout-fix-probe/remedy_probe.ps1`) proved S3
# clean under both 5.1 and pwsh 7 with -- kept here for parity even though this script only grades
# by exit code, never by parsing $out, so a shell that mis-renders the label cannot mis-grade this
# suite the way it could S3's own regex.
$ErrorActionPreference = "Continue"
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
	$out = & $exe "--model=$Model" "--mode=$mode" "--adapter-index=$AdapterIndex" 2>&1 |
		ForEach-Object { "$_" } | Out-String
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
