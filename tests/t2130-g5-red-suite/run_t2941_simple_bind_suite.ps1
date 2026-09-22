param(
    [string]$ScratchRoot = '',
    [string]$ShaderDir = ''
)
$ErrorActionPreference = 'Stop'
$Here = $PSScriptRoot
$Engine = (Resolve-Path (Join-Path $Here '..\..')).Path
if (-not $ScratchRoot) { $ScratchRoot = Join-Path $Engine 'build\t2941-simple-bind-suite' }
if (-not $ShaderDir) { $ShaderDir = Join-Path $Engine 'build\gpu-shaders-staged' }
$Python = 'C:\Users\dansu\AppData\Local\Programs\Python\Python313\python.exe'
if (-not (Test-Path -LiteralPath $Python)) { $Python = 'python' }
New-Item -ItemType Directory -Force $ScratchRoot | Out-Null

function Invoke-Logged {
    param([string]$Name, [scriptblock]$Action)
    $priorErrorPreference = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    $output = & $Action 2>&1
    $exitCode = $LASTEXITCODE
    $ErrorActionPreference = $priorErrorPreference
    $output | Set-Content -LiteralPath (Join-Path $ScratchRoot "$Name.log") -Encoding utf8
    return [pscustomobject]@{ ExitCode = $exitCode; Lines = @($output) }
}

$portable = Invoke-Logged 'portable-contract' {
    & $Python -m unittest "$Here\t2922_schema_query_contract.py" -v
}
if ($portable.ExitCode -ne 0) {
    Write-Output "APPARATUS portable_contract=FAIL exit=$($portable.ExitCode)"
    exit 2
}

$lifecycle = Invoke-Logged 'lifecycle' {
    & powershell.exe -NoProfile -ExecutionPolicy Bypass -File (Join-Path $Here 'run_t2922_lifecycle.ps1') `
        -ScratchRoot (Join-Path $ScratchRoot 'lifecycle-work') -ShaderDir $ShaderDir
}
$cellLines = @($lifecycle.Lines | ForEach-Object { "$_" } | Where-Object { $_ -match '^CELL ' })
if ($cellLines.Count -ne 8 -or -not ($lifecycle.Lines -join "`n" -match 'SUMMARY checks=8 failures=\d+ skips=0')) {
    Write-Output "APPARATUS lifecycle_population=FAIL cell_lines=$($cellLines.Count) exit=$($lifecycle.ExitCode)"
    exit 2
}

$query = Invoke-Logged 'query-runtime' {
    & powershell.exe -NoProfile -ExecutionPolicy Bypass -File (Join-Path $Here 'run_t2922_query_runtime.ps1') `
        -ScratchRoot (Join-Path $ScratchRoot 'query-work') -ShaderDir $ShaderDir
}
$priorGate = $env:SUPERSLM_G5_REAL_MODEL_TESTS
$env:SUPERSLM_G5_REAL_MODEL_TESTS = '1'
try {
    $real = Invoke-Logged 'real-model' {
        & powershell.exe -NoProfile -ExecutionPolicy Bypass -File (Join-Path $Here 'run_t2922_real_model.ps1') `
            -ScratchRoot (Join-Path $ScratchRoot 'real-work') -ShaderDir $ShaderDir
    }
} finally {
    $env:SUPERSLM_G5_REAL_MODEL_TESTS = $priorGate
}

$green = 0
$red = 0
foreach ($line in $cellLines) {
    Write-Output $line
    if ($line -match ' status=GREEN ') { ++$green } else { ++$red }
}
$queryGreen = $query.ExitCode -eq 0 -and $real.ExitCode -eq 0 -and
    (($query.Lines -join "`n") -match 'SUMMARY checks=\d+ failures=0 skips=0') -and
    (($real.Lines -join "`n") -match 'SUMMARY checks=\d+ failures=0 skips=0')
if ($queryGreen) { ++$green } else { ++$red }
Write-Output "CELL SCHEMA_QUERY_REAL_DECODE status=$(if ($queryGreen) { 'GREEN' } else { 'RED' }) reason=query_exit=$($query.ExitCode),real_model_exit=$($real.ExitCode); registered real decode must reach STOP=budget"
Write-Output "SUMMARY cells=9 green=$green red=$red apparatus_failures=0"
if ($red) { exit 1 }
exit 0
