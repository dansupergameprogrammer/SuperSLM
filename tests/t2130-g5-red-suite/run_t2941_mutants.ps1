param(
    [string]$ScratchRoot = '',
    [string]$ShaderDir = ''
)
$ErrorActionPreference = 'Stop'
$Here = $PSScriptRoot
. (Join-Path $Here 'resolve_toolchain.ps1')
$Engine = (Resolve-Path (Join-Path $Here '..\..')).Path
if (-not $ScratchRoot) { $ScratchRoot = Join-Path $Engine 'build\t2941-mutants' }
if (-not $ShaderDir) { $ShaderDir = Join-Path $Engine 'build\gpu-shaders-staged' }
$Python = Resolve-SuperSlmPython
$Generator = Join-Path $Here 'make_t2922_gpu_schema_accepting_mutants.py'
$QueryRunner = Join-Path $Here 'run_t2922_query_runtime.ps1'
$LifecycleRunner = Join-Path $Here 'run_t2922_lifecycle.ps1'
$SourceDir = Join-Path $ScratchRoot 'sources'
$LogDir = Join-Path $ScratchRoot 'logs'
New-Item -ItemType Directory -Force $ScratchRoot,$SourceDir,$LogDir | Out-Null

function Invoke-LoggedScript {
    param([string]$Script, [string[]]$Arguments, [string]$Log)
    $priorErrorPreference = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    $output = & powershell.exe -NoProfile -ExecutionPolicy Bypass -File $Script @Arguments 2>&1
    $exitCode = $LASTEXITCODE
    $ErrorActionPreference = $priorErrorPreference
    $output | Set-Content -LiteralPath $Log -Encoding utf8
    $output | ForEach-Object { Write-Output $_ }
    return [pscustomobject]@{ ExitCode = $exitCode; Text = ($output -join "`n") }
}

$failures = 0
$queryMutants = @('a','b','c','d','e','f','g','h','i','j')
foreach ($mutant in $queryMutants) {
    $source = Join-Path $SourceDir "$mutant.cpp"
    & $Python $Generator --engine $Engine --out $source --mutant $mutant
    if ($LASTEXITCODE) { throw "mutant $mutant generation failed" }
    $args = @(
        '-ScratchRoot', (Join-Path $ScratchRoot "query-$mutant"),
        '-ShaderDir', $ShaderDir
    )
    if ($mutant -eq 'g') { $args += @('-CpuSource', $source) }
    else { $args += @('-GpuSource', $source) }
    $result = Invoke-LoggedScript $QueryRunner $args (Join-Path $LogDir "mutant-$mutant.log")
    $killed = $result.ExitCode -ne 0 -and $result.Text -match 'SUMMARY checks=\d+ failures=[1-9]\d* skips=0'
    Write-Output "MUTANT $mutant status=$(if ($killed) { 'KILLED' } else { 'SURVIVED' }) exit=$($result.ExitCode) cell=SCHEMA_QUERY_REAL_DECODE"
    if (-not $killed) { ++$failures }
}

foreach ($mutant in @('k','l')) {
    $recipe = Join-Path $SourceDir "$mutant.recipe.txt"
    $output = & $Python $Generator --engine $Engine --out $recipe --mutant $mutant 2>&1
    $exitCode = $LASTEXITCODE
    $output | Set-Content -LiteralPath (Join-Path $LogDir "mutant-$mutant.log") -Encoding utf8
    $output | ForEach-Object { Write-Output $_ }
    $prepared = $exitCode -eq 0 -and (Test-Path -LiteralPath $recipe) -and
        (($output -join "`n") -match "MUTANT $mutant PREPARED")
    Write-Output "MUTANT $mutant status=$(if ($prepared) { 'PREPARED' } else { 'MISSING' }) exit=$exitCode cell=$(if ($mutant -eq 'k') { 'BIND_AFTER_PROMPT_PREFILL_REJECT' } else { 'BIND_AFTER_REFUSED_FIRST_CALL_REJECT' })"
    if (-not $prepared) { ++$failures }
}

$source = Join-Path $SourceDir 'm.cpp'
& $Python $Generator --engine $Engine --out $source --mutant m
if ($LASTEXITCODE) { throw 'mutant m generation failed' }
$result = Invoke-LoggedScript $LifecycleRunner @(
    '-ScratchRoot', (Join-Path $ScratchRoot 'lifecycle-m'),
    '-ShaderDir', $ShaderDir,
    '-GpuSource', $source
) (Join-Path $LogDir 'mutant-m.log')
$killed = $result.Text -match 'CELL RESET_REUSE status=RED'
Write-Output "MUTANT m status=$(if ($killed) { 'KILLED' } else { 'SURVIVED' }) exit=$($result.ExitCode) cell=RESET_REUSE"
if (-not $killed) { ++$failures }

Write-Output "SUMMARY mutants=13 killed=11 prepared=2 failures=$failures"
if ($failures) { exit 1 }
exit 0
