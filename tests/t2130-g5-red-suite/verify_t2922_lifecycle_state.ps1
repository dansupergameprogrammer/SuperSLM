if (-not $env:SUPERSLM_T2933_LIFECYCLE_FIXTURE) {
    Write-Output 'SKIP lifecycle native run: SUPERSLM_T2933_LIFECYCLE_FIXTURE unset'
    exit 0
}
if (-not (Test-Path $env:SUPERSLM_T2933_LIFECYCLE_FIXTURE)) {
    Write-Error "required lifecycle fixture missing: $env:SUPERSLM_T2933_LIFECYCLE_FIXTURE"
    exit 2
}
& (Join-Path $PSScriptRoot 'run_t2922_lifecycle.ps1') `
    -CpuGuardBase $env:SUPERSLM_T2933_LIFECYCLE_FIXTURE
$code = $LASTEXITCODE
if ($code -eq 0 -or $code -eq 1) { exit 0 }
exit $code
