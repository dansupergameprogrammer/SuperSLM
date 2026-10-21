# Step 1d: re-runs the failed fast-set cells with CI's setup: the generated fixtures (tests.yml windows-x64 env) and
# the source tree's build\ folder, which the t2919/t2933 Python cells use as their temp parent. Windows PowerShell 5.1.
$ErrorActionPreference = 'Continue'
$wt = 'D:\SuperSLM\.worktrees\sfd-arm-aa0bc94c'
$bd = "$wt\build-ctest"
$out = 'D:\_scratch\box-2026-10-21'
$status = "$out\step1-status.txt"
function Say($t) { for ($i = 0; $i -lt 20; $i++) { try { Add-Content -LiteralPath $status -Value "[$(Get-Date -Format o)] $t" -ErrorAction Stop; return } catch { Start-Sleep -Milliseconds 250 } } }
try {
    . "$wt\tests\t2130-g5-red-suite\resolve_toolchain.ps1"
    Enter-SuperSlmVsDevShell | Out-Null
    $env:SUPERSLM_DECODE_THREADING_FIXTURE_DIR = "$out\dt-fixtures"
    $env:SUPERSLM_SCHEMA_FAST_DECODE_FIXTURE_DIR = "$out\sfd-fixtures"
    New-Item -ItemType Directory -Force -Path "$wt\build" | Out-Null
    $t0 = Get-Date
    Push-Location $bd
    try { & ctest --output-on-failure --timeout 600 -R '^(superslm_tests|superslm_decode_threading_alloc|superslm_schema_fast_decode_alloc|t2919_g5_schema_compiler_tests|t2933_schema_query_contract|t2933_schema_query_native_state|t2933_schema_lifecycle_native_state)$' *> "$out\step1-rerun2.log" } finally { Pop-Location }
    Say "rerun2 ctest exit $LASTEXITCODE after $([math]::Round(((Get-Date) - $t0).TotalMinutes, 1)) min"
    Say 'rerun2 end PASS'
}
catch { Say "rerun2 end FAIL $($_.Exception.Message)" }
