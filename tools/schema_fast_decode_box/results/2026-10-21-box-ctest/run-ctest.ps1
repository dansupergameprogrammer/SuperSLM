# Step 1c: builds the two small EXCLUDE_FROM_ALL precache identity cells, then runs the fast ctest set: everything but
# the long precache grid (superslm_precache_e1: real_two_process, e2_reference, e3_reference). Windows PowerShell 5.1.
$ErrorActionPreference = 'Stop'
$wt = 'D:\SuperSLM\.worktrees\sfd-arm-aa0bc94c'
$bd = "$wt\build-ctest"
$out = 'D:\_scratch\box-2026-10-21'
$status = "$out\step1-status.txt"
function Say($t) { for ($i = 0; $i -lt 20; $i++) { try { Add-Content -LiteralPath $status -Value "[$(Get-Date -Format o)] $t" -ErrorAction Stop; return } catch { Start-Sleep -Milliseconds 250 } } }
try {
    . "$wt\tests\t2130-g5-red-suite\resolve_toolchain.ps1"
    Enter-SuperSlmVsDevShell | Out-Null
    & cmake --build $bd --target superslm_precache_reference_header superslm_precache_compiler_job_identity *> "$out\step1-build2.log"
    if ($LASTEXITCODE -ne 0) { throw 'identity build failed (step1-build2.log)' }
    Say 'identity cells built'
    $t0 = Get-Date
    Push-Location $bd
    try { & ctest --output-on-failure --timeout 600 -E 'superslm_precache_(real_two_process|e2_reference|e3_reference)$' *> "$out\step1-ctest.log" } finally { Pop-Location }
    Say "ctest exit $LASTEXITCODE after $([math]::Round(((Get-Date) - $t0).TotalMinutes, 1)) min"
    Say 'end PASS'
}
catch { Say "end FAIL $($_.Exception.Message)" }
