$ErrorActionPreference = "Stop"
Set-Location D:\SuperSLM\.worktrees\t1697-outlier-migration

$runs = @(
    # --- Arm 2 only (reframed as noise-floor calibration): placebo channels,
    # perturbation-matched to ch609/alpha0.18 (t1698_match_search.log) ---
    @{ch=269;  alpha=0.175; label="placebo_ch269"},
    @{ch=870;  alpha=0.165; label="placebo_ch870"},
    @{ch=5967; alpha=0.17;  label="placebo_ch5967"},
    @{ch=6015; alpha=0.17;  label="placebo_ch6015"},
    @{ch=7693; alpha=0.175; label="placebo_ch7693"}
)

$failed = @()
foreach ($r in $runs) {
    Write-Host "`n########## RUN START: ch=$($r.ch) alpha=$($r.alpha) label=$($r.label) ##########"
    python tools\t1698_run_arm.py $r.ch $r.alpha $r.label 2>&1 | Tee-Object -FilePath "out\t1698_run_$($r.label).log"
    if ($LASTEXITCODE -ne 0) {
        Write-Host "########## RUN FAILED: $($r.label) (exit $LASTEXITCODE) ##########"
        $failed += $r.label
    } else {
        Write-Host "########## RUN OK: $($r.label) ##########"
    }
}

if ($failed.Count -gt 0) {
    Write-Host "`nALL PLACEBO RUNS DONE -- FAILURES: $($failed -join ', ')"
    exit 1
} else {
    Write-Host "`nALL PLACEBO RUNS DONE -- ALL 5 PASSED"
    exit 0
}
