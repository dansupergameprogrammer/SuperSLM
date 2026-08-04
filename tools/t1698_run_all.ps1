$ErrorActionPreference = "Stop"
Set-Location D:\SuperSLM\.worktrees\t1697-outlier-migration

$runs = @(
    # --- Arm 1: density sweep around the alpha=0.18 spike, channel 609 ---
    @{ch=609; alpha=0.16;  label="alpha0.160"},
    @{ch=609; alpha=0.17;  label="alpha0.170"},
    @{ch=609; alpha=0.175; label="alpha0.175"},
    @{ch=609; alpha=0.18;  label="alpha0.180_v2"},
    @{ch=609; alpha=0.185; label="alpha0.185"},
    @{ch=609; alpha=0.19;  label="alpha0.190"},
    @{ch=609; alpha=0.195; label="alpha0.195"},
    # --- Arm 2: placebo channels, perturbation-matched to ch609/alpha0.18 ---
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
    Write-Host "`nALL RUNS DONE -- FAILURES: $($failed -join ', ')"
    exit 1
} else {
    Write-Host "`nALL RUNS DONE -- ALL 12 PASSED"
    exit 0
}
