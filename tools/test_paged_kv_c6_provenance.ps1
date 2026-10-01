# Regression test for tools/paged_kv_c6_provenance.ps1 (the C6 box run's resume markers and standing).
# Self-contained (no Pester): runs on any pwsh or Windows PowerShell, exits 1 on the first failure.
#   pwsh -NoProfile -File tools/test_paged_kv_c6_provenance.ps1
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'paged_kv_c6_provenance.ps1')

$script:failures = 0
function Check([bool]$ok, [string]$what) {
    if ($ok) { Write-Output "ok   $what" } else { Write-Output "FAIL $what"; $script:failures++ }
}
function Throws([scriptblock]$b, [string]$pattern, [string]$what) {
    try { & $b; Check $false "$what (did not throw)" }
    catch { Check ($_.Exception.Message -match $pattern) "$what ($($_.Exception.Message))" }
}

$tmp = Join-Path ([IO.Path]::GetTempPath()) ("pkv-c6-prov-" + [guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $tmp | Out-Null
try {
    $inputs = [ordered]@{ 'artifact:a.sslm' = ('a' * 64); 'baseline:79_v1.11.0.txt' = ('b' * 64) }
    $common = @{ Head = ('c' * 40); Binary = 'X:\b\superslm_pkv_c6.exe'; BinarySha256 = ('d' * 64); Inputs = $inputs }
    $marker = Join-Path $tmp '7.9_C6.done'

    # ---- S2: quarantined run, then a resume with -Commissioned and no -Fresh ------------------------
    # Run 1: nothing commissioned. The leg runs and writes its record.
    $run1 = New-PkvLegRecord -Leg '7.9/C6' -Rests @('timing79') -Commissioned @() @common
    Check ((Get-PkvRecordEnv $run1).Count -eq 0) 'run 1 (quarantined) sets no assertion variable'
    $run1.Outcome = 'GREEN exit=0 1 cells, 0 red; 9 checks, 0 failures'
    Write-PkvLegRecord $marker $run1
    $stored = Read-PkvLegRecord $marker
    Check ($null -ne $stored) 'run 1 marker reads back as a record'
    Check (Test-PkvLegRecordCurrent $stored $run1) 'a record round-trips through the marker file unchanged'
    Check ((Get-PkvLegStanding -Rests @('timing79') -Record $stored) -eq 'quarantined (timing79)') 'run 1 stands quarantined'

    # Run 2: same HEAD, binary and inputs (-NoBuild), now -Commissioned timing79. Must not skip.
    $run2 = New-PkvLegRecord -Leg '7.9/C6' -Rests @('timing79') -Commissioned @('timing79') @common
    Check (-not (Test-PkvLegRecordCurrent $stored $run2)) 'resume with -Commissioned timing79 does NOT skip the quarantined run'
    Check ((Get-PkvLegRecordDiff $stored $run2) -join ',' -eq 'Commissioned,AssertEnv') 'the difference is named: Commissioned, AssertEnv'
    # Even before the rerun, the stored (quarantined) record's standing ignores the current arguments.
    Check ((Get-PkvLegStanding -Rests @('timing79') -Record $stored) -like 'quarantined*') 'stored quarantined record stays quarantined under new arguments'
    # The rerun asserts and writes its own record: only that one stands.
    Check ((Get-PkvRecordEnv $run2)['SUPERSLM_PAGED_KV_TIMING_COMMISSIONED'] -eq '1') 'run 2 sets SUPERSLM_PAGED_KV_TIMING_COMMISSIONED=1'
    $run2.Outcome = 'GREEN exit=0'
    Write-PkvLegRecord $marker $run2
    $stored = Read-PkvLegRecord $marker
    Check ((Get-PkvLegStanding -Rests @('timing79') -Record $stored) -eq 'standing') 'the asserted rerun stands'
    Check (Test-PkvLegRecordCurrent $stored (New-PkvLegRecord -Leg '7.9/C6' -Rests @('timing79') -Commissioned @('timing79') @common)) 'a third identical invocation skips'
    # And the reverse: a later quarantined invocation does not reuse the standing record either.
    Check (-not (Test-PkvLegRecordCurrent $stored $run1)) 'dropping -Commissioned reruns too'

    # Any other provenance change reruns.
    foreach ($k in 'Head', 'BinarySha256') {
        $c = @{} + $common; $c[$k] = ('e' * $c[$k].Length)
        Check (-not (Test-PkvLegRecordCurrent $stored (New-PkvLegRecord -Leg '7.9/C6' -Rests @('timing79') -Commissioned @('timing79') @c))) "a changed $k reruns"
    }
    $c = @{} + $common; $c.Inputs = [ordered]@{ 'artifact:a.sslm' = ('a' * 64); 'baseline:79_v1.11.0.txt' = ('f' * 64) }
    Check (-not (Test-PkvLegRecordCurrent $stored (New-PkvLegRecord -Leg '7.9/C6' -Rests @('timing79') -Commissioned @('timing79') @c))) 'a changed baseline reruns'
    # Only the ids a leg rests on are recorded, and -Max74Ratio only for 7.4: other instruments'
    # commissioning, or a ratio, does not rerun a 7.9 leg.
    Check (Test-PkvLegRecordCurrent $stored (New-PkvLegRecord -Leg '7.9/C6' -Rests @('timing79') -Commissioned @('oracle', 'admission', 'timing74', 'timing79') -Max74Ratio '2.0' @common)) 'commissioning other instruments and a ratio does not rerun 7.9'

    # ---- commissioning timing79 reruns only the legs resting on it --------------------------------
    $before = @{ Commissioned = @('oracle', 'admission', 'timing74'); Max74Ratio = '2.0' }
    $after = @{ Commissioned = @('oracle', 'admission', 'timing74', 'timing79'); Max74Ratio = '2.0' }
    $legsUnderTest = @(
        @{ Leg = '10.1/C6'; Rests = @('oracle'); Rerun = $false },
        @{ Leg = '10.6/C6'; Rests = @('admission', 'oracle'); Rerun = $false },
        @{ Leg = '7.4/C6'; Rests = @('timing74'); Rerun = $false },
        @{ Leg = 'lifecycle/C6'; Rests = @('lifecycle'); Rerun = $false },
        @{ Leg = '7.9-v1.11.0'; Rests = @('timing79'); Rerun = $true },
        @{ Leg = '7.9/C6'; Rests = @('timing79'); Rerun = $true }
    )
    foreach ($t in $legsUnderTest) {
        $m = Join-Path $tmp (($t.Leg -replace '[/\\:]', '_') + '.done')
        $old = New-PkvLegRecord -Leg $t.Leg -Rests $t.Rests @before @common
        $old.Outcome = 'GREEN exit=0'
        Write-PkvLegRecord $m $old
        $cur = Test-PkvLegRecordCurrent (Read-PkvLegRecord $m) (New-PkvLegRecord -Leg $t.Leg -Rests $t.Rests @after @common)
        Check ($cur -eq (-not $t.Rerun)) "adding timing79: $($t.Leg) $(if ($t.Rerun) { 'reruns' } else { 'is skipped' })"
    }
    # The ratio is 7.4's alone: changing it reruns 7.4 and nothing else.
    $r74a = New-PkvLegRecord -Leg '7.4/C6' -Rests @('timing74') -Commissioned @('timing74') -Max74Ratio '2.0' @common
    $r74b = New-PkvLegRecord -Leg '7.4/C6' -Rests @('timing74') -Commissioned @('timing74') -Max74Ratio '1.5' @common
    Check (-not (Test-PkvLegRecordCurrent $r74a $r74b)) 'a changed -Max74Ratio reruns 7.4'
    $r101a = New-PkvLegRecord -Leg '10.1/C6' -Rests @('oracle') -Commissioned @('oracle', 'timing74') -Max74Ratio '2.0' @common
    $r101b = New-PkvLegRecord -Leg '10.1/C6' -Rests @('oracle') -Commissioned @('oracle', 'timing74') -Max74Ratio '1.5' @common
    Check (Test-PkvLegRecordCurrent $r101a $r101b) 'a changed -Max74Ratio does not rerun a correctness leg'
    Check ($r101a.Commissioned -eq 'oracle' -and $r101a.Max74Ratio -eq '') 'a correctness leg records only its own ids and no ratio'
    # Dropping an instrument a leg rests on reruns it.
    Check (-not (Test-PkvLegRecordCurrent $r101a (New-PkvLegRecord -Leg '10.1/C6' -Rests @('oracle') -Commissioned @('timing74') @common))) 'dropping oracle reruns 10.1'

    # A pre-provenance marker ("GREEN exit=0 ...") is no record: rerun, and quarantined.
    Set-Content -LiteralPath $marker -Encoding utf8 -Value 'GREEN exit=0 1 cells, 0 red; 9 checks, 0 failures (12 s) 2026-07-01T00:00:00'
    Check ($null -eq (Read-PkvLegRecord $marker)) 'an old-format marker reads as no record'
    Check (-not (Test-PkvLegRecordCurrent (Read-PkvLegRecord $marker) $run2)) 'an old-format marker never skips'
    Check ((Get-PkvLegStanding -Rests @('timing79') -Record $null) -eq 'quarantined (no provenance record)') 'no record: quarantined'

    # ---- 7.4: commissioned only with a ratio ------------------------------------------------------
    $r74 = New-PkvLegRecord -Leg '7.4/C6' -Rests @('timing74') -Commissioned @('timing74') @common
    Check ((Get-PkvRecordEnv $r74).Count -eq 0) '7.4 without -Max74Ratio sets nothing'
    Check ((Get-PkvLegStanding -Rests @('timing74') -Record $r74) -eq 'quarantined (ratio)') '7.4 without -Max74Ratio stays quarantined'
    $r74 = New-PkvLegRecord -Leg '7.4/C6' -Rests @('timing74') -Commissioned @('timing74') -Max74Ratio '2.0' @common
    Check ((Get-PkvRecordEnv $r74)['SUPERSLM_PAGED_KV_74_MAX_RATIO'] -eq '2.0') '7.4 with the ratio sets SUPERSLM_PAGED_KV_74_MAX_RATIO'
    Check (-not (Get-PkvRecordEnv $r74).Contains('SUPERSLM_PAGED_KV_TIMING_COMMISSIONED')) 'timing74 does not set the 7.9 variable'
    Check ((Get-PkvLegStanding -Rests @('timing74') -Record $r74) -eq 'standing') '7.4 with timing74 and a ratio stands'
    $r74only79 = New-PkvLegRecord -Leg '7.4/C6' -Rests @('timing74') -Commissioned @('timing79') -Max74Ratio '2.0' @common
    Check ((Get-PkvRecordEnv $r74only79).Count -eq 0) 'timing79 alone sets nothing for 7.4'
    Check ((Get-PkvLegStanding -Rests @('timing74') -Record $r74only79) -eq 'quarantined (timing74)') 'timing79 alone leaves 7.4 quarantined'
    $r79only74 = New-PkvLegRecord -Leg '7.9/C6' -Rests @('timing79') -Commissioned @('timing74') -Max74Ratio '2.0' @common
    Check ((Get-PkvRecordEnv $r79only74).Count -eq 0) 'timing74 alone sets nothing for 7.9'
    Check ((Get-PkvLegStanding -Rests @('timing79') -Record $r79only74) -eq 'quarantined (timing79)') 'timing74 alone leaves 7.9 quarantined'

    # ---- S1: lifecycle is never commissioned -------------------------------------------------------
    $all = Get-PkvCommissionedIds @('oracle,admission', 'timing74', 'timing79')
    Check (($all -join ',') -eq 'admission,oracle,timing74,timing79') 'comma-joined ids split and sort'
    foreach ($leg in 'lifecycle-v1.11.0', 'lifecycle/C6') {
        $rl = New-PkvLegRecord -Leg $leg -Rests @('lifecycle') -Commissioned $all -Max74Ratio '2.0' @common
        Check ((Get-PkvRecordEnv $rl).Count -eq 0) "$leg gets no assertion variable with everything commissioned"
        Check ((Get-PkvLegStanding -Rests @('lifecycle') -Record $rl) -like 'quarantined*') "$leg stays quarantined with everything commissioned"
    }
    # A hand-edited record naming lifecycle still does not stand.
    $forged = New-PkvLegRecord -Leg 'lifecycle/C6' -Rests @('lifecycle') -Commissioned @() @common
    $forged.Commissioned = 'lifecycle'
    Check ((Get-PkvLegStanding -Rests @('lifecycle') -Record $forged) -like 'quarantined*') 'a record naming lifecycle does not stand'
    Throws { Get-PkvCommissionedIds @('lifecycle') } 'no registry entry' '-Commissioned lifecycle is rejected'
    Throws { Get-PkvCommissionedIds @('oracle,lifecycle') } 'no registry entry' '-Commissioned oracle,lifecycle is rejected'
    Throws { Get-PkvCommissionedIds @('timing') } "not 'timing'" '-Commissioned timing (the old shared id) is rejected'
    Throws { Get-PkvCommissionedIds @('bogus') } "not 'bogus'" 'an unknown id is rejected'
    Check ((Get-PkvCommissionedIds @()).Count -eq 0) 'no ids: empty set'
} finally {
    Remove-Item -LiteralPath $tmp -Recurse -Force -ErrorAction SilentlyContinue
}
if ($script:failures -gt 0) { Write-Output "$($script:failures) failure(s)"; exit 1 }
Write-Output 'all passed'
