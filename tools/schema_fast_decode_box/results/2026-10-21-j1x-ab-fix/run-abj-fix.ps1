# One-thread A/B of the J1-X fix (B = f4f69efb) against the Q6 C fix (A = 2af41659), MSVC Release, the aex-pr model.
# Rows: jump-forward off, jump-forward on (canonical), a plain prefill of 7 and 16 tokens at the example prompt's depth,
# and on B only jump-forward on in exact mode. One process per arm per row per round; round 0 is a discarded warm-up;
# the arm order and the row order rotate per round. A's bench is the fresh 2af41659 build from the 2026-10-20 ab-fix run;
# B is built fresh here, its bench carrying the `on:exact` arm. Run under Windows PowerShell 5.1.
param([int]$Rounds = 15)
$ErrorActionPreference = 'Stop'
$SlmRepo = 'D:\SuperSLM'
$out = 'D:\_scratch\decode-j1x-ab\fix'
$parts = "$out\parts"
$status = "$out\status.txt"
$model = 'D:\_artifacts\SuperSLM\aex\qwen2.5-0.5b-instruct-cap4096-aex-pr.sslm'
New-Item -ItemType Directory -Force -Path $parts | Out-Null
function Say($t) { for ($i = 0; $i -lt 20; $i++) { try { Add-Content -LiteralPath $status -Value "[$(Get-Date -Format o)] $t" -ErrorAction Stop; return } catch { Start-Sleep -Milliseconds 250 } } }
function Stat([double[]]$v) {
    $s = $v | Sort-Object; $n = $s.Count
    $med = if ($n % 2) { $s[[int](($n - 1) / 2)] } else { ($s[$n / 2 - 1] + $s[$n / 2]) / 2 }
    $lo = if ($n -eq 15) { $s[3] } else { $s[[int][math]::Floor(($n - 1) * 0.25)] }
    $hi = if ($n -eq 15) { $s[11] } else { $s[[int][math]::Ceiling(($n - 1) * 0.75)] }
    return [ordered]@{ n = $n; median = [math]::Round($med, 4); sign_lo = [math]::Round($lo, 4); sign_hi = [math]::Round($hi, 4) }
}
Say "start pid $PID"
try {
    $busy = @(Get-Process -Name 'UnrealEditor', 'UnrealEditor-Cmd', 'UnrealBuildTool', 'cl', 'link' -ErrorAction SilentlyContinue | ForEach-Object { "$($_.ProcessName) ($($_.Id))" })
    if ($busy.Count -gt 0) { throw "box not clear: $($busy -join ', ')" }
    if (Test-Path 'D:\_ssu_build_lock') { throw 'box not clear: D:\_ssu_build_lock is held' }
    Say "box clear; load $((Get-CimInstance Win32_Processor | Measure-Object LoadPercentage -Average).Average) %"
    . "$SlmRepo\.worktrees\schema-fast-decode\tests\t2130-g5-red-suite\resolve_toolchain.ps1"
    Enter-SuperSlmVsDevShell | Out-Null
    $aWt = "$SlmRepo\.worktrees\sfd-arm-2af41659"; $aBench = "$aWt\build-fresh\sslm_schema_fast_decode_bench.exe"
    $bWt = "$SlmRepo\.worktrees\sfd-arm-f4f69efb"; $bBuild = "$bWt\build-fresh"; $bBench = "$bBuild\sslm_schema_fast_decode_bench.exe"
    if ((git -C $aWt rev-parse HEAD).Trim() -ne '2af41659fc090368e356af383e67837df04e5d21') { throw "$aWt is not at 2af41659" }
    if ((git -C $bWt rev-parse HEAD).Trim() -ne 'f4f69efb3992d2f927d54d3b7ebf8dd13877cbcd') { throw "$bWt is not at f4f69efb" }
    if (-not (Test-Path $bBench)) {
        if (Test-Path $bBuild) { throw "$bBuild exists without a bench: inspect it" }
        & cmake -S $bWt -B $bBuild -G Ninja -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF -DSUPERSLM_BUILD_GPU=OFF *> "$out\b-configure.log"
        if ($LASTEXITCODE -ne 0) { throw 'B configure failed (b-configure.log)' }
        & cmake --build $bBuild --target superslm sslm_schema_fast_decode_bench *> "$out\b-build.log"
        if ($LASTEXITCODE -ne 0) { throw 'B build failed (b-build.log)' }
    }
    Say "B bench diff against f4f69efb (expected none): $((git -C $bWt diff --stat -- tools/schema_fast_decode_bench.cpp) -join ' ')"
    $arms = @(
        [ordered]@{ label = 'A-2af41659'; bench = $aBench; rows = @('off', 'on', 'prefill') },
        [ordered]@{ label = 'B-f4f69efb'; bench = $bBench; rows = @('off', 'on', 'prefill', 'exact') })
    foreach ($a in $arms) { $a.sha256 = (Get-FileHash $a.bench).Hash.ToLower(); Say "arm $($a.label) bench $($a.sha256)" }
    Say "model sha256 $((Get-FileHash $model).Hash.ToLower())"
    $ex = @($model, 'extract', 'potion_shop_order', '--example', '--max-new', '64', '--pairs', '1', '--warmup', '1')
    $rowArgs = @{
        off     = $ex + @('--pair', 'off')
        on      = $ex + @('--pair', 'on')
        exact   = $ex + @('--pair', 'on:exact')
        prefill = @($model, 'pass', '--example', '--sizes', '7,16', '--no-one-row', '--pairs', '1', '--warmup', '1')
    }
    $val = @{}; $dig = @{}
    for ($r = 0; $r -le $Rounds; ++$r) {
        for ($q = 0; $q -lt $arms.Count; ++$q) {
            $a = $arms[($q + $r) % $arms.Count]
            $rows = $a.rows
            for ($k = 0; $k -lt $rows.Count; ++$k) {
                $row = $rows[($k + $r) % $rows.Count]
                $json = "$parts\r$r-$($a.label)-$row.json"
                $p = Start-Process -FilePath $a.bench -ArgumentList ($rowArgs[$row] + @('--json', $json)) -RedirectStandardOutput "$json.log" `
                    -RedirectStandardError "$json.err" -NoNewWindow -PassThru
                $null = $p.Handle
                if (-not $p.WaitForExit(5 * 60000)) { $p.Kill(); throw "r$r $($a.label) $row passed 5 min" }
                if ($p.ExitCode -ne 0) { throw "r$r $($a.label) $row exited $($p.ExitCode)" }
                if ($r -eq 0) { continue }
                $j = Get-Content $json -Raw | ConvertFrom-Json
                if ($row -eq 'prefill') {
                    foreach ($s in $j.sizes) {
                        $key = "$($a.label)|prefill$($s.M)"
                        if (-not $val[$key]) { $val[$key] = @() }; $val[$key] += [double]$s.batched_ms.median
                    }
                } else {
                    $key = "$($a.label)|$row"
                    if (-not $val[$key]) { $val[$key] = @(); $dig[$key] = @() }
                    $val[$key] += [double]$j.arms[0].gen_ms.median; $dig[$key] += $j.arms[0].digest
                }
            }
        }
        Say "round $r done"
    }
    $summary = [ordered]@{ rounds = $Rounds; model_sha256 = (Get-FileHash $model).Hash.ToLower(); arms = @(); rows = @() }
    foreach ($a in $arms) { $summary.arms += [ordered]@{ label = $a.label; bench = $a.bench; bench_sha256 = $a.sha256 } }
    foreach ($row in 'off', 'on', 'prefill7', 'prefill16', 'exact') {
        $ka = "A-2af41659|$row"; $kb = "B-f4f69efb|$row"
        $entry = [ordered]@{ row = $row }
        if ($val[$ka]) { $entry.A_ms = (Stat $val[$ka]); if ($dig[$ka]) { $entry.A_digests = @($dig[$ka] | Sort-Object -Unique) } }
        if ($val[$kb]) { $entry.B_ms = (Stat $val[$kb]); if ($dig[$kb]) { $entry.B_digests = @($dig[$kb] | Sort-Object -Unique) } }
        if ($val[$ka] -and $val[$kb]) {
            $ratios = @(); for ($i = 0; $i -lt $Rounds; ++$i) { $ratios += $val[$kb][$i] / $val[$ka][$i] }
            $entry.B_over_A = (Stat $ratios)
        }
        $summary.rows += $entry
        $line = "$row :: A $($entry.A_ms.median) | B $($entry.B_ms.median)"
        if ($entry.B_over_A) { $line += " | B/A $($entry.B_over_A.median) [$($entry.B_over_A.sign_lo), $($entry.B_over_A.sign_hi)]" }
        if ($entry.A_digests -or $entry.B_digests) { $line += " | digests A [$($entry.A_digests -join ',')] B [$($entry.B_digests -join ',')]" }
        Say $line
    }
    ($summary | ConvertTo-Json -Depth 6) | Set-Content "$out\summary.json"
    Say 'end PASS'
}
catch { Say "end FAIL $($_.Exception.Message) (line $($_.InvocationInfo.ScriptLineNumber))" }
