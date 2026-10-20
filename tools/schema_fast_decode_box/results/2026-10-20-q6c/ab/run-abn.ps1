# N-arm, same-session timing of B2's jump-forward-on extraction at one thread. Each arm is a SuperSLM commit with its own
# worktree and a FRESH build folder (configured and built here; an existing folder is refused, so no arm is an
# incremental build), except an arm given with an existing bench path. One process per arm per round; round 0 is a
# discarded warm-up; the arm order rotates by one per round. Writes per-run JSON under <Out>\parts and a summary JSON.
# Run under Windows PowerShell 5.1.  -Arms 'label=commit-or-exe;label=...'  -Out <dir>
param([Parameter(Mandatory)][string]$Arms, [Parameter(Mandatory)][string]$Out, [int]$Rounds = 15)
$ErrorActionPreference = 'Stop'
$SlmRepo = 'D:\SuperSLM'
$model = 'D:\_artifacts\SuperSLM\aex\qwen2.5-0.5b-instruct-cap4096-aex-pr.sslm'
$parts = "$Out\parts"
$status = "$Out\status.txt"
New-Item -ItemType Directory -Force -Path $Out, $parts | Out-Null
function Say($t) { for ($i = 0; $i -lt 20; $i++) { try { Add-Content -LiteralPath $status -Value "[$(Get-Date -Format o)] $t" -ErrorAction Stop; return } catch { Start-Sleep -Milliseconds 250 } } }
function Stat([double[]]$v) {
    $s = $v | Sort-Object
    $n = $s.Count
    $med = if ($n % 2) { $s[[int](($n - 1) / 2)] } else { ($s[$n / 2 - 1] + $s[$n / 2]) / 2 }
    $lo = if ($n -eq 15) { $s[3] } else { $s[[int][math]::Floor(($n - 1) * 0.25)] }
    $hi = if ($n -eq 15) { $s[11] } else { $s[[int][math]::Ceiling(($n - 1) * 0.75)] }
    return [ordered]@{ n = $n; median = [math]::Round($med, 4); sign_lo = [math]::Round($lo, 4); sign_hi = [math]::Round($hi, 4) }
}
Say "start pid $PID arms '$Arms'"
try {
    . "$SlmRepo\.worktrees\schema-fast-decode\tests\t2130-g5-red-suite\resolve_toolchain.ps1"
    Enter-SuperSlmVsDevShell | Out-Null
    $armList = @()
    foreach ($spec in ($Arms -split ';')) {
        $label, $what = $spec -split '=', 2
        if ($what -like '*.exe') {
            $armList += [ordered]@{ label = $label; bench = $what; commit = 'given'; build = 'existing' }
            continue
        }
        $commit = (git -C $SlmRepo rev-parse --verify "$what^{commit}").Trim()
        $wt = "$SlmRepo\.worktrees\sfd-arm-$($commit.Substring(0,8))"
        if (-not (Test-Path $wt)) { git -C $SlmRepo worktree add -q --detach $wt $commit; if ($LASTEXITCODE) { throw "worktree add $commit" } }
        if ((git -C $wt rev-parse HEAD).Trim() -ne $commit) { throw "$wt is not at $commit" }
        $bd = "$wt\build-fresh"
        if (Test-Path $bd) { throw "$bd exists: this script only measures fresh builds" }
        & cmake -S $wt -B $bd -G Ninja -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF -DSUPERSLM_BUILD_GPU=OFF *> "$Out\$label-configure.log"
        if ($LASTEXITCODE -ne 0) { throw "$label configure failed" }
        & cmake --build $bd --target superslm sslm_schema_fast_decode_bench *> "$Out\$label-build.log"
        if ($LASTEXITCODE -ne 0) { throw "$label build failed" }
        $armList += [ordered]@{ label = $label; bench = "$bd\sslm_schema_fast_decode_bench.exe"; commit = $commit; build = 'fresh' }
    }
    foreach ($a in $armList) { $a.sha256 = (Get-FileHash $a.bench).Hash.ToLower(); Say "arm $($a.label): $($a.commit) ($($a.build)) bench $($a.sha256)" }
    Say "model sha256 $((Get-FileHash $model).Hash.ToLower())"
    $args1 = @($model, 'extract', 'potion_shop_order', '--example', '--max-new', '64', '--pair', 'on', '--pairs', '1', '--warmup', '1')
    $gen = @{}; $digest = @{}; foreach ($a in $armList) { $gen[$a.label] = @(); $digest[$a.label] = @() }
    $orders = @()
    for ($r = 0; $r -le $Rounds; ++$r) {
        $order = @(); for ($q = 0; $q -lt $armList.Count; ++$q) { $order += $armList[($q + $r) % $armList.Count] }
        if ($r -gt 0) { $orders += (($order | ForEach-Object { $_.label }) -join ',') }
        foreach ($a in $order) {
            $json = "$parts\r$r-$($a.label).json"
            $p = Start-Process -FilePath $a.bench -ArgumentList ($args1 + @('--json', $json)) -RedirectStandardOutput "$parts\r$r-$($a.label).log" `
                -RedirectStandardError "$parts\r$r-$($a.label).log.err" -NoNewWindow -PassThru
            $null = $p.Handle
            if (-not $p.WaitForExit(5 * 60000)) { $p.Kill(); throw "r$r $($a.label) passed 5 min" }
            if ($p.ExitCode -ne 0) { throw "r$r $($a.label) exited $($p.ExitCode)" }
            if ($r -eq 0) { continue }
            $j = Get-Content $json -Raw | ConvertFrom-Json
            $gen[$a.label] += [double]$j.arms[0].gen_ms.median
            $digest[$a.label] += $j.arms[0].digest
        }
    }
    $ref = $armList[0].label
    $summary = [ordered]@{ rounds = $Rounds; reference = $ref; orders = $orders; arms = @() }
    foreach ($a in $armList) {
        $ratios = @(); for ($i = 0; $i -lt $Rounds; ++$i) { $ratios += $gen[$ref][$i] / $gen[$a.label][$i] }
        $summary.arms += [ordered]@{ label = $a.label; commit = $a.commit; build = $a.build; bench_sha256 = $a.sha256
            gen_ms = (Stat $gen[$a.label]); ratio_ref_over_arm = (Stat $ratios); digests = @($digest[$a.label] | Sort-Object -Unique) }
        Say "$($a.label): gen $((Stat $gen[$a.label]).median) [$((Stat $gen[$a.label]).sign_lo), $((Stat $gen[$a.label]).sign_hi)] ratio $ref/$($a.label) $((Stat $ratios).median) [$((Stat $ratios).sign_lo), $((Stat $ratios).sign_hi)] digests $(@($digest[$a.label] | Sort-Object -Unique) -join ',')"
    }
    ($summary | ConvertTo-Json -Depth 6) | Set-Content "$Out\summary.json"
    Say 'end PASS'
}
catch { Say "end FAIL $($_.Exception.Message) (line $($_.InvocationInfo.ScriptLineNumber): $($_.InvocationInfo.Line.Trim()))" }
