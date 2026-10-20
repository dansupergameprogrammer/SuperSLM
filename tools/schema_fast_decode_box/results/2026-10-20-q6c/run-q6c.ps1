# Schema fast-decode §16 Q6 C re-time at the new tip: rebuild the bench, then the span readings at 4 and 2 tasks and
# the off/on digest check at 1 thread and 4 tasks. One bench run per part, 45 s rest between, stop on the first failure.
# Run under Windows PowerShell 5.1 (the toolchain helper's shell).
$ErrorActionPreference = 'Stop'
$wt = 'D:\SuperSLM\.worktrees\schema-fast-decode'
$build = "$wt\build-box"
$bench = "$build\sslm_schema_fast_decode_bench.exe"
$model = 'D:\_artifacts\SuperSLM\aex\qwen2.5-0.5b-instruct-cap4096-aex-pr.sslm'
$out = 'D:\_scratch\schema-fast-decode\out-q6c'
$logs = "$out\logs"
New-Item -ItemType Directory -Force -Path $out, $logs | Out-Null
$status = "$out\status.txt"
function Say($t) {
    # A reader holding the file open must not kill the run: retry, then give up on the line, never throw.
    for ($i = 0; $i -lt 20; $i++) {
        try { Add-Content -LiteralPath $status -Value "[$(Get-Date -Format o)] $t" -ErrorAction Stop; return } catch { Start-Sleep -Milliseconds 250 }
    }
}
Say "start pid $PID"
try {
    $head = (git -C $wt rev-parse HEAD).Trim()
    Say "worktree $wt at $head"
    . "$wt\tests\t2130-g5-red-suite\resolve_toolchain.ps1"
    Enter-SuperSlmVsDevShell | Out-Null
    & cmake --build $build --target superslm sslm_schema_fast_decode_bench *> "$logs\build.log"
    if ($LASTEXITCODE -ne 0) { throw "build failed (logs\build.log)" }
    Say "build ok; bench sha256 $((Get-FileHash $bench).Hash.ToLower()) lib sha256 $((Get-FileHash "$build\superslm.lib").Hash.ToLower())"
    Say "model sha256 $((Get-FileHash $model).Hash.ToLower())"
    $n = @('--pairs', '15', '--warmup', '1')
    $ex = @('extract', 'potion_shop_order', '--example', '--max-new', '64')
    $parts = [ordered]@{
        'pass-t4'   = @($model, 'pass', '--example', '--sizes', '1,2,3,4,5,6,7,8,9') + $n + @('--tasks', '4')
        'spans-t4'  = @($model, 'spans', 'potion_shop_order', '--example', '--pairs', '15', '--tasks', '4')
        'pass-t2'   = @($model, 'pass', '--example', '--sizes', '1,2,3,4,5,6,7,8,9') + $n + @('--tasks', '2')
        'spans-t2'  = @($model, 'spans', 'potion_shop_order', '--example', '--pairs', '15', '--tasks', '2')
        'jf-t1'     = @($model) + $ex + @('--pair', 'off,on') + $n
        'jf-t4'     = @($model) + $ex + @('--pair', 'off,on') + $n + @('--tasks', '4')
    }
    $first = $true
    foreach ($k in $parts.Keys) {
        if (-not $first) { Start-Sleep -Seconds 45 }
        $first = $false
        $json = "$out\$k.json"
        if (Test-Path -LiteralPath $json) { Say "$k already done ($json): skipped"; $first = $true; continue }
        Say "$k start: $($parts[$k][1..($parts[$k].Count-1)] -join ' ')"
        $p = Start-Process -FilePath $bench -ArgumentList ($parts[$k] + @('--json', $json)) -RedirectStandardOutput "$logs\$k.log" `
            -RedirectStandardError "$logs\$k.log.err" -NoNewWindow -PassThru
        $null = $p.Handle
        if (-not $p.WaitForExit(15 * 60000)) { $p.Kill(); throw "$k passed 15 min; killed" }
        if ($p.ExitCode -ne 0) { throw "$k exited $($p.ExitCode) (logs\$k.log.err)" }
        Say "$k done"
    }
    Say 'end PASS'
}
catch { Say "end FAIL $($_.Exception.Message)" }
