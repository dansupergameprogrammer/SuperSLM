# Same-session A/B of jump-forward on at one thread: the bench built at 79e5b92 (A) against the bench built at 36fe4b3
# (B), one process per arm per pair (B2's main-vs-M1 shape), pair 0 a discarded warm-up, measured pair k running A first
# when k is even, pooled by the bench's own `pool` mode. Run under Windows PowerShell 5.1.
$ErrorActionPreference = 'Stop'
$aWt = 'D:\SuperSLM\.worktrees\sfd-ab-79e5b92'
$bWt = 'D:\SuperSLM\.worktrees\schema-fast-decode'
$aBench = "$aWt\build-box\sslm_schema_fast_decode_bench.exe"
$bBench = "$bWt\build-box\sslm_schema_fast_decode_bench.exe"
$model = 'D:\_artifacts\SuperSLM\aex\qwen2.5-0.5b-instruct-cap4096-aex-pr.sslm'
$out = 'D:\_scratch\schema-fast-decode\out-q6c\ab'
$parts = "$out\parts"
$status = "$out\status.txt"
New-Item -ItemType Directory -Force -Path $out, $parts | Out-Null
function Say($t) { for ($i = 0; $i -lt 20; $i++) { try { Add-Content -LiteralPath $status -Value "[$(Get-Date -Format o)] $t" -ErrorAction Stop; return } catch { Start-Sleep -Milliseconds 250 } } }
function Bench([string]$Exe, [string]$Part, [string[]]$A) {
    $json = "$parts\$Part.json"
    $p = Start-Process -FilePath $Exe -ArgumentList ($A + @('--json', $json)) -RedirectStandardOutput "$parts\$Part.log" `
        -RedirectStandardError "$parts\$Part.log.err" -NoNewWindow -PassThru
    $null = $p.Handle
    if (-not $p.WaitForExit(5 * 60000)) { $p.Kill(); throw "$Part passed 5 min; killed" }
    if ($p.ExitCode -ne 0) { throw "$Part exited $($p.ExitCode)" }
    return $json
}
Say "start pid $PID"
try {
    Say "A $aWt at $((git -C $aWt rev-parse HEAD).Trim()); B $bWt at $((git -C $bWt rev-parse HEAD).Trim())"
    . "$bWt\tests\t2130-g5-red-suite\resolve_toolchain.ps1"
    Enter-SuperSlmVsDevShell | Out-Null
    if (-not (Test-Path "$aWt\build-box\CMakeCache.txt")) {
        & cmake -S $aWt -B "$aWt\build-box" -G Ninja -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF -DSUPERSLM_BUILD_GPU=OFF *> "$out\a-configure.log"
        if ($LASTEXITCODE -ne 0) { throw 'A configure failed (a-configure.log)' }
    }
    & cmake --build "$aWt\build-box" --target superslm sslm_schema_fast_decode_bench *> "$out\a-build.log"
    if ($LASTEXITCODE -ne 0) { throw 'A build failed (a-build.log)' }
    & cmake --build "$bWt\build-box" --target superslm sslm_schema_fast_decode_bench *> "$out\b-build.log"
    if ($LASTEXITCODE -ne 0) { throw 'B build failed (b-build.log)' }
    Say "A bench sha256 $((Get-FileHash $aBench).Hash.ToLower()); B bench sha256 $((Get-FileHash $bBench).Hash.ToLower())"
    Say "model sha256 $((Get-FileHash $model).Hash.ToLower())"
    $args1 = @($model, 'extract', 'potion_shop_order', '--example', '--max-new', '64', '--pair', 'on', '--pairs', '1', '--warmup', '1')
    $files = @(); $first = ''
    for ($p = 0; $p -le 15; ++$p) {
        $k = $p - 1
        $aFirst = ($k -lt 0) -or ($k % 2 -eq 0)
        if ($k -ge 0) { if ($aFirst) { $first += 'A' } else { $first += 'B' } }
        $order = if ($aFirst) { @('a', 'b') } else { @('b', 'a') }
        foreach ($side in $order) {
            $exe = if ($side -eq 'a') { $aBench } else { $bBench }
            $null = Bench $exe "p$p$side" $args1
        }
        $files += "$parts\p${p}a.json"; $files += "$parts\p${p}b.json"
    }
    $null = Bench $bBench 'pool' (@('-', 'pool', '--warmup', '1', '--labels', '79e5b92,36fe4b3', '--first', $first) + $files)
    Copy-Item "$parts\pool.json" "$out\ab-on-t1.json" -Force
    Say 'end PASS'
}
catch { Say "end FAIL $($_.Exception.Message)" }
