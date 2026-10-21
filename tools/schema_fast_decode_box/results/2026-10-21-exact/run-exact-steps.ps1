# Exact skip-ahead box steps (plan rev 1.3.4, B7) on f03a5358 plus the bench's draft sources (build-exact), the aex-pr
# model, the frozen D2 table. Resumable: each step writes <step>.done and is skipped when it exists.
#   b7b-t1, b7b-t4  exactness: the 40 population prompts and B2's prompt, arms off, on:exact:learned, on:exact (NULL draft)
#                   and on:exact:wrong, one process per prompt; every exact arm must reproduce off's stream and text
#                   byte for byte. Any difference stops the run.
#   b7c, b7d        speed on B2's prompt, rows off, on, on:exact, on:exact:learned; one process per row per round,
#                   15 rounds after a warm-up, the row order rotated; one thread, then 4 tasks.
#   b7e             speed over B4b's 20 prompts, one thread: off against on:exact:learned, 5 interleaved pairs each
#                   after one warm-up pair, in one process per prompt.
# Run under Windows PowerShell 5.1.
param([string[]]$Steps = @('b7b-t1', 'b7b-t4', 'b7c', 'b7d', 'b7e'))
$ErrorActionPreference = 'Stop'
$out = 'D:\_scratch\decode-j1x-ab\exact'
$bench = 'D:\SuperSLM\.worktrees\sfd-results\build-exact\sslm_schema_fast_decode_bench.exe'
$model = 'D:\_artifacts\SuperSLM\aex\qwen2.5-0.5b-instruct-cap4096-aex-pr.sslm'
$table = "$out\d2-table.txt"
$pop = 'D:\SuperSLM\.worktrees\schema-fast-decode\tools\schema_fast_decode_box\box\potion-population-v1.jsonl'
$status = "$out\status.txt"
function Say($t) { for ($i = 0; $i -lt 20; $i++) { try { Add-Content -LiteralPath $status -Value "[$(Get-Date -Format o)] $t" -ErrorAction Stop; return } catch { Start-Sleep -Milliseconds 250 } } }
function Stat([double[]]$v) {
    $s = $v | Sort-Object; $n = $s.Count
    $med = if ($n % 2) { $s[[int](($n - 1) / 2)] } else { ($s[$n / 2 - 1] + $s[$n / 2]) / 2 }
    $lo = if ($n -eq 15) { $s[3] } else { $s[[int][math]::Floor(($n - 1) * 0.25)] }
    $hi = if ($n -eq 15) { $s[11] } else { $s[[int][math]::Ceiling(($n - 1) * 0.75)] }
    return [ordered]@{ n = $n; median = [math]::Round($med, 4); sign_lo = [math]::Round($lo, 4); sign_hi = [math]::Round($hi, 4) }
}
function Bench([string[]]$A, [string]$Json) {
    & $bench @A --json $Json *> "$Json.log"
    if ($LASTEXITCODE -ne 0) { throw "bench exited $LASTEXITCODE ($Json.log)" }
    return (Get-Content $Json -Raw | ConvertFrom-Json)
}
$prompts = @([ordered]@{ id = 'B2'; args = @('--example') })
foreach ($line in Get-Content $pop) {
    $p = $line | ConvertFrom-Json
    $prompts += [ordered]@{ id = $p.id; args = @('--example', '--utterance', $p.customer_line) }
}
Say "start pid $PID; bench $((Get-FileHash $bench).Hash.ToLower()); table $((Get-FileHash $table).Hash.ToLower()); model $((Get-FileHash $model).Hash.ToLower()); population $((Get-FileHash $pop).Hash.ToLower()); prompts $($prompts.Count)"
try {
    $busy = @(Get-Process -Name 'UnrealEditor', 'UnrealEditor-Cmd', 'UnrealBuildTool', 'cl', 'link' -ErrorAction SilentlyContinue)
    if ($busy.Count -gt 0 -or (Test-Path 'D:\_ssu_build_lock')) { throw 'box not clear' }
    foreach ($step in $Steps) {
        if (Test-Path "$out\$step.done") { Say "$step already done: skipped"; continue }
        $dir = "$out\$step"; New-Item -ItemType Directory -Force -Path $dir | Out-Null
        Say "$step start"
        if ($step -like 'b7b-*') {
            $tasks = if ($step -eq 'b7b-t4') { @('--tasks', '4') } else { @() }
            $rows = @(); $diff = 0
            foreach ($pr in $prompts) {
                $j = Bench (@($model, 'extract', 'potion_shop_order') + $pr.args + @('--max-new', '64', '--pair', 'off,on:exact:learned,on:exact,on:exact:wrong', '--pairs', '1', '--warmup', '0', '--draft-table', $table) + $tasks) "$dir\$($pr.id).json"
                $off = $j.arms | Where-Object label -eq 'off'
                $offStream = ($off.runs[0].stream -join ','); $offText = $off.runs[0].text
                $row = [ordered]@{ id = $pr.id; off_digest = $off.runs[0].digest; off_tokens = $off.runs[0].tokens; off_stop = $off.runs[0].stop; arms = @() }
                foreach ($a in ($j.arms | Where-Object label -ne 'off')) {
                    $same = (($a.runs[0].stream -join ',') -eq $offStream) -and ($a.runs[0].text -ceq $offText)
                    if (-not $same) { $diff++ }
                    $row.arms += [ordered]@{ label = $a.label; digest = $a.runs[0].digest; identical_to_off = $same; spans = @($a.runs[0].spans) }
                }
                $rows += $row
                if ($diff -gt 0) { break }
            }
            ([ordered]@{ step = $step; prompts = $rows.Count; differences = $diff; rows = $rows } | ConvertTo-Json -Depth 6) | Set-Content "$dir\summary.json"
            Say "${step}: $($rows.Count) prompts, $diff arm(s) not identical to off"
            if ($diff -gt 0) { Say "${step} STOP: an exact arm differs from off"; throw 'STOP: exactness' }
        } elseif ($step -eq 'b7c' -or $step -eq 'b7d') {
            $tasks = if ($step -eq 'b7d') { @('--tasks', '4') } else { @() }
            $rowsList = @('off', 'on', 'on:exact', 'on:exact:learned')
            $val = @{}; $dig = @{}; foreach ($r in $rowsList) { $val[$r] = @(); $dig[$r] = @() }
            for ($round = 0; $round -le 15; ++$round) {
                for ($k = 0; $k -lt $rowsList.Count; ++$k) {
                    $r = $rowsList[($k + $round) % $rowsList.Count]
                    $j = Bench (@($model, 'extract', 'potion_shop_order', '--example', '--max-new', '64', '--pair', $r, '--pairs', '1', '--warmup', '1', '--draft-table', $table) + $tasks) "$dir\r$round-$($r -replace ':', '_').json"
                    if ($round -eq 0) { continue }
                    $val[$r] += [double]$j.arms[0].gen_ms.median; $dig[$r] += $j.arms[0].digest
                }
            }
            $sum = [ordered]@{ step = $step; rounds = 15; rows = @() }
            foreach ($r in $rowsList) {
                $ratios = @(); for ($i = 0; $i -lt 15; ++$i) { $ratios += $val['off'][$i] / $val[$r][$i] }
                $sum.rows += [ordered]@{ row = $r; gen_ms = (Stat $val[$r]); off_over_row = (Stat $ratios); digests = @($dig[$r] | Sort-Object -Unique) }
                Say "$step $r : $((Stat $val[$r]).median) ms | off/row $((Stat $ratios).median) [$((Stat $ratios).sign_lo), $((Stat $ratios).sign_hi)] | digests $(@($dig[$r] | Sort-Object -Unique) -join ',')"
            }
            ($sum | ConvertTo-Json -Depth 6) | Set-Content "$dir\summary.json"
        } elseif ($step -eq 'b7e') {
            $b4b = $prompts | Where-Object { $_.id -like 'B4b-*' }
            $rows = @(); $offAll = 0.0; $exAll = 0.0
            foreach ($pr in $b4b) {
                $j = Bench (@($model, 'extract', 'potion_shop_order') + $pr.args + @('--max-new', '64', '--pair', 'off,on:exact:learned', '--pairs', '5', '--warmup', '1', '--draft-table', $table)) "$dir\$($pr.id).json"
                $o = $j.arms | Where-Object label -eq 'off'; $x = $j.arms | Where-Object label -eq 'on:exact:learned'
                $offAll += [double]$o.gen_ms.median; $exAll += [double]$x.gen_ms.median
                $rows += [ordered]@{ id = $pr.id; off_ms = $o.gen_ms.median; exact_ms = $x.gen_ms.median; ratio = $j.ratio.median; off_digest = $o.digest; exact_digest = $x.digest; same = $o.digest -eq $x.digest }
            }
            $ratios = @($rows | ForEach-Object { [double]$_.ratio })
            $sum = [ordered]@{ step = 'b7e'; prompts = $rows.Count; per_prompt_ratio = (Stat $ratios); total_off_ms = [math]::Round($offAll, 2); total_exact_ms = [math]::Round($exAll, 2); total_ratio = [math]::Round($offAll / $exAll, 4); all_same = -not ($rows | Where-Object { -not $_.same }); rows = $rows }
            ($sum | ConvertTo-Json -Depth 6) | Set-Content "$dir\summary.json"
            Say "b7e: per-prompt off/exact median $((Stat $ratios).median) [$((Stat $ratios).sign_lo), $((Stat $ratios).sign_hi)]; total $($sum.total_off_ms) / $($sum.total_exact_ms) ms = $($sum.total_ratio); all digests same: $($sum.all_same)"
        }
        Set-Content "$out\$step.done" (Get-Date -Format o)
        Say "$step done"
    }
    Say 'end PASS'
}
catch { Say "end FAIL $($_.Exception.Message) (line $($_.InvocationInfo.ScriptLineNumber))" }
