# Engine 1.15.0 box checks on the release tree (claude/sfd-1p15): B7 exactness at 1 and 4 tasks, the MSVC GEMM pin
# placement, and a one-thread 7-token prefill A/B against the B7 build. The run sheet is README.md beside this file.
# Resumable: each step writes <Scratch>\<step>.done and is skipped when that file exists; delete it to re-run a step.
# Run under Windows PowerShell 5.1 from any directory:
#   powershell -ExecutionPolicy Bypass -File D:\_scratch\sfd-1p15\src\docs\release\1p15-box\run-1p15-box.ps1
# Steps: build, pinmap, b7b-t1, b7b-t4, abp (default: all, in that order). Exit 0 only when every step passed.
param(
    [string[]]$Steps = @('build', 'pinmap', 'b7b-t1', 'b7b-t4', 'abp'),
    [string]$Scratch = 'D:\_scratch\sfd-1p15',
    [string]$Model = 'D:\_artifacts\SuperSLM\aex\qwen2.5-0.5b-instruct-cap4096-aex-pr.sslm',
    # The A/B reference: B7's timed bench (f03a5358 plus the bench's draft sources, sha256 fcba88db...). When it is not
    # on disk, or its hash differs, aa0bc94c (the same sources, committed) is built fresh and used instead.
    [string]$RefBench = 'D:\SuperSLM\.worktrees\sfd-results\build-exact\sslm_schema_fast_decode_bench.exe',
    [int]$Rounds = 15
)
$ErrorActionPreference = 'Stop'
$src = Join-Path $Scratch 'src'
$bd = Join-Path $Scratch 'build-msvc'
$out = Join-Path $Scratch 'out'
$status = Join-Path $out 'status.txt'
$engineCommit = 'e82d7d40'   # the engine tree this sheet was written for; later branch commits touch docs/release only
New-Item -ItemType Directory -Force -Path $out | Out-Null
function Say($t) { $line = "[$(Get-Date -Format o)] $t"; Write-Output $line; Add-Content -LiteralPath $status -Value $line }
function Stat([double[]]$v) {
    $s = $v | Sort-Object; $n = $s.Count
    $med = if ($n % 2) { $s[[int](($n - 1) / 2)] } else { ($s[$n / 2 - 1] + $s[$n / 2]) / 2 }
    $lo = if ($n -eq 15) { $s[3] } else { $s[[int][math]::Floor(($n - 1) * 0.25)] }
    $hi = if ($n -eq 15) { $s[11] } else { $s[[int][math]::Ceiling(($n - 1) * 0.75)] }
    return [ordered]@{ n = $n; median = [math]::Round($med, 4); sign_lo = [math]::Round($lo, 4); sign_hi = [math]::Round($hi, 4) }
}
function Bench([string]$Exe, [string[]]$A, [string]$Json) {
    & $Exe @A --json $Json *> "$Json.log"
    if ($LASTEXITCODE -ne 0) { throw "bench exited $LASTEXITCODE ($Json.log)" }
    return (Get-Content $Json -Raw | ConvertFrom-Json)
}
function Sha([string]$p) { return (Get-FileHash -Algorithm SHA256 $p).Hash.ToLower() }

$failed = $false
try {
    # --- preconditions (every run) ----------------------------------------------------------------------------------
    if (-not (Test-Path (Join-Path $src '.git'))) { throw "$src is not a worktree: run piece P0 of the run sheet first" }
    $head = (git -C $src rev-parse HEAD).Trim()
    git -C $src diff --quiet $engineCommit HEAD -- src include CMakeLists.txt tools tests
    if ($LASTEXITCODE -ne 0) { throw "the engine tree at $head differs from $engineCommit outside docs/: this sheet does not apply to it" }
    if (@(git -C $src status --porcelain).Count -ne 0) { throw "$src has local changes" }
    $busy = @(Get-Process -Name 'UnrealEditor', 'UnrealEditor-Cmd', 'UnrealBuildTool', 'cl', 'link' -ErrorAction SilentlyContinue)
    if ($busy.Count -gt 0 -or (Test-Path 'D:\_ssu_build_lock')) { throw 'box not clear (an editor, a build or the build lock is present)' }
    if ((Sha $Model) -ne '9ccf7e378bbe47aa0ffd3810aa82875926cbe51af1bdfa08393c3fae657463da') { throw "model hash differs: $Model" }
    $pop = Join-Path $src 'tools\schema_fast_decode_box\frozen\potion-population-v1.jsonl'
    if ((Sha $pop) -ne 'd846b9e301bdd4369ff3fc51fca43af6c06de0acd97cb09489eaac107e5a67da') { throw "population hash differs: $pop" }
    # The frozen D2 table, with any CRLF a checkout added taken out, so its hash is the recorded one.
    $table = Join-Path $out 'd2-table.txt'
    $raw = [System.IO.File]::ReadAllText((Join-Path $src 'tools\schema_fast_decode_box\results\2026-10-21-exact\d2-table.txt'))
    [System.IO.File]::WriteAllText($table, $raw.Replace("`r`n", "`n"))
    if ((Sha $table) -ne '4cc216d2fe691d060887d90b12b9267afe17d1c7b97a47e575137abf7fb2b853') { throw "D2 table hash differs: $table" }
    . (Join-Path $src 'tests\t2130-g5-red-suite\resolve_toolchain.ps1')
    Enter-SuperSlmVsDevShell | Out-Null
    $bench = Join-Path $bd 'sslm_schema_fast_decode_bench.exe'
    Say "start: tree $head (engine = $engineCommit), steps $($Steps -join ',')"

    foreach ($step in $Steps) {
        if (Test-Path (Join-Path $out "$step.done")) { Say "$step already done: skipped"; continue }
        Say "$step start"
        if ($step -eq 'build') {
            if (-not (Test-Path (Join-Path $bd 'build.ninja'))) {
                & cmake -S $src -B $bd -G Ninja -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF -DSUPERSLM_BUILD_GPU=OFF *> (Join-Path $out 'configure.log')
                if ($LASTEXITCODE -ne 0) { throw 'configure failed (configure.log)' }
            }
            & cmake --build $bd --target superslm sslm_schema_fast_decode_bench sslm_decode_threading_bench *> (Join-Path $out 'build.log')
            if ($LASTEXITCODE -ne 0) { throw 'build failed (build.log)' }
            Say "build: bench sha256 $(Sha $bench)"
        } elseif ($step -eq 'pinmap') {
            & powershell -NoProfile -ExecutionPolicy Bypass -File (Join-Path $src 'tools\msvc_pin\run-pinmap.ps1') -Worktree $src -BuildDir $bd -MapDir (Join-Path $out 'pinmap') *> (Join-Path $out 'pinmap.log')
            $code = $LASTEXITCODE
            Get-Content (Join-Path $out 'pinmap.log') | ForEach-Object { Say "pinmap: $_" }
            if ($code -ne 0) { throw "run-pinmap.ps1 exited $code (pinmap.log)" }
        } elseif ($step -like 'b7b-*') {
            $dir = Join-Path $out $step; New-Item -ItemType Directory -Force -Path $dir | Out-Null
            $tasks = @(); if ($step -eq 'b7b-t4') { $tasks = @('--tasks', '4') }
            $prompts = @([ordered]@{ id = 'B2'; args = @('--example') })
            foreach ($line in Get-Content $pop) {
                $p = $line | ConvertFrom-Json
                $prompts += [ordered]@{ id = $p.id; args = @('--example', '--utterance', $p.customer_line) }
            }
            $rows = @(); $diff = 0; $arms = 0
            foreach ($pr in $prompts) {
                $j = Bench $bench (@($Model, 'extract', 'potion_shop_order') + $pr.args + @('--max-new', '64', '--pair', 'off,on:exact:learned,on:exact,on:exact:wrong', '--pairs', '1', '--warmup', '0', '--draft-table', $table) + $tasks) (Join-Path $dir "$($pr.id).json")
                $off = $j.arms | Where-Object label -eq 'off'
                $offStream = ($off.runs[0].stream -join ','); $offText = $off.runs[0].text
                $row = [ordered]@{ id = $pr.id; off_digest = $off.runs[0].digest; off_tokens = $off.runs[0].tokens; arms = @() }
                foreach ($a in ($j.arms | Where-Object label -ne 'off')) {
                    $arms++
                    $same = (($a.runs[0].stream -join ',') -eq $offStream) -and ($a.runs[0].text -ceq $offText)
                    if (-not $same) { $diff++ }
                    $row.arms += [ordered]@{ label = $a.label; digest = $a.runs[0].digest; identical_to_off = $same }
                }
                $rows += $row
                if ($diff -gt 0) { break }
            }
            ([ordered]@{ step = $step; tree = $head; prompts = $rows.Count; arms = $arms; differences = $diff; rows = $rows } | ConvertTo-Json -Depth 6) | Set-Content (Join-Path $dir 'summary.json')
            $b2 = $rows | Where-Object id -eq 'B2'
            Say "${step}: $($rows.Count) prompts, $arms exact arms, $diff not identical to off; B2 off digest $($b2.off_digest), $($b2.off_tokens) tokens"
            if ($diff -gt 0) { throw "STOP: an exact arm differs from off ($dir\summary.json)" }
            if ($rows.Count -ne 41 -or $arms -ne 123) { throw "expected 41 prompts and 123 exact arms" }
        } elseif ($step -eq 'abp') {
            $ref = $RefBench
            if (-not (Test-Path $ref) -or (Sha $ref) -ne 'fcba88dbd5b1be5bca1e5d827d1b9be6e03c49eab258df06c391183120ade583') {
                Say "abp: B7's bench is not at $RefBench with its recorded hash; building aa0bc94c fresh as the reference"
                $refWt = Join-Path $Scratch 'ref-aa0bc94c'
                if (-not (Test-Path $refWt)) { git -C $src worktree add -q --detach $refWt aa0bc94c; if ($LASTEXITCODE) { throw 'worktree add aa0bc94c' } }
                $refBd = Join-Path $refWt 'build-msvc'
                if (-not (Test-Path (Join-Path $refBd 'build.ninja'))) {
                    & cmake -S $refWt -B $refBd -G Ninja -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF -DSUPERSLM_BUILD_GPU=OFF *> (Join-Path $out 'ref-configure.log')
                    if ($LASTEXITCODE -ne 0) { throw 'reference configure failed (ref-configure.log)' }
                }
                & cmake --build $refBd --target superslm sslm_schema_fast_decode_bench *> (Join-Path $out 'ref-build.log')
                if ($LASTEXITCODE -ne 0) { throw 'reference build failed (ref-build.log)' }
                $ref = Join-Path $refBd 'sslm_schema_fast_decode_bench.exe'
            }
            $armList = @([ordered]@{ label = 'ref'; bench = $ref }, [ordered]@{ label = 'release'; bench = $bench })
            foreach ($a in $armList) { Say "abp arm $($a.label): $($a.bench) sha256 $(Sha $a.bench)" }
            $dir = Join-Path $out 'abp'; New-Item -ItemType Directory -Force -Path $dir | Out-Null
            $args1 = @($Model, 'pass', '--example', '--sizes', '7', '--no-one-row', '--pairs', '1', '--warmup', '1')
            $ms = @{ ref = @(); release = @() }
            for ($r = 0; $r -le $Rounds; ++$r) {
                $order = $armList; if ($r % 2) { $order = @($armList[1], $armList[0]) }
                foreach ($a in $order) {
                    $j = Bench $a.bench $args1 (Join-Path $dir "r$r-$($a.label).json")
                    if ($r -gt 0) { $ms[$a.label] += [double]$j.sizes[0].batched_ms.median }
                }
            }
            $ratios = @(); for ($i = 0; $i -lt $Rounds; ++$i) { $ratios += $ms['release'][$i] / $ms['ref'][$i] }
            $sum = [ordered]@{ rounds = $Rounds; ref = (Stat $ms['ref']); release = (Stat $ms['release']); release_over_ref = (Stat $ratios) }
            ($sum | ConvertTo-Json -Depth 6) | Set-Content (Join-Path $dir 'summary.json')
            Say "abp: 7-token prefill, one thread: ref $($sum.ref.median) ms, release $($sum.release.median) ms, release/ref $($sum.release_over_ref.median) [$($sum.release_over_ref.sign_lo), $($sum.release_over_ref.sign_hi)]"
            if ($sum.release_over_ref.median -gt 1.03 -or $sum.release_over_ref.median -lt 0.97) { throw 'abp: the release tree is not within 3% of the reference' }
        } else { throw "unknown step $step" }
        Set-Content (Join-Path $out "$step.done") (Get-Date -Format o)
        Say "$step done"
    }
    Say 'end PASS'
}
catch { $failed = $true; Say "end FAIL $($_.Exception.Message) (line $($_.InvocationInfo.ScriptLineNumber))" }
if ($failed) { exit 1 } else { exit 0 }
