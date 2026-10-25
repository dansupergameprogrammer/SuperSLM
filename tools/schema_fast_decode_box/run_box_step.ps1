# Schema fast-decode plan §10, steps B0-B4, on the box. A thin driver: every measurement, statistic
# and score is the bench's (tools/schema_fast_decode_bench.cpp); this script builds, sequences the
# bench's runs, checks the frozen inputs, and wraps each step's readings with their provenance.
#
#   B0   build the J1 ref's library and bench in <WorktreeRoot>\<Name>, and the main ref's library
#        in <WorktreeRoot>\<Name>-main with the same bench source compiled against it
#        (SFD_BENCH_DECODE_ONLY). Incremental.
#   B1   batched pass T(M), M = 1..8, at the example prompt's depth; the finish at 0, 1, 8, 15, 16,
#        22, 1000 and V legal rows, greedy and damped, at 1 and at -Tasks tasks.
#   B2   one thread: AA (off vs off), 2X (off twice vs once), main vs M1 (process pairs, pooled),
#        jump-forward off vs on, CB1 (off vs on at chunk budget 1).
#   B3   -Tasks tasks: AA, main vs M1, off vs on, span size D(M) against one-token threaded
#        admissions, and the real spans.
#   B4a/b  the labelled potion population, rows 0-19 / 20-39, off and on; the on arm sealed.
#   B4c/d  the te368 Prompt_Result population, rows 0-19 / 20-39; the on arm sealed.
#
# Each step writes <OutDir>\out\<step>.json, whose text ends with "complete": true}; a step whose file
# ends so is skipped (-Force reruns it). One step per invocation, or a range (B1-B3, B4 = B4a-B4d,
# all), with -RestSeconds between steps. Timed comparisons are -Pairs interleaved pairs after one
# discarded warm-up pair, alternating which arm runs first.
#
# MSVC GEMM placement (2026-10-21, results\2026-10-21-msvc-pin\): under MSVC, src\matmul.cpp's code is pinned in
# its own image section (.sslmgem) behind a 16-byte head, so the DotRow loop that runs every pass below 8 tokens
# lands at 0 mod 64 whatever else links ahead of it. Unpinned, unrelated edits moved it by 16 bytes and the one-
# thread 7-token prefill by ~28 %. The pin is fixed by matmul.cpp's own content, so after ANY edit to
# src\matmul.cpp (or a toolchain change) re-run the placement check before trusting a timed step:
#   powershell -ExecutionPolicy Bypass -File tools\schema_fast_decode_box\results\2026-10-21-msvc-pin\run-pinmap.ps1 -Commit <sha>
# and confirm GemmInt8AccumulateCols at 0, DotRowAvx2 at 16, TiledGemmAvx2 at 32 and ActiveGemmTier at 32 mod 64.
# If they moved, re-tune the head's size in matmul.cpp (README there) before measuring.
#
# Usage:  powershell -ExecutionPolicy Bypass -File tools\schema_fast_decode_box\run_box_step.ps1 -Step B0
param(
    [Parameter(Mandatory = $true)][string]$Step,
    [string]$Name = 'schema-fast-decode',
    [string]$Ref = 'origin/claude/project-thread-nu1czu',
    [string]$MainRef = 'v1.13.1',
    [string]$WorktreeRoot = 'D:\SuperSLM\.worktrees',
    # A-EX with the tokenizer and both schemas (potion_shop_order, prompt_result): the example's ExampleModel.sslm,
    # sha256 9ccf7e37... The plain -aex.sslm has no tokenizer, and every step that tokenizes fails on it.
    [string]$Model = 'D:\_artifacts\SuperSLM\aex\qwen2.5-0.5b-instruct-cap4096-aex-pr.sslm',
    [string]$OutDir = 'D:\_scratch\schema-fast-decode',
    [string]$BoxDir = '',
    [string]$Te368 = '',
    [int]$Pairs = 15,
    [int]$Tasks = 4,
    [int]$RestSeconds = 60,
    [switch]$Force
)
$ErrorActionPreference = 'Stop'
# Launched from pwsh 7, Windows PowerShell 5.1 inherits pwsh's PSModulePath, and Get-FileHash then fails to load.
# Under 5.1, use its own module path.
if ($PSVersionTable.PSEdition -ne 'Core') {
    $env:PSModulePath = (@([Environment]::GetFolderPath('MyDocuments') + '\WindowsPowerShell\Modules',
        "$env:ProgramFiles\WindowsPowerShell\Modules", "$env:SystemRoot\system32\WindowsPowerShell\v1.0\Modules") -join ';')
}

# The frozen inputs (sha256). A mismatch refuses B1-B4: a changed file is a different population or
# a different set of rules, never an edit to these.
$RulesFile = 'acceptance-rules.md'
$RulesSha = 'cef1d73e3bdc31fa7c5f896c78a8ae5a7561c58b61bbb7338eb71c5e9e91e273'
$PotionFile = 'potion-population-v1.jsonl'
$PotionSha = 'd846b9e301bdd4369ff3fc51fca43af6c06de0acd97cb09489eaac107e5a67da'
$Te368Sha = 'ef597551e186016d71cf72627e10c09c37afbf6f07f3c36b3091f1cccd6bd582'

$Steps = @('B0', 'B1', 'B2', 'B3', 'B4a', 'B4b', 'B4c', 'B4d')
if (-not $BoxDir) { $BoxDir = Join-Path $PSScriptRoot 'box' }
$Out = Join-Path $OutDir 'out'
$Parts = Join-Path $OutDir 'parts'
$Logs = Join-Path $OutDir 'logs'
$Sealed = Join-Path $OutDir 'sealed'
New-Item -ItemType Directory -Force -Path $OutDir, $Out, $Parts, $Logs, $Sealed | Out-Null
$Utf8 = New-Object System.Text.UTF8Encoding($false)

function Get-Sha256([string]$Path) {
    return (Get-FileHash -Algorithm SHA256 -LiteralPath $Path).Hash.ToLowerInvariant()
}
function Read-Utf8([string]$Path) { return [System.IO.File]::ReadAllText($Path, $Utf8).Trim() }
function Write-Utf8([string]$Path, [string]$Text) {
    $tmp = "$Path.tmp"
    [System.IO.File]::WriteAllText($tmp, $Text, $Utf8)
    Move-Item -LiteralPath $tmp -Destination $Path -Force
}
function Test-Complete([string]$Path) {
    if (-not (Test-Path -LiteralPath $Path)) { return $false }
    return (Read-Utf8 $Path) -match '"complete":\s*true\}$'
}
function ConvertTo-J($Value) { return (ConvertTo-Json -InputObject $Value -Depth 8 -Compress) }

# Runs a native program with stdout and stderr streamed to files (a crash keeps what was written);
# throws on a nonzero exit, since a crashed run fails the step and is never dropped.
function Invoke-Logged([string]$Exe, [string[]]$Arguments, [string]$Log) {
    foreach ($a in $Arguments) { if ($a -match '\s') { throw "argument with whitespace is not supported: '$a'" } }
    $p = Start-Process -FilePath $Exe -ArgumentList $Arguments -RedirectStandardOutput $Log `
        -RedirectStandardError "$Log.err" -WorkingDirectory (Split-Path -Parent $Log) -NoNewWindow -PassThru
    $null = $p.Handle  # keeps ExitCode readable after WaitForExit on Windows PowerShell 5.1
    $p.WaitForExit()
    if ($p.ExitCode -ne 0) { throw "$(Split-Path -Leaf $Exe) exited $($p.ExitCode); see $Log and $Log.err" }
}

# One bench run whose --json output is one named part of the step; returns the part's raw JSON text.
function Invoke-Bench([string]$Exe, [string]$Part, [string[]]$Arguments) {
    $json = Join-Path $Parts "$Part.json"
    Remove-Item -LiteralPath $json -ErrorAction SilentlyContinue
    Invoke-Logged $Exe ($Arguments + @('--json', $json)) (Join-Path $Logs "$Part.log")
    return Read-Utf8 $json
}

function Get-HostReport {
    $r = [ordered]@{ computer = $env:COMPUTERNAME; utc = (Get-Date).ToUniversalTime().ToString('o') }
    try { $r.cpu = (Get-CimInstance Win32_Processor | Select-Object -First 1).Name } catch { $r.cpu = 'unavailable' }
    try { $r.power_plan = ((& powercfg /getactivescheme) -join ' ').Trim() } catch { $r.power_plan = 'unavailable' }
    $load = @()
    try {
        for ($i = 0; $i -lt 3; ++$i) {
            $load += [int](Get-CimInstance Win32_Processor | Measure-Object -Property LoadPercentage -Average).Average
            Start-Sleep -Seconds 1
        }
    } catch { $load = @('unavailable') }
    $r.load_percent = ($load -join ',')
    $busy = @(Get-Process -Name 'UnrealEditor', 'UnrealEditor-Cmd', 'cl', 'link', 'UnrealBuildTool' -ErrorAction SilentlyContinue |
        ForEach-Object { "$($_.ProcessName) ($($_.Id))" })
    $r.offenders = $busy
    return $r
}

# ---- the frozen-input checks --------------------------------------------------------------------
function Assert-Frozen([string]$Path, [string]$Want, [string]$What) {
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) { throw "$What not found at $Path (stage it there first)" }
    $have = Get-Sha256 $Path
    if ($have -ne $Want) { throw "REFUSED: $What at $Path has sha256 $have, not the frozen $Want" }
    return $have
}
function Assert-RulesAndPopulation {
    $null = Assert-Frozen (Join-Path $BoxDir $RulesFile) $RulesSha 'the acceptance rules'
    $null = Assert-Frozen (Join-Path $BoxDir $PotionFile) $PotionSha 'the potion population'
}

# ---- B0's record: where the binaries are -------------------------------------------------------
$B0Path = Join-Path $Out 'B0.json'
function Get-Build {
    if (-not (Test-Complete $B0Path)) { throw "B0 is not complete: run -Step B0 first" }
    $b = (Read-Utf8 $B0Path | ConvertFrom-Json).build
    foreach ($f in @($b.j1.bench, $b.j1.lib, $b.main.bench, $b.main.lib)) {
        if (-not (Test-Path -LiteralPath $f)) { throw "B0's binary $f is gone: rerun B0 with -Force" }
    }
    return $b
}
function Get-Provenance($b) {
    $p = [ordered]@{
        j1     = [ordered]@{ ref = $b.j1.ref; commit = $b.j1.commit; lib = $b.j1.lib; lib_sha256 = (Get-Sha256 $b.j1.lib)
            bench = $b.j1.bench; bench_sha256 = (Get-Sha256 $b.j1.bench) }
        main   = [ordered]@{ ref = $b.main.ref; commit = $b.main.commit; lib = $b.main.lib; lib_sha256 = (Get-Sha256 $b.main.lib)
            bench = $b.main.bench; bench_sha256 = (Get-Sha256 $b.main.bench) }
        model  = [ordered]@{ path = $Model; sha256 = (Get-Sha256 $Model) }
        rules_sha256 = $RulesSha
        driver_sha256 = (Get-Sha256 $PSCommandPath)
        pairs  = $Pairs
        tasks  = $Tasks
    }
    foreach ($k in @('j1', 'main')) {
        if ($p[$k].lib_sha256 -ne $b.$k.lib_sha256 -or $p[$k].bench_sha256 -ne $b.$k.bench_sha256) {
            throw "the $k binaries changed since B0 recorded them: rerun B0 with -Force"
        }
    }
    return $p
}

# ---- B0 -----------------------------------------------------------------------------------------
function Initialize-Worktree([string]$Repo, [string]$Path, [string]$WantRef) {
    $want = & git -C $Repo rev-parse --verify --quiet "$WantRef^{commit}"
    if ($LASTEXITCODE -ne 0 -or -not $want) { throw "$WantRef not found in $Repo (git fetch first)" }
    if (-not (Test-Path -LiteralPath $Path)) {
        & git -C $Repo worktree prune | Out-Host
        & git -C $Repo worktree add --detach $Path $want | Out-Host
        if ($LASTEXITCODE -ne 0) { throw "git worktree add $Path failed" }
    }
    $dirty = & git -C $Path status --porcelain --untracked-files=no
    if ($dirty) { throw "$Path has local changes: clean it or remove it" }
    $have = & git -C $Path rev-parse HEAD
    if ($have -ne $want) {
        & git -C $Path checkout --detach $want | Out-Host
        if ($LASTEXITCODE -ne 0) { throw "checkout of $want in $Path failed" }
    }
    return $want
}
function Invoke-CMakeBuild([string]$Src, [string]$Build, [string[]]$Targets) {
    if (-not (Test-Path -LiteralPath (Join-Path $Build 'CMakeCache.txt'))) {
        & cmake -S $Src -B $Build -G Ninja -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF -DSUPERSLM_BUILD_GPU=OFF | Out-Host
        if ($LASTEXITCODE -ne 0) { throw "configure of $Src failed" }
    }
    & cmake --build $Build --target @Targets | Out-Host
    if ($LASTEXITCODE -ne 0) { throw "build of $($Targets -join ',') in $Build failed" }
}
function Invoke-B0 {
    $repo = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
    . (Join-Path $repo 'tests\t2130-g5-red-suite\resolve_toolchain.ps1')
    Enter-SuperSlmVsDevShell | Out-Host
    foreach ($tool in @('cmake', 'ninja', 'cl', 'git')) {
        if (-not (Get-Command $tool -ErrorAction SilentlyContinue)) { throw "$tool not on PATH in the developer shell" }
    }
    $j1Src = Join-Path $WorktreeRoot $Name
    $mainSrc = Join-Path $WorktreeRoot "$Name-main"
    $j1Commit = Initialize-Worktree $repo $j1Src $Ref
    $mainCommit = Initialize-Worktree $repo $mainSrc $MainRef
    & git -C $repo merge-base --is-ancestor $mainCommit $j1Commit
    if ($LASTEXITCODE -ne 0) { throw "$MainRef ($mainCommit) is not an ancestor of $Ref ($j1Commit)" }
    $j1Build = Join-Path $j1Src 'build-box'
    $mainBuild = Join-Path $mainSrc 'build-box'
    Invoke-CMakeBuild $j1Src $j1Build @('superslm', 'sslm_schema_fast_decode_bench')
    Invoke-CMakeBuild $mainSrc $mainBuild @('superslm')
    $j1Bench = Join-Path $j1Build 'sslm_schema_fast_decode_bench.exe'
    $j1Lib = Join-Path $j1Build 'superslm.lib'
    $mainLib = Join-Path $mainBuild 'superslm.lib'
    foreach ($f in @($j1Bench, $j1Lib, $mainLib)) { if (-not (Test-Path -LiteralPath $f)) { throw "missing $f after the build" } }
    # The main-side bench: the J1 worktree's bench source, decode-only, against main's headers and
    # library, with CMake's Release flags for MSVC and the project's own.
    $mainBench = Join-Path $mainBuild 'sslm_schema_fast_decode_bench_main.exe'
    $objDir = "$mainBench.obj"
    New-Item -ItemType Directory -Force -Path $objDir | Out-Null
    $clArgs = @('/nologo', '/std:c++20', '/O2', '/MD', '/DNDEBUG', '/EHsc', '/fp:precise', '/W3', '/DSFD_BENCH_DECODE_ONLY',
        "/I$(Join-Path $mainSrc 'include')", (Join-Path $j1Src 'tools\schema_fast_decode_bench.cpp'), $mainLib, "/Fe$mainBench")
    Push-Location $objDir
    try { & cl @clArgs | Out-Host; $code = $LASTEXITCODE } finally { Pop-Location }
    if ($code -ne 0 -or -not (Test-Path -LiteralPath $mainBench)) { throw "cl failed for $mainBench (exit $code)" }
    $build = [ordered]@{
        j1   = [ordered]@{ ref = $Ref; commit = $j1Commit; worktree = $j1Src; lib = $j1Lib; lib_sha256 = (Get-Sha256 $j1Lib)
            bench = $j1Bench; bench_sha256 = (Get-Sha256 $j1Bench) }
        main = [ordered]@{ ref = $MainRef; commit = $mainCommit; worktree = $mainSrc; lib = $mainLib; lib_sha256 = (Get-Sha256 $mainLib)
            bench = $mainBench; bench_sha256 = (Get-Sha256 $mainBench); bench_source_commit = $j1Commit }
    }
    # cl prints its banner on stderr, which Windows PowerShell 5.1 turns into an error under 'Stop'.
    $cl = 'unavailable'
    $eap = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try { $cl = ((& cl 2>&1 | ForEach-Object { "$_" }) | Select-Object -First 1) } catch { } finally { $ErrorActionPreference = $eap }
    return '"build":' + (ConvertTo-J $build) + ',"toolchain":' + (ConvertTo-J ([ordered]@{ cl = "$cl" }))
}

# ---- B1-B4 --------------------------------------------------------------------------------------
$Extract = @('extract', 'potion_shop_order', '--example', '--max-new', '64')

# main vs M1: one process per arm per pair (the bench pools them). Pair 0 is the warm-up and runs
# main first; measured pair k (from 0) runs main first when k is even.
function Invoke-MainVsM1($b, [string]$Prefix, [string[]]$Extra) {
    $files = @()
    $first = ''
    for ($p = 0; $p -le $Pairs; ++$p) {
        $k = $p - 1
        $mainFirst = ($k -lt 0) -or ($k % 2 -eq 0)
        if ($k -ge 0) { if ($mainFirst) { $first += 'A' } else { $first += 'B' } }
        $order = @('a', 'b')
        if (-not $mainFirst) { $order = @('b', 'a') }
        foreach ($side in $order) {
            $exe = $b.j1.bench
            if ($side -eq 'a') { $exe = $b.main.bench }
            $null = Invoke-Bench $exe "$Prefix-p$p$side" (@($Model) + $Extract + @('--pair', 'off', '--pairs', '1', '--warmup', '1') + $Extra)
        }
        $files += (Join-Path $Parts "$Prefix-p${p}a.json")
        $files += (Join-Path $Parts "$Prefix-p${p}b.json")
    }
    return Invoke-Bench $b.j1.bench $Prefix (@('-', 'pool', '--warmup', '1', '--labels', 'main,m1', '--first', $first) + $files)
}

function Invoke-Readings([string]$S, $b) {
    $j1 = $b.j1.bench
    $n = @('--pairs', "$Pairs", '--warmup', '1')
    $t = @('--tasks', "$Tasks")
    $r = [ordered]@{}
    switch ($S) {
        'B1' {
            $r.pass = Invoke-Bench $j1 'B1-pass' (@($Model, 'pass', '--example', '--no-one-row') + $n)
            $r.finishrows = Invoke-Bench $j1 'B1-finishrows' (@($Model, 'finishrows', '--rows', '0,1,8,15,16,22,1000,V') + $n)
            $r.finishrows_tasks = Invoke-Bench $j1 'B1-finishrows-tasks' (@($Model, 'finishrows', '--rows', '0,1,8,15,16,22,1000,V') + $n + $t)
        }
        'B2' {
            # The controls are read before the real pairs (the rules' §3); CB1 runs with them.
            $r.aa = Invoke-Bench $j1 'B2-aa' (@($Model) + $Extract + @('--pair', 'off,off') + $n)
            $r.x2 = Invoke-Bench $j1 'B2-2x' (@($Model) + $Extract + @('--pair', 'off2,off') + $n)
            $r.main_vs_m1 = Invoke-MainVsM1 $b 'B2-mm' @()
            $r.off_vs_on = Invoke-Bench $j1 'B2-jf' (@($Model) + $Extract + @('--pair', 'off,on') + $n)
            $r.cb1 = Invoke-Bench $j1 'B2-cb1' (@($Model) + $Extract + @('--pair', 'off,on:1') + $n)
        }
        'B3' {
            $r.aa = Invoke-Bench $j1 'B3-aa' (@($Model) + $Extract + @('--pair', 'off,off') + $n + $t)
            $r.main_vs_m1 = Invoke-MainVsM1 $b 'B3-mm' $t
            $r.off_vs_on = Invoke-Bench $j1 'B3-jf' (@($Model) + $Extract + @('--pair', 'off,on') + $n + $t)
            $r.span_size = Invoke-Bench $j1 'B3-pass' (@($Model, 'pass', '--example') + $n + $t)
            $r.spans = Invoke-Bench $j1 'B3-spans' (@($Model, 'spans', 'potion_shop_order', '--example', '--pairs', "$Pairs") + $t)
        }
        { $_ -in @('B4a', 'B4b', 'B4c', 'B4d') } {
            $from = '0'
            if ($S -eq 'B4b' -or $S -eq 'B4d') { $from = '20' }
            if ($S -eq 'B4a' -or $S -eq 'B4b') {
                $pop = Join-Path $BoxDir $PotionFile
                $args4 = @($Model, 'quality', 'potion_shop_order', '--population', $pop, '--kind', 'potion', '--max-new', '64')
            } else {
                $pop = $Te368
                if (-not $pop) { $pop = Join-Path $b.j1.worktree 'tests\t2130-g5-red-suite\reference\te368-probe\te368_prompts.json' }
                $null = Assert-Frozen $pop $Te368Sha 'the te368 population'
                $args4 = @($Model, 'quality', 'prompt_result', '--population', $pop, '--kind', 'prompt_result', '--max-new', '300')
            }
            $seal = Join-Path $Sealed "$S.on.json"
            Remove-Item -LiteralPath $seal -ErrorAction SilentlyContinue
            $r.quality_off = Invoke-Bench $j1 "$S-quality" ($args4 + @('--from', $from, '--count', '20', '--sealed', $seal))
            $r.sealed_on = ConvertTo-J ([ordered]@{ path = $seal; sha256 = (Get-Sha256 $seal); population = $pop; population_sha256 = (Get-Sha256 $pop) })
        }
    }
    $parts = @()
    foreach ($k in $r.Keys) { $parts += ((ConvertTo-J $k) + ':' + $r[$k]) }
    return '"readings":{' + ($parts -join ',') + '}'
}

function Invoke-Step([string]$S) {
    $path = Join-Path $Out "$S.json"
    if ((Test-Complete $path) -and -not $Force) {
        Write-Host "$S is complete ($path): skipped"
        return $false
    }
    Write-Host "== $S"
    $started = (Get-Date).ToUniversalTime().ToString('o')
    if ($S -eq 'B0') {
        $body = Invoke-B0
        $text = '{"step":"B0",' + $body + ',"started":' + (ConvertTo-J $started) + ',"finished":' +
            (ConvertTo-J ((Get-Date).ToUniversalTime().ToString('o'))) + ',"complete": true}'
    } else {
        if ($S -eq 'B1' -or $S -like 'B4*') { Assert-RulesAndPopulation }
        if (-not (Test-Path -LiteralPath $Model -PathType Leaf)) { throw "artifact not found: $Model" }
        $b = Get-Build
        $prov = Get-Provenance $b
        $before = Get-HostReport
        $readings = Invoke-Readings $S $b
        $after = Get-HostReport
        $text = '{"step":' + (ConvertTo-J $S) + ',"provenance":' + (ConvertTo-J $prov) + ',"host_before":' + (ConvertTo-J $before) +
            ',"host_after":' + (ConvertTo-J $after) + ',"started":' + (ConvertTo-J $started) + ',' + $readings + ',"complete": true}'
    }
    Write-Utf8 $path $text
    Write-Host "$S complete: $path"
    return $true
}

# ---- which steps --------------------------------------------------------------------------------
function Resolve-Steps([string]$Spec) {
    if ($Spec -eq 'all') { return $Steps }
    if ($Spec -eq 'B4') { $Spec = 'B4a-B4d' }
    $ends = $Spec -split '-'
    foreach ($e in $ends) { if ($Steps -notcontains $e) { throw "unknown step '$e' (one of $($Steps -join ', '), B4, all, or a range like B1-B3)" } }
    if ($ends.Count -eq 1) { return @($ends[0]) }
    if ($ends.Count -ne 2) { throw "a range is two steps: $Spec" }
    $i = [array]::IndexOf($Steps, $ends[0])
    $j = [array]::IndexOf($Steps, $ends[1])
    if ($j -lt $i) { throw "range $Spec runs backwards" }
    return $Steps[$i..$j]
}

$todo = @(Resolve-Steps $Step)
$ran = $false
foreach ($s in $todo) {
    if ($ran -and $RestSeconds -gt 0) {
        Write-Host "resting $RestSeconds s"
        Start-Sleep -Seconds $RestSeconds
    }
    $ran = Invoke-Step $s
}
