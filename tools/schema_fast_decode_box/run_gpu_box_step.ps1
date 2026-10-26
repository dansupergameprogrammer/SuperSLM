# Schema fast-decode plan rev 1.4.4 Sec. 10, the GPU steps on the box (the RTX 2080 SUPER), and the kit's steps on the work
# machine (now workhorse's Radeon RX 7900 XT, the same device id as the XTX). run_box_step.ps1 hands every GB, GK and GB-XG step to this script.
#
#   GB0       fixtures (F-SFD and its --untied variant, generated and hashed), the build at -Tip (the G cells, the
#             smoke, the GPU jump-forward bench), G0.1, and the F-SFD smoke at value 2, whose debug-layer message IDs
#             are the baseline for every later step
#   GB-F-r    G1-F's cells at its red-suite commit: each must be red (not linked, or a failing run)
#   GB-F-g    G1-F's cells at its build commit: each must be green
#   GB-M-r/g  the same for G1-M (G5.1, G6.1-G6.5, G8.1), on F-SFD and its untied variant
#   GB-J-r/g  the same for G1-J (G7.1, G9.1, G11.1, G12.1-G12.3, G12.6-G12.9); g also runs P2.16's reference loop
#             (tools/schema_fast_decode_gpu_p2_ref.cpp) and files P2_16_expected.json with its sha256
#   GB1       the release candidate at -Tip: every G cell, each binary alone
#   GB-XG-1..7  the engine GPU source mutants (tests/schema-fast-decode-gpu/make_sfdg_mutants.py), two per step:
#             each mutant is applied to a mutant worktree at -Tip, built, and must be killed by its cell
#   GBK       pack the work machine's kit from GB1's build, with a sha256 manifest (not a GPU step)
#   GB2c      CPU off and canonical streams on B7b's 41 prompts at -Tip (CPU only; not a GPU step)
#   GB2       exactness, whole token: GPU off and canonical at chunk budget 1, 2 and 4 (device head), off and 1
#             (host head), each against GB2c's CPU streams
#   GB3       speed, headless, at value 0: off at -Base against off at -Tip (the masked finish), the probe's
#             composition against the fused verb at 4, off against canonical at 1 and at 2; device head, then host
#   GK0, GK1, GK2  the kit's steps (on the work machine, from the kit's directory): the F-SFD smoke, the G cells at
#             value 2, and GB2 against the kit's GB2c streams
#
# The limits (Sec. 10; rev 1.4.1 review S3, audit G-9; rev 1.4.2 strike N3, N6), which this driver enforces:
#   - one step per invocation: no ranges for GB, GK or GB-XG steps;
#   - a hard cap: the step's process tree is killed at -CapMinutes (15) of wall time. The cell it was in is recorded
#     as "timeout", the step file gets "complete": false and "next_cell", and a re-run resumes at next_cell. A step
#     with a timeout is never complete: split it by cell id (-Cells) and re-run;
#   - SSLM_GPU_ENABLE_DEBUG_LAYER=2 for every G binary (GB3: 0). A binary whose banner reports GPU-based validation
#     fails the step. GPU-based validation never runs on the home box (GKV is the kit's optional step, not here);
#   - the load ledger (<OutDir>\gpu-load-ledger.json, kept across invocations): a GPU step starts only when the GPU
#     wall time since the last pause of >= -PauseMinutes (5), plus the step's own cap, is at most the cap; so every
#     GPU step follows a pause of >= 5 min. Otherwise the driver refuses and says how long to wait;
#   - the debug layer is read: every G binary prints each queued message's ID at exit ("# SFDG DEBUG LAYER MESSAGE
#     id=..."); from GB0 on, a message ID outside GB0's baseline fails a green cell. A binary built before that
#     report existed prints none, which is recorded as "no report".
#
# Each step writes <OutDir>\out\<step>.json, which ends "complete": true}; a complete step is skipped (-Force reruns it).
#
# Usage:  powershell -ExecutionPolicy Bypass -File tools\schema_fast_decode_box\run_gpu_box_step.ps1 -Step GB0
#         (or through run_box_step.ps1 -Step GB0)
param(
    [Parameter(Mandatory = $true)][string]$Step,
    [string]$Name = 'sfd-gpu',
    [string]$WorktreeRoot = 'D:\SuperSLM\.worktrees',
    [string]$OutDir = 'D:\_scratch\sfd-gpu',
    [string]$Model = 'D:\_artifacts\SuperSLM\aex\qwen2.5-0.5b-instruct-cap4096-aex-pr.sslm',
    [string]$Python = 'python',
    [string]$BoxDir = '',
    # The GB rules file the debunking seat freezes before GB2 (its sha256 is recorded with every GB2 and GB3 reading).
    [string]$GbRules = '',
    [string]$Base = '5599f7e4',
    [string]$RedF = '88518b14',
    [string]$GreenF = '1d58e32a',
    [string]$RedM = '0c996cd3',
    [string]$GreenM = '13eeeadb',
    [string]$RedJ = 'f9411c6a',
    [string]$GreenJ = 'origin/claude/sfd-gpu-1p17',
    [string]$Tip = 'origin/claude/sfd-gpu-1p17',
    [string]$Dxc = 'C:\Program Files (x86)\Windows Kits\10\bin\10.0.19041.0\x64\dxc.exe',
    [string]$KitDir = '',
    # The kit's own directory, when this script runs from a packed kit (GK0-GK2).
    [string]$Kit = '',
    [string]$Cells = '',
    [int]$Pairs = 15,
    [int]$CapMinutes = 15,
    [int]$PauseMinutes = 5,
    [switch]$Force
)
$ErrorActionPreference = 'Stop'
if ($PSVersionTable.PSEdition -ne 'Core') {
    $env:PSModulePath = (@([Environment]::GetFolderPath('MyDocuments') + '\WindowsPowerShell\Modules',
        "$env:ProgramFiles\WindowsPowerShell\Modules", "$env:SystemRoot\system32\WindowsPowerShell\v1.0\Modules") -join ';')
}

$GpuSteps = @('GB0', 'GB-F-r', 'GB-F-g', 'GB-M-r', 'GB-M-g', 'GB-J-r', 'GB-J-g', 'GB1', 'GB2', 'GB3',
    'GB-XG-1', 'GB-XG-2', 'GB-XG-3', 'GB-XG-4', 'GB-XG-5', 'GB-XG-6', 'GB-XG-7', 'GK0', 'GK1', 'GK2')
$OtherSteps = @('GBK', 'GB2c')
if (($GpuSteps + $OtherSteps) -notcontains $Step) {
    throw "unknown or ranged step '$Step': one of $(($GpuSteps + $OtherSteps) -join ', '), one per invocation (Sec. 10: no ranges)"
}
$IsGpuStep = $GpuSteps -contains $Step
if (-not $BoxDir) { $BoxDir = Join-Path $PSScriptRoot 'frozen' }
$Repo = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$Out = Join-Path $OutDir 'out'
$Logs = Join-Path $OutDir 'logs'
$Fixtures = Join-Path $OutDir 'fixtures'
New-Item -ItemType Directory -Force -Path $OutDir, $Out, $Logs | Out-Null
$Utf8 = New-Object System.Text.UTF8Encoding($false)
$StepPath = Join-Path $Out "$Step.json"
$Ledger = Join-Path $OutDir 'gpu-load-ledger.json'
$Started = Get-Date
$Deadline = $Started.AddMinutes($CapMinutes)

function Get-Sha256([string]$Path) { return (Get-FileHash -Algorithm SHA256 -LiteralPath $Path).Hash.ToLowerInvariant() }
function Read-Utf8([string]$Path) { return [System.IO.File]::ReadAllText($Path, $Utf8) }
function Write-Utf8([string]$Path, [string]$Text) {
    $tmp = "$Path.tmp"
    [System.IO.File]::WriteAllText($tmp, $Text, $Utf8)
    Move-Item -LiteralPath $tmp -Destination $Path -Force
}
function ConvertTo-J($Value) { return (ConvertTo-Json -InputObject $Value -Depth 10 -Compress) }
function Get-Utc { return (Get-Date).ToUniversalTime().ToString('o') }

# ---- the load ledger --------------------------------------------------------------------------------------------
function Read-Ledger {
    if (-not (Test-Path -LiteralPath $Ledger)) { return [pscustomobject]@{ last_end_utc = ''; since_pause_s = 0; entries = @() } }
    return (Read-Utf8 $Ledger | ConvertFrom-Json)
}
function Assert-Ledger {
    $l = Read-Ledger
    if (-not $l.last_end_utc) { return 0 }
    $idle = ((Get-Date).ToUniversalTime() - [datetime]::Parse($l.last_end_utc).ToUniversalTime()).TotalSeconds
    $since = [double]$l.since_pause_s
    if ($idle -ge $PauseMinutes * 60) { $since = 0 }
    if ($since + $CapMinutes * 60 -gt $CapMinutes * 60) {
        $wait = [math]::Ceiling($PauseMinutes * 60 - $idle)
        throw "REFUSED by the load ledger: the last GPU step ended $([int]$idle) s ago; wait $wait s more (a pause of >= $PauseMinutes min after every GPU step, Sec. 10)"
    }
    return $since
}
function Update-Ledger([double]$SinceBefore) {
    $l = Read-Ledger
    $elapsed = ((Get-Date) - $Started).TotalSeconds
    $entries = @($l.entries) + @([ordered]@{ step = $Step; started = $Started.ToUniversalTime().ToString('o'); finished = (Get-Utc); seconds = [int]$elapsed })
    $n = [ordered]@{ last_end_utc = (Get-Utc); since_pause_s = [int]($SinceBefore + $elapsed); entries = $entries }
    Write-Utf8 $Ledger (ConvertTo-J $n)
}

# ---- the step file (resumable) ----------------------------------------------------------------------------------
$State = [ordered]@{ step = $Step; started = (Get-Utc); cells = [ordered]@{}; complete = $false }
# The adapter (engine code review N10): a stale SSLM_GPU_ADAPTER_INDEX would silently pick another adapter, so it
# is recorded and cleared, and every step records the machine's display adapters and drivers. The harness then
# picks the high-performance hardware adapter.
$State.adapter_index_env_cleared = $(if ($env:SSLM_GPU_ADAPTER_INDEX) { "$env:SSLM_GPU_ADAPTER_INDEX" } else { '' })
Remove-Item Env:\SSLM_GPU_ADAPTER_INDEX -ErrorAction SilentlyContinue
try {
    $State.adapters = @(Get-CimInstance Win32_VideoController | ForEach-Object { [ordered]@{ name = $_.Name; driver = $_.DriverVersion } })
} catch { $State.adapters = @("(not read: $($_.Exception.Message))") }
if (Test-Path -LiteralPath $StepPath) {
    $prev = Read-Utf8 $StepPath | ConvertFrom-Json
    if ($prev.complete -and -not $Force) { Write-Host "$Step is complete ($StepPath): skipped"; exit 0 }
    if (-not $Force -and $prev.cells) {
        foreach ($p in $prev.cells.PSObject.Properties) { $State.cells[$p.Name] = $p.Value }
    }
}
function Save-State([bool]$Complete, [string]$NextCell) {
    $State.finished = (Get-Utc)
    if ($NextCell) { $State.next_cell = $NextCell } elseif ($State.Contains('next_cell')) { $State.Remove('next_cell') }
    $copy = [ordered]@{}
    foreach ($k in $State.Keys) { if ($k -ne 'complete') { $copy[$k] = $State[$k] } }
    $State.complete = $Complete
    # The completion marker is the file's last text, as run_box_step.ps1's steps end.
    $text = (ConvertTo-J $copy)
    $text = $text.Substring(0, $text.Length - 1) + ',"complete": ' + $(if ($Complete) { 'true' } else { 'false' }) + '}'
    Write-Utf8 $StepPath $text
}
function Get-RemainingMs { return [math]::Max(0, [int](($Deadline - (Get-Date)).TotalMilliseconds)) }

# ---- building ---------------------------------------------------------------------------------------------------
function Initialize-Worktree([string]$Path, [string]$WantRef) {
    $want = & git -C $Repo rev-parse --verify --quiet "$WantRef^{commit}"
    if ($LASTEXITCODE -ne 0 -or -not $want) { throw "$WantRef not found in $Repo (git fetch first)" }
    if (-not (Test-Path -LiteralPath $Path)) {
        & git -C $Repo worktree prune | Out-Host
        & git -C $Repo worktree add --detach $Path $want | Out-Host
        if ($LASTEXITCODE -ne 0) { throw "git worktree add $Path failed" }
    }
    $dirty = & git -C $Path status --porcelain --untracked-files=no
    if ($dirty) { throw "$Path has local changes: clean it or remove it" }
    if ((& git -C $Path rev-parse HEAD) -ne $want) {
        & git -C $Path checkout --detach $want | Out-Host
        if ($LASTEXITCODE -ne 0) { throw "checkout of $want in $Path failed" }
    }
    return $want
}
$DevShell = $false
function Enter-DevShell {
    if ($script:DevShell) { return }
    . (Join-Path $Repo 'tests\t2130-g5-red-suite\resolve_toolchain.ps1')
    Enter-SuperSlmVsDevShell | Out-Host
    foreach ($tool in @('cl', 'git')) { if (-not (Get-Command $tool -ErrorAction SilentlyContinue)) { throw "$tool not on PATH" } }
    $script:DevShell = $true
}
# A process with stdout and stderr in files, killed with its tree at the step's deadline. Returns exit code, or
# 'timeout'.
function Invoke-Capped([string]$Exe, [string[]]$Arguments, [string]$Log, [string]$WorkDir) {
    foreach ($a in $Arguments) { if ($a -match '\s') { throw "argument with whitespace is not supported: '$a'" } }
    $ms = Get-RemainingMs
    if ($ms -le 0) { return 'timeout' }
    $sp = @{ FilePath = $Exe; RedirectStandardOutput = $Log; RedirectStandardError = "$Log.err"; NoNewWindow = $true; PassThru = $true
        WorkingDirectory = $WorkDir }
    if ($Arguments.Count -gt 0) { $sp.ArgumentList = $Arguments }
    $p = Start-Process @sp
    $null = $p.Handle
    if (-not $p.WaitForExit($ms)) {
        & taskkill /T /F /PID $p.Id 2>&1 | Out-Null
        return 'timeout'
    }
    return $p.ExitCode
}
function Build-Shaders([string]$Wt) {
    $dst = Join-Path $Wt 'out\shaders'
    New-Item -ItemType Directory -Force -Path $dst | Out-Null
    if (-not (Test-Path -LiteralPath $Dxc)) { throw "dxc not found at $Dxc" }
    foreach ($h in Get-ChildItem (Join-Path $Wt 'src\gpu\shaders\*.hlsl')) {
        $cso = Join-Path $dst ($h.BaseName + '.cso')
        if ((Test-Path -LiteralPath $cso) -and (Get-Item $cso).LastWriteTimeUtc -gt $h.LastWriteTimeUtc) { continue }
        & $Dxc -T cs_6_2 -E main -Fo $cso $h.FullName -O3 -HV 2018 -WX | Out-Host
        if ($LASTEXITCODE -ne 0) { throw "dxc failed on $($h.Name)" }
    }
    return $dst
}
# The G binaries of one worktree, through build_sfd_gpu.bat. Returns name -> LINKED | RED BY LINK | RED BY COMPILE |
# NOT WRITTEN.
function Build-Cells([string]$Wt, [string[]]$Targets) {
    Enter-DevShell
    $shaders = Build-Shaders $Wt
    $outDir = Join-Path $Wt 'build\sfdg'
    $log = Join-Path $Logs "$Step-build.log"
    $bat = Join-Path $Wt 'tests\schema-fast-decode-gpu\build_sfd_gpu.bat'
    # cmd splits a batch argument at '=', so the bat sees --out=/--shaders= only when they arrive quoted (its usage line).
    $code = Invoke-Capped 'cmd.exe' (@('/c', $bat, "`"--out=$outDir`"", "`"--shaders=$shaders`"") + $Targets) $log $Wt
    if ($code -eq 'timeout') { return $null }
    $r = [ordered]@{}
    foreach ($line in Get-Content -LiteralPath $log) {
        if ($line -match '^\[(G1[FMJ])\] (\S+): (LINKED|RED BY LINK|RED BY COMPILE|NOT WRITTEN)') { $r[$Matches[2]] = $Matches[3] }
    }
    $r['_bin'] = Join-Path $outDir 'bin'
    return $r
}
# The GPU jump-forward bench (plain engine, no test seams), for GB0 and GB3.
function Build-Bench([string]$Wt, [string]$Tag) {
    Enter-DevShell
    $dir = Join-Path $Wt "build\jfbench-$Tag"
    $exe = Join-Path $dir 'sslm_schema_fast_decode_gpu_jf_bench.exe'
    if (Test-Path -LiteralPath $exe) { return $exe }
    New-Item -ItemType Directory -Force -Path $dir | Out-Null
    $srcs = @('artifact', 'sha256', 'tokenizer', 'model', 'intmath', 'silu_lut', 'matmul', 'proof_manifest', 'trace_hook',
        'forward\checked_chain_funnel', 'forward\forward_sites', 'forward\parallel_split', 'gpu\gpu_inflight_registry',
        'decode_digest', 'kv_pages', 'spx1_format', 'sslm_abi', 'damped_greedy_antilm', 'damped_greedy_topk',
        'damped_greedy_phaseD', 'damped_greedy_phaseD_loop', 'gpu\superslm_gpu', 'gpu\gpu_1p0') |
        ForEach-Object { Join-Path $Wt "src\$_.cpp" } | Where-Object { Test-Path -LiteralPath $_ }
    # A pre-1.17.0 bench reads the chain through the CPU test seam; the define is inert at 1.17.0.
    $cl = @('/nologo', '/std:c++20', '/O2', '/MD', '/DNDEBUG', '/EHsc', '/fp:precise', '/W3', '/DNOMINMAX',
        '/DSUPERSLM_ENABLE_SCHEMA_FAST_DECODE_TEST_SEAMS', "/I$(Join-Path $Wt 'include')", "/I$(Join-Path $Wt 'src')",
        "/I$(Join-Path $Wt 'src\gpu')", "/I$(Join-Path $Wt 'tests')", "/Fo$dir\", "/Fe$exe",
        (Join-Path $Wt 'tools\schema_fast_decode_gpu_jf_bench.cpp')) + $srcs + @('d3d12.lib', 'dxgi.lib', 'dxguid.lib', 'shlwapi.lib', 'psapi.lib')
    $log = Join-Path $Logs "$Step-bench-$Tag.log"
    Push-Location $dir
    try { & cl @cl > $log 2>&1; $code = $LASTEXITCODE } finally { Pop-Location }
    if ($code -ne 0 -or -not (Test-Path -LiteralPath $exe)) { throw "the bench did not build at $Wt (see $log)" }
    $sh = Join-Path $dir 'shaders'
    New-Item -ItemType Directory -Force -Path $sh | Out-Null
    Copy-Item (Join-Path (Build-Shaders $Wt) '*.cso') $sh -Force
    return $exe
}

# ---- fixtures -----------------------------------------------------------------------------------------------------
function Initialize-Fixtures([string]$Wt) {
    $fx = Join-Path $Fixtures 'fsfd.sslm'
    $ux = Join-Path $Fixtures 'fsfd_untied.sslm'
    if (-not ((Test-Path -LiteralPath $fx) -and (Test-Path -LiteralPath $ux) -and (Test-Path -LiteralPath (Join-Path $Fixtures 'fsfd_ref.txt')))) {
        New-Item -ItemType Directory -Force -Path $Fixtures | Out-Null
        $code = Invoke-Capped $Python @((Join-Path $Wt 'tools\gen_schema_fast_decode_fixture.py'), $Fixtures, '--untied') (Join-Path $Logs 'fixtures.log') $Wt
        if ($code -ne 0) { throw "the fixture generator failed ($code): see $(Join-Path $Logs 'fixtures.log')" }
    }
    return [ordered]@{ dir = $Fixtures; fsfd = (Get-Sha256 $fx); fsfd_untied = (Get-Sha256 $ux); fsfd_ref = (Get-Sha256 (Join-Path $Fixtures 'fsfd_ref.txt')) }
}

# ---- running one cell ---------------------------------------------------------------------------------------------
# The debug-layer baseline: GB0's on the box, GK0's on the kit (engine code review N2), so GK1 compares against
# the kit card's own smoke run and never against nothing.
function Read-Baseline {
    $p = Join-Path $Out $(if ($Step -like 'GK*') { 'GK0.json' } else { 'GB0.json' })
    if (-not (Test-Path -LiteralPath $p)) { return $null }
    $g = Read-Utf8 $p | ConvertFrom-Json
    if (-not $g.complete) { return $null }
    return @($g.baseline_ids)
}
function Invoke-Cell([string]$Id, [string]$Exe, [string[]]$CellArgs, [string]$Expect, [int]$Layer) {
    if ($State.cells.Contains($Id) -and $State.cells[$Id].result -ne 'timeout') { Write-Host "  $Id done: $($State.cells[$Id].result)"; return $true }
    Write-Host "  $Id"
    $rec = [ordered]@{ exe = $Exe; args = $CellArgs; expect = $Expect; layer_value = $Layer; started = (Get-Utc) }
    if (-not (Test-Path -LiteralPath $Exe)) {
        $rec.result = $(if ($Expect -eq 'red') { 'red (not linked)' } else { 'FAIL (not linked)' })
        $State.cells[$Id] = $rec
        return $true
    }
    $env:SSLM_GPU_ENABLE_DEBUG_LAYER = "$Layer"
    if ($Layer -eq 0) { Remove-Item Env:\SSLM_GPU_ENABLE_DEBUG_LAYER -ErrorAction SilentlyContinue }
    $log = Join-Path $Logs ("$Step-$Id.log" -replace '[^\w\.\-]', '_')
    $code = Invoke-Capped $Exe $CellArgs $log (Split-Path -Parent $Exe)
    if ($code -eq 'timeout') {
        $rec.result = 'timeout'
        $State.cells[$Id] = $rec
        return $false
    }
    $stdout = if (Test-Path -LiteralPath $log) { Read-Utf8 $log } else { '' }
    $stderr = if (Test-Path -LiteralPath "$log.err") { Read-Utf8 "$log.err" } else { '' }
    $rec.exit = $code
    $rec.summary = (($stdout -split "`n") | Where-Object { $_ -match 'checks=|-> (PASS|FAIL)|RED|GREEN' } | Select-Object -Last 2) -join ' | '
    $rec.fails = @(($stdout -split "`n") | Where-Object { $_ -match '^\s*FAIL' } | Select-Object -First 20)
    # The preflight: GPU-based validation must not be on.
    $rec.gbv_reported = [bool]($stderr -match 'GPU-based validation enabled')
    $rec.layer_banner = (($stderr -split "`n") | Where-Object { $_ -match 'SSLM_GPU_ENABLE_DEBUG_LAYER' } | Select-Object -First 1)
    $ids = @([regex]::Matches($stdout, '# SFDG DEBUG LAYER MESSAGE id=(-?\d+)') | ForEach-Object { [int]$_.Groups[1].Value } | Sort-Object -Unique)
    $rec.debug_layer = $(if ($stdout -match '# SFDG DEBUG LAYER: mode=') { 'reported' } else { 'no report' })
    $rec.debug_ids = $ids
    $base = Read-Baseline
    $rec.new_ids = @()
    if ($null -ne $base) { $rec.new_ids = @($ids | Where-Object { $base -notcontains $_ }) }
    if ($rec.gbv_reported) {
        $rec.result = 'FAIL (GPU-based validation reported)'
    } elseif ($Expect -eq 'red') {
        $rec.result = $(if ($code -ne 0) { 'red (failing run)' } else { 'NOT RED (passed)' })
    } elseif ($Expect -eq 'killed') {
        $rec.result = $(if ($code -ne 0) { 'killed' } else { 'SURVIVED' })
    } else {
        $rec.result = $(if ($code -eq 0 -and $rec.new_ids.Count -eq 0) { 'green' } elseif ($code -eq 0) { 'FAIL (debug-layer IDs outside the baseline)' } else { 'FAIL' })
    }
    $State.cells[$Id] = $rec
    Save-State $false ''
    return $true
}

# The cells, by id: id = exe [args]. A cell list (-Cells a,b) restricts a step to those ids.
$FxArg = "--fixture-dir=$Fixtures"
$CellTable = [ordered]@{
    'G0.1'  = @('sfdg_cell_g0_1', @())
    'smoke' = @('sfdg_smoke', @($FxArg))
    'G2.1'  = @('sfdg_cell_g2_1', @($FxArg))
    'G12.4' = @('sfdg_cell_g12_4', @($FxArg))
    'G12.5' = @('sfdg_cell_g12_5', @($FxArg))
    'G5.1'  = @('sfdg_cell_g5_1', @($FxArg))
    'G6'    = @('sfdg_cell_g6', @($FxArg))
    'G8.1'  = @('sfdg_cell_g8_1', @($FxArg))
    'G7.1'  = @('sfdg_cell_g7_1', @($FxArg))
    'G9.1'  = @('sfdg_cell_g9_1', @($FxArg))
    'G11.1' = @('sfdg_cell_g11_1', @($FxArg))
    'G12.1' = @('sfdg_cell_g12_verb', @($FxArg, '--only=G12.1'))
    'G12.2' = @('sfdg_cell_g12_verb', @($FxArg, '--only=G12.2'))
    'G12.3' = @('sfdg_cell_g12_verb', @($FxArg, '--only=G12.3'))
    'G12.8' = @('sfdg_cell_g12_verb', @($FxArg, '--only=G12.8'))
    'G12.9' = @('sfdg_cell_g12_verb', @($FxArg, '--only=G12.9'))
    'G12.6' = @('sfdg_cell_g12_6', @($FxArg))
    'G12.7' = @('sfdg_cell_g12_7', @($FxArg))
    'G12.10' = @('sfdg_cell_g12_10', @($FxArg))
}
$SliceCells = @{
    F = @('G2.1', 'G12.4', 'G12.5')
    M = @('G5.1', 'G6', 'G8.1')
    J = @('G7.1', 'G9.1', 'G11.1', 'G12.1', 'G12.2', 'G12.3', 'G12.8', 'G12.9', 'G12.6', 'G12.7', 'G12.10')
}
# Sec. 12's engine GPU mutants and the cell that kills each; two per step.
$Mutants = [ordered]@{
    'XG1' = 'G6'; 'XG9' = 'G6'; 'XG2' = 'G11.1'; 'XG8' = 'G11.1'; 'XG3' = 'G7.1'; 'XG15' = 'G12.6'
    'XG19' = 'G12.7'; 'XG30' = 'G12.7'; 'XG22' = 'G0.1'; 'XG26' = 'G0.1'; 'XG31' = 'G12.2'; 'XG37' = 'G12.9'
    'XGB1' = 'G12.10'
}

function Invoke-CellList([string[]]$Ids, [string]$Bin, [string]$Expect, [int]$Layer) {
    if ($Cells) { $Ids = @($Ids | Where-Object { ($Cells -split ',') -contains $_ }) }
    for ($i = 0; $i -lt $Ids.Count; ++$i) {
        $id = $Ids[$i]
        $t = $CellTable[$id]
        $ok = Invoke-Cell $id (Join-Path $Bin "$($t[0]).exe") $t[1] $Expect $Layer
        if (-not $ok) {
            $next = $(if ($i + 1 -lt $Ids.Count) { $Ids[$i + 1] } else { '' })
            Save-State $false $(if ($next) { $next } else { '(none: split the step)' })
            throw "TIMEOUT in $id at the $CapMinutes-minute cap: recorded; the step is not complete. Split it by cell id (-Cells) and re-run."
        }
    }
}
function Assert-Cells([string]$What) {
    $bad = @($State.cells.Keys | Where-Object { $State.cells[$_].result -notmatch '^(green|red|killed)' })
    if ($bad.Count -gt 0) { Save-State $false ''; throw "$What not met for: $($bad -join ', ') (see $StepPath)" }
}

# ---- the steps ----------------------------------------------------------------------------------------------------
function Invoke-GB0 {
    $wt = Join-Path $WorktreeRoot "$Name-tip"
    $State.commit = Initialize-Worktree $wt $Tip
    $State.fixtures = Initialize-Fixtures $wt
    $b = Build-Cells $wt @('sfdg_smoke', 'sfdg_cell_g0_1')
    if ($null -eq $b) { Save-State $false '(build)'; throw 'TIMEOUT in the build: re-run to continue it' }
    $State.build = $b
    $State.bench = Build-Bench $wt 'tip'
    Invoke-CellList @('G0.1', 'smoke') $b['_bin'] 'green' 2
    $State.baseline_ids = @($State.cells['smoke'].debug_ids)
    Assert-Cells 'GB0'
}
function Invoke-Slice([string]$Slice, [string]$Phase) {
    $ref = (Get-Variable -Name "$(if ($Phase -eq 'r') { 'Red' } else { 'Green' })$Slice" -ValueOnly)
    $wt = Join-Path $WorktreeRoot "$Name-$($Slice.ToLower())$Phase"
    $State.commit = Initialize-Worktree $wt $ref
    $State.fixtures = Initialize-Fixtures $wt
    $targets = @("g1$($Slice.ToLower())")
    if ($Slice -eq 'J' -and $Phase -eq 'g') { $targets += 'schema_fast_decode_gpu_p2_ref' }
    $b = Build-Cells $wt $targets
    if ($null -eq $b) { Save-State $false '(build)'; throw 'TIMEOUT in the build: re-run to continue it' }
    $State.build = $b
    if ($Phase -eq 'g' -and $null -eq (Read-Baseline)) { throw 'GB0 is not complete: its debug-layer baseline is needed first' }
    Invoke-CellList $SliceCells[$Slice] $b['_bin'] $(if ($Phase -eq 'r') { 'red' } else { 'green' }) 2
    if ($Slice -eq 'J' -and $Phase -eq 'g' -and -not $Cells) {
        $exp = Join-Path $Out 'P2_16_expected.json'
        $ok = Invoke-Cell 'P2.16-ref' (Join-Path $b['_bin'] 'schema_fast_decode_gpu_p2_ref.exe') @($FxArg, "--out=$exp") 'green' 2
        if (-not $ok) { Save-State $false '(none: split the step)'; throw 'TIMEOUT in the P2.16 reference loop' }
        if (Test-Path -LiteralPath $exp) { $State.p2_16_expected = [ordered]@{ path = $exp; sha256 = (Get-Sha256 $exp) } }
    }
    Assert-Cells "GB-$Slice-$Phase"
}
function Invoke-GB1 {
    $wt = Join-Path $WorktreeRoot "$Name-tip"
    $State.commit = Initialize-Worktree $wt $Tip
    $State.fixtures = Initialize-Fixtures $wt
    $b = Build-Cells $wt @('g1f', 'g1m', 'g1j')
    if ($null -eq $b) { Save-State $false '(build)'; throw 'TIMEOUT in the build: re-run to continue it' }
    $State.build = $b
    if ($null -eq (Read-Baseline)) { throw 'GB0 is not complete: its debug-layer baseline is needed first' }
    Invoke-CellList (@('G0.1', 'smoke') + $SliceCells.F + $SliceCells.M + $SliceCells.J) $b['_bin'] 'green' 2
    Assert-Cells 'GB1'
}
function Invoke-GBXG([int]$Group) {
    $names = @(@($Mutants.Keys)[(2 * ($Group - 1))..(2 * $Group - 1)] | Where-Object { $_ })
    $wt = Join-Path $WorktreeRoot "$Name-mut"
    $State.commit = Initialize-Worktree $wt $Tip
    $State.fixtures = Initialize-Fixtures $wt
    if ($null -eq (Read-Baseline)) { throw 'GB0 is not complete' }
    foreach ($m in $names) {
        if ($State.cells.Contains("$m") -and $State.cells["$m"].result -ne 'timeout') { continue }
        & git -C $wt checkout -- . | Out-Host
        $code = Invoke-Capped $Python @((Join-Path $wt 'tests\schema-fast-decode-gpu\make_sfdg_mutants.py'), $m, "--repo=$wt", "--out=$wt") (Join-Path $Logs "$Step-$m-apply.log") $wt
        if ($code -ne 0) { & git -C $wt checkout -- . | Out-Host; throw "mutant $m did not apply ($code)" }
        $cell = $Mutants[$m]
        $exe = $CellTable[$cell][0]
        $b = Build-Cells $wt @($exe)
        & git -C $wt checkout -- . | Out-Host
        if ($null -eq $b) { Save-State $false $m; throw "TIMEOUT building mutant $m" }
        $ok = Invoke-Cell $m (Join-Path $b['_bin'] "$exe.exe") $CellTable[$cell][1] 'killed' 2
        if (-not $ok) { Save-State $false '(none: split the step)'; throw "TIMEOUT running mutant $m's cell" }
        $State.cells[$m].cell = $cell
    }
    Assert-Cells "GB-XG-$Group"
}
function Get-PotionLines {
    $pop = Join-Path $BoxDir 'potion-population-v1.jsonl'
    if (-not (Test-Path -LiteralPath $pop)) { throw "the potion population is not at $pop" }
    $lines = @()
    foreach ($l in Get-Content -LiteralPath $pop -Encoding UTF8) {
        if (-not $l.Trim()) { continue }
        $o = $l | ConvertFrom-Json
        $u = $o.customer_line
        if (-not $u) { $u = $o.utterance }
        if (-not $u) { $u = $o.text }
        $lines += $u
    }
    return , $lines
}
function Invoke-GB2c {
    $wt = Join-Path $WorktreeRoot "$Name-tip"
    $State.commit = Initialize-Worktree $wt $Tip
    Enter-DevShell
    $build = Join-Path $wt 'build-gb2c'
    if (-not (Test-Path -LiteralPath (Join-Path $build 'CMakeCache.txt'))) {
        & cmake -S $wt -B $build -G Ninja -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF -DSUPERSLM_BUILD_GPU=OFF | Out-Host
        if ($LASTEXITCODE -ne 0) { throw 'configure failed' }
    }
    & cmake --build $build --target sslm_schema_fast_decode_bench | Out-Host
    if ($LASTEXITCODE -ne 0) { throw 'the CPU bench did not build' }
    $bench = Join-Path $build 'sslm_schema_fast_decode_bench.exe'
    # B7b's 41 prompts: B2's prompt (one line, labels unused) and the 40 labelled potion prompts.
    $b2 = Join-Path $OutDir 'gb2c-b2.jsonl'
    Write-Utf8 $b2 ('{"id":"B2","customer_line":"Hello! Could I get two health potions, please?","intent":"-","item":"-","quantity":"-","polite":"-"}' + "`n")
    $pop = Join-Path $BoxDir 'potion-population-v1.jsonl'
    foreach ($part in @(@('b2', $b2, '1'), @('pop', $pop, '40'))) {
        $json = Join-Path $Out "GB2c-$($part[0]).json"
        $code = Invoke-Capped $bench @($Model, 'quality', 'potion_shop_order', '--population', $part[1], '--kind', 'potion', '--from', '0',
            '--count', $part[2], '--max-new', '64', '--budget', '64', '--json', $json) (Join-Path $Logs "GB2c-$($part[0]).log") $OutDir
        if ($code -ne 0) { throw "the CPU bench failed ($code) on $($part[0])" }
        $State["cpu_$($part[0])"] = [ordered]@{ path = $json; sha256 = (Get-Sha256 $json) }
    }
    $State.model_sha256 = Get-Sha256 $Model
}
function Read-CpuStreams {
    $off = @(); $can = @()
    foreach ($part in @('b2', 'pop')) {
        $j = Read-Utf8 (Join-Path $Out "GB2c-$part.json") | ConvertFrom-Json
        # The CPU bench's quality JSON: arms.off and arms.on (canonical at --budget), one item per prompt.
        $offRows = @($j.arms.off.items); $onRows = @($j.arms.on.items)
        $off += , @($offRows | ForEach-Object { , @($_.stream) })
        $can += , @($onRows | ForEach-Object { , @($_.stream) })
    }
    return @{ off = @($off[0] + $off[1]); canonical = @($can[0] + $can[1]) }
}
function Invoke-GB2([string]$Root, [string]$BenchExe) {
    if (-not $GbRules -or -not (Test-Path -LiteralPath $GbRules)) { throw 'REFUSED: GB2 runs only after the debunking seat freezes the GB rules file (-GbRules)' }
    $State.gb_rules_sha256 = Get-Sha256 $GbRules
    $utt = Join-Path $OutDir 'gb2-utterances.txt'
    Write-Utf8 $utt ((Get-PotionLines) -join "`n")
    $runs = @(@('device', @('off', 'fused:1', 'fused:2', 'fused:4')), @('host', @('off', 'fused:1')))
    foreach ($r in $runs) {
        $id = "streams-$($r[0])"
        if ($State.cells.Contains($id) -and $State.cells[$id].result -ne 'timeout') { continue }
        $json = Join-Path $Out "GB2-$($r[0]).json"
        $a = @($Model, '--streams', ($r[1] -join ','), '--utterances', $utt, '--max-new', '64', '--json', $json)
        if ($r[0] -eq 'device') { $a += '--device-head' }
        $env:SSLM_GPU_ENABLE_DEBUG_LAYER = '2'
        $code = Invoke-Capped $BenchExe $a (Join-Path $Logs "GB2-$($r[0]).log") (Split-Path -Parent $BenchExe)
        if ($code -eq 'timeout') { $State.cells[$id] = [ordered]@{ result = 'timeout' }; Save-State $false '(none: split the step)'; throw "TIMEOUT in $id" }
        if ($code -ne 0) { throw "the GPU bench failed ($code) in $id" }
        $State.cells[$id] = [ordered]@{ result = 'ran'; path = $json; sha256 = (Get-Sha256 $json) }
        Save-State $false ''
    }
    # The comparisons (the rules decide; the driver reports): GPU off against CPU off, every GPU canonical arm against
    # CPU canonical, prompt by prompt.
    $cpu = Read-CpuStreams
    $cmp = [ordered]@{}
    foreach ($r in $runs) {
        $g = Read-Utf8 (Join-Path $Out "GB2-$($r[0]).json") | ConvertFrom-Json
        foreach ($arm in $r[1]) {
            $want = $(if ($arm -eq 'off') { $cpu.off } else { $cpu.canonical })
            $diff = @()
            for ($i = 0; $i -lt $g.prompts.Count; ++$i) {
                $got = @(($g.prompts[$i].arms | Where-Object { $_.arm -eq $arm }).stream)
                if (($got -join ',') -ne (@($want[$i]) -join ',')) { $diff += $i }
            }
            $cmp["$($r[0])/$arm"] = [ordered]@{ prompts = $g.prompts.Count; differ = $diff.Count; differing_prompts = $diff }
        }
    }
    $State.comparisons = $cmp
}
function Invoke-GB3 {
    if (-not $GbRules -or -not (Test-Path -LiteralPath $GbRules)) { throw 'REFUSED: GB3 runs after the GB rules file is frozen (-GbRules)' }
    $State.gb_rules_sha256 = Get-Sha256 $GbRules
    $State.predictions = 'plan rev 1.4.4 Sec. 7.2 and Sec. 7.4, stated before this run'
    $wtTip = Join-Path $WorktreeRoot "$Name-tip"
    $wtBase = Join-Path $WorktreeRoot "$Name-base"
    $State.commit = Initialize-Worktree $wtTip $Tip
    $State.base_commit = Initialize-Worktree $wtBase $Base
    $tip = Build-Bench $wtTip 'tip'
    $base = Build-Bench $wtBase 'base'
    $State.bench = [ordered]@{ tip = $tip; tip_sha256 = (Get-Sha256 $tip); base = $base; base_sha256 = (Get-Sha256 $base) }
    Remove-Item Env:\SSLM_GPU_ENABLE_DEBUG_LAYER -ErrorAction SilentlyContinue  # value 0 (strike N3)
    foreach ($head in @('device', 'host')) {
        $h = $(if ($head -eq 'device') { @('--device-head') } else { @() })
        # Off today (-Base) against off with G1-M (-Tip): two binaries, so one process per arm per pair, ABBA.
        $id = "$head/off-base-vs-tip"
        if (-not ($State.cells.Contains($id) -and $State.cells[$id].result -eq 'ran')) {
            $ratios = @()
            for ($p = 0; $p -le $Pairs; ++$p) {
                $order = $(if ($p % 2 -eq 0) { @('base', 'tip') } else { @('tip', 'base') })
                $w = @{}
                foreach ($side in $order) {
                    $exe = $(if ($side -eq 'base') { $base } else { $tip })
                    $json = Join-Path $Logs "GB3-$head-p$p-$side.json"
                    $code = Invoke-Capped $exe (@($Model, '--pair', 'off,off', '--pairs', '1', '--warmup', '1', '--json', $json) + $h) "$json.log" (Split-Path -Parent $exe)
                    if ($code -eq 'timeout') { $State.cells[$id] = [ordered]@{ result = 'timeout' }; Save-State $false '(none: split the step)'; throw "TIMEOUT in $id" }
                    if ($code -ne 0) { throw "bench failed ($code) in $id" }
                    $w[$side] = (Read-Utf8 $json | ConvertFrom-Json).arms[0].query_wall_ms.median
                }
                if ($p -gt 0) { $ratios += ($w['base'] / $w['tip']) }
            }
            $s = @($ratios | Sort-Object)
            $State.cells[$id] = [ordered]@{ result = 'ran'; ratio_base_over_tip = $ratios; median = $s[[int][math]::Floor($s.Count / 2)]
                sign_interval = $(if ($s.Count -eq 15) { @($s[3], $s[11]) } else { @() }) }
            Save-State $false ''
        }
        foreach ($pair in @('on,fused:4', 'off,fused:1', 'off,fused:2')) {
            $id = "$head/$pair"
            if ($State.cells.Contains($id) -and $State.cells[$id].result -eq 'ran') { continue }
            $json = Join-Path $Out ("GB3-$head-$pair.json" -replace '[:,]', '_')
            $code = Invoke-Capped $tip (@($Model, '--pair', $pair, '--pairs', "$Pairs", '--warmup', '1', '--json', $json) + $h) (Join-Path $Logs ("GB3-$head-$pair.log" -replace '[:,]', '_')) (Split-Path -Parent $tip)
            if ($code -eq 'timeout') { $State.cells[$id] = [ordered]@{ result = 'timeout' }; Save-State $false '(none: split the step)'; throw "TIMEOUT in $id" }
            if ($code -ne 0) { throw "bench failed ($code) in $id" }
            $State.cells[$id] = [ordered]@{ result = 'ran'; path = $json; sha256 = (Get-Sha256 $json) }
            Save-State $false ''
        }
    }
    # GB3's digests against GB2's (review N5): the bench's per-arm digest of B2's prompt.
    $State.digests_note = 'compare each GB3 arm digest with GB2-<head>.json prompt 0 for the same arm'
}
function Invoke-GBK {
    if (-not $KitDir) { throw '-KitDir is required for GBK' }
    $g1 = Join-Path $Out 'GB1.json'
    if (-not (Test-Path -LiteralPath $g1) -or -not (Read-Utf8 $g1 | ConvertFrom-Json).complete) { throw 'GB1 is not complete' }
    $gb1 = Read-Utf8 $g1 | ConvertFrom-Json
    $bin = $gb1.build._bin
    New-Item -ItemType Directory -Force -Path $KitDir | Out-Null
    $kb = Join-Path $KitDir 'bin'
    New-Item -ItemType Directory -Force -Path $kb | Out-Null
    Copy-Item (Join-Path $bin '*.exe') $kb -Force
    Copy-Item (Join-Path $bin 'shaders') $kb -Recurse -Force
    $bench = (Read-Utf8 (Join-Path $Out 'GB0.json') | ConvertFrom-Json).bench
    Copy-Item $bench $kb -Force
    $kf = Join-Path $KitDir 'fixtures'
    New-Item -ItemType Directory -Force -Path $kf | Out-Null
    Copy-Item (Join-Path $Fixtures '*') $kf -Force
    $ka = Join-Path $KitDir 'artifacts'
    New-Item -ItemType Directory -Force -Path $ka | Out-Null
    Copy-Item $Model $ka -Force
    $kd = Join-Path $KitDir 'driver'
    New-Item -ItemType Directory -Force -Path $kd | Out-Null
    Copy-Item $PSCommandPath $kd -Force
    Copy-Item (Join-Path $PSScriptRoot 'run_box_step.ps1') $kd -Force
    Copy-Item (Join-Path $BoxDir 'potion-population-v1.jsonl') $kd -Force
    if ($GbRules) { Copy-Item $GbRules $kd -Force }
    foreach ($f in @('GB2c-b2.json', 'GB2c-pop.json')) { if (Test-Path -LiteralPath (Join-Path $Out $f)) { Copy-Item (Join-Path $Out $f) $kd -Force } }
    $readme = @(
        '# Schema fast-decode GPU kit (plan rev 1.4.4 Sec. 10, G-K)',
        "Packed $(Get-Utc) from GB1 at commit $($gb1.commit).",
        '',
        'Steps, in order, one per invocation, each under a 15-minute kill, with a pause of >= 5 min after each:',
        '  powershell -ExecutionPolicy Bypass -File driver\run_gpu_box_step.ps1 -Step GK0 -Kit <this directory> -OutDir <results>',
        '  ... -Step GK1   (the G cells at SSLM_GPU_ENABLE_DEBUG_LAYER=2; value 0 and recorded layer-off if Graphics Tools is absent)',
        '  ... -Step GK2   (GB2 on this card, against driver\GB2c-*.json; needs -GbRules driver\<rules file>)',
        'GKV (GPU-based validation, optional, gates nothing) is not driven by this script.',
        'File the results directory beside the 2080 SUPER results. manifest.txt holds every file''s sha256.')
    Write-Utf8 (Join-Path $KitDir 'README.md') ($readme -join "`r`n")
    $manifest = Get-ChildItem -LiteralPath $KitDir -Recurse -File | Where-Object { $_.Name -ne 'manifest.txt' } |
        ForEach-Object { "$(Get-Sha256 $_.FullName)  $($_.FullName.Substring($KitDir.Length).TrimStart('\'))" }
    Write-Utf8 (Join-Path $KitDir 'manifest.txt') (($manifest -join "`r`n") + "`r`n")
    $State.kit = [ordered]@{ dir = $KitDir; files = $manifest.Count; manifest_sha256 = (Get-Sha256 (Join-Path $KitDir 'manifest.txt')) }
}
function Invoke-GK([string]$S) {
    if (-not $Kit) { throw "-Kit (the kit's directory) is required for $S" }
    $script:Fixtures = Join-Path $Kit 'fixtures'
    $script:FxArg = "--fixture-dir=$Fixtures"
    foreach ($k in @($CellTable.Keys)) { $CellTable[$k] = @($CellTable[$k][0], @($CellTable[$k][1] | ForEach-Object { if ($_ -like '--fixture-dir=*') { $FxArg } else { $_ } })) }
    $bin = Join-Path $Kit 'bin'
    $State.kit_manifest_sha256 = Get-Sha256 (Join-Path $Kit 'manifest.txt')
    switch ($S) {
        'GK0' { Invoke-CellList @('smoke') $bin 'green' 2; $State.baseline_ids = @($State.cells['smoke'].debug_ids) }
        'GK1' {
            if ($null -eq (Read-Baseline)) { throw 'GK0 is not complete: its debug-layer baseline is needed first' }
            Invoke-CellList (@('G0.1') + $SliceCells.F + $SliceCells.M + $SliceCells.J) $bin 'green' 2
        }
        'GK2' {
            $script:Model = (Get-ChildItem (Join-Path $Kit 'artifacts\*.sslm') | Select-Object -First 1).FullName
            $script:BoxDir = Join-Path $Kit 'driver'
            foreach ($f in @('GB2c-b2.json', 'GB2c-pop.json')) { Copy-Item (Join-Path $Kit "driver\$f") $Out -Force }
            Invoke-GB2 $Kit (Join-Path $bin 'sslm_schema_fast_decode_gpu_jf_bench.exe')
        }
    }
    if ($S -ne 'GK2') { Assert-Cells $S }
}

# ---- run ----------------------------------------------------------------------------------------------------------
$since = 0
if ($IsGpuStep) { $since = Assert-Ledger }
Write-Host "== $Step (cap $CapMinutes min)"
try {
    switch -Regex ($Step) {
        '^GB0$' { Invoke-GB0 }
        '^GB-([FMJ])-([rg])$' { Invoke-Slice $Matches[1] $Matches[2] }
        '^GB1$' { Invoke-GB1 }
        '^GB-XG-(\d)$' { Invoke-GBXG ([int]$Matches[1]) }
        '^GBK$' { Invoke-GBK }
        '^GB2c$' { Invoke-GB2c }
        '^GB2$' {
            $wt = Join-Path $WorktreeRoot "$Name-tip"
            $State.commit = Initialize-Worktree $wt $Tip
            Invoke-GB2 $wt (Build-Bench $wt 'tip')
        }
        '^GB3$' { Invoke-GB3 }
        '^GK[012]$' { Invoke-GK $Step }
    }
    Save-State $true ''
    Write-Host "$Step complete: $StepPath"
} finally {
    if ($IsGpuStep) { Update-Ledger $since }
}
