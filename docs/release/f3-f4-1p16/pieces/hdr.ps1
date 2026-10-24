# F3 certification and 9.5 [F4] on the home box's RTX 2080 SUPER (engine 1.16 line, branch claude/f3-f4-cells-1p16):
# what run.ps1 dot-sources. MSVC 14.34 and the SDK come from the E4 box environment (session state only), as F2's
# pieces did. Each piece writes logs\<piece>.log (UTF-8, no BOM) and logs\<piece>.done ("PASS ..." or "FAIL ...") when
# it completed, so a crashed piece is rerun as is and a done one is skipped (-Force reruns it).
#
# Box rules (2026-10-24): every GPU piece is bounded at 15 minutes by its own stopwatch, starts at least 5 minutes after
# the previous GPU piece ended (gpu-last.txt), never sets SSLM_GPU_ENABLE_DEBUG_LAYER (it turns on GPU-based
# validation, which this box does not run), takes the box lock, and refuses to start while an editor or a build runs.
. D:\_scratch\e5-1p14\src\tests\precache\e4_box_env.ps1
$global:F3       = 'D:\_scratch\f3-f4-1p16'
$global:F3Src    = 'D:\SuperSLM\.worktrees\f3-f4-1p16'
$global:F3Logs   = Join-Path $F3 'logs'
$global:F3Out    = Join-Path $F3 'pkvg'
$global:F3Bin    = Join-Path $F3Out 'bin'
$global:F3Mut    = Join-Path $F3 'mutants'
$global:F3Cmake  = Join-Path $F3 'build-msvc'
$global:F3Blobs  = Join-Path $F3 'blobs'          # 4.6's saves and 10.4's two-process blobs
$global:F3FX     = 'D:\_artifacts\SuperSLM\t3035-precache-refs\fixtures'
$global:F3E0     = 'D:\_artifacts\SuperSLM\t3030-commission'
$global:F3M05    = 'D:\_artifacts\SuperSLM\te442-qwen2.5-0.5b-instruct-dg\qwen2.5-0.5b-instruct-dg.sslm'
$global:F3M15    = 'D:\_artifacts\SuperSLM\t2901-qwen2.5-1.5b-dg\qwen2.5-1.5b-instruct-dg-g5.sslm'
$global:F3Prompts = 'D:\_scratch\e4\t3035\tests\t3035-precache-refs\prompts'   # origin/curie/t3035-precache-refs
$global:F3Box    = 'D:\Wizard\Claude\Tooling\rel10-box.ps1'
$global:F3Py     = 'C:\Users\dansu\AppData\Local\Programs\Python\Python312\python.exe'
$global:F3Cells  = '77cbac40'                     # the commit whose seams the F3 cells link against
$global:F3GpuMinutes = 15
$global:F3PauseMinutes = 5
foreach ($d in @($F3, $F3Logs, $F3Blobs)) { New-Item -ItemType Directory -Force $d | Out-Null }
$env:SSLM_GPU_ENABLE_DEBUG_LAYER = $null
$env:SUPERSLM_PAGED_KV_FIXTURE_DIR = $F3FX

function global:F3-Log([string]$n) { Join-Path $F3Logs "$n.log" }
function global:F3-Done([string]$n) { Test-Path (Join-Path $F3Logs "$n.done") }
function global:F3-Mark([string]$n, [string]$v) { Set-Content (Join-Path $F3Logs "$n.done") "$v $(Get-Date -Format s)" }

# The shared box lock (the UE build lock), so no Wizard build or editor run shares the GPU with a piece.
function global:F3-Lock {
    $env:SUPERSLM_BUILD_LOCK = 'D:\_ssu_build_lock'
    $out = & pwsh -NoProfile -File $F3Box -Action Lock -DriverPid $PID 2>&1 | ForEach-Object { "$_" }
    if ($LASTEXITCODE -ne 0) { throw "box lock not taken (exit $LASTEXITCODE): $($out -join ' | ')" }
    $env:SSU_BUILD_LOCK_TOKEN = (($out | Where-Object { $_ -match '^SSU_BUILD_LOCK_TOKEN=' } | Select-Object -First 1) -split '=', 2)[1]
}
function global:F3-Unlock { & pwsh -NoProfile -File $F3Box -Action Unlock 2>&1 | Out-Null }
function global:F3-Busy {
    $busy = @(Get-Process -Name 'UnrealEditor', 'UnrealEditor-Cmd', 'UnrealBuildTool', 'ExampleProject' -ErrorAction SilentlyContinue)
    if ($busy.Count -gt 0) { throw "box not clear: $(($busy | ForEach-Object { "$($_.ProcessName)#$($_.Id)" }) -join ', ')" }
}

# The RTX 2080 SUPER by its ids (vendor 10de, at least 4 GB), never by a fixed index: an integrated GPU, if one is ever
# enabled, enumerates too. adapters.exe is the work-machine kit's lister, built by the build piece.
function global:F3-Adapter([string]$Log) {
    $lines = @(& (Join-Path $F3 'adapters.exe') | ForEach-Object { "$_" })
    foreach ($l in $lines) { E4-Append $Log "adapter: $l" }
    $hit = $lines | Where-Object { $_ -match '^(\d+) 10de ([0-9a-f]{4}) (\d+) ' -and [int64]$Matches[3] -ge 4096 } | Select-Object -First 1
    if (-not $hit) { throw 'no NVIDIA adapter with at least 4 GB' }
    $null = $hit -match '^(\d+) 10de ([0-9a-f]{4}) '
    $env:SSLM_GPU_ADAPTER_INDEX = $Matches[1]
    E4-Append $Log "using adapter $($Matches[1]) (10de:$($Matches[2]))"
    return "10de_$($Matches[2])"
}

# The 5-minute rest between GPU pieces: gpu-last.txt holds the end time of the last one.
function global:F3-Rest([string]$Log) {
    $f = Join-Path $F3 'gpu-last.txt'
    if (-not (Test-Path $f)) { return }
    $last = [datetime]::Parse((Get-Content $f | Select-Object -First 1))
    $wait = $last.AddMinutes($F3PauseMinutes) - (Get-Date)
    if ($wait.TotalSeconds -gt 0) {
        E4-Append $Log ('rest: {0:N0} s after the previous GPU piece' -f $wait.TotalSeconds)
        Write-Host ('resting {0:N0} s (5 min between GPU pieces)' -f $wait.TotalSeconds)
        Start-Sleep -Seconds ([int][math]::Ceiling($wait.TotalSeconds))
    }
}
function global:F3-RestStamp { Set-Content (Join-Path $F3 'gpu-last.txt') (Get-Date -Format 'o') }

# Runs one executable with a time bound (the smaller of $Minutes and what is left of the piece's stopwatch), logging
# its output and exit code. 124 means it was killed at the bound.
$global:F3Deadline = $null
function global:F3-Cell([string]$Log, [string]$Exe, [string[]]$Arguments, [int]$Minutes = 15) {
    $ms = $Minutes * 60000
    if ($F3Deadline) {
        $left = [int](($F3Deadline - (Get-Date)).TotalMilliseconds)
        if ($left -lt $ms) { $ms = $left }
        if ($ms -le 0) { E4-Append $Log "SKIPPED $Exe : the piece's 15-minute stopwatch has run out"; return 125 }
    }
    E4-Append $Log ('[{0}] {1} {2}' -f (Get-Date -Format 's'), $Exe, ($Arguments -join ' '))
    $o = "$Log.out"; $e = "$Log.err"
    $p = Start-Process -FilePath $Exe -ArgumentList $Arguments -WorkingDirectory $F3Src -NoNewWindow -PassThru -RedirectStandardOutput $o -RedirectStandardError $e
    $null = $p.Handle
    $code = 0
    if (-not $p.WaitForExit($ms)) { $p.Kill($true); $code = 124 } else { $code = $p.ExitCode }
    foreach ($f in @($o, $e)) { if (Test-Path $f) { Get-Content $f | ForEach-Object { E4-Append $Log "$_" }; Remove-Item $f } }
    if ($code -eq 124) { E4-Append $Log ('TIMEOUT after {0:N1} min: killed' -f ($ms / 60000)) }
    E4-Append $Log ('[{0}] exit {1}' -f (Get-Date -Format 's'), $code)
    return $code
}
function global:F3-LastExit([string]$Log) {
    $l = Get-Content -LiteralPath $Log | Where-Object { $_ -match '^\[[^\]]+\] exit (-?\d+)$' } | Select-Object -Last 1
    if ($l -match 'exit (-?\d+)$') { return [int]$Matches[1] }
    return -1
}
