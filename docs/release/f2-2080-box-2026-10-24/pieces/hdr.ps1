# F2 GPU twins on the home box's RTX 2080 SUPER (engine 1.16.0 line, SuperSLM main 5599f7e4): what every piece
# dot-sources. MSVC 14.34 and the SDK come from the E4 box environment (it sets session state only); the paths are
# this run's. Each piece writes its log under logs\ (UTF-8, no BOM) and a <piece>.done marker when it completed, so a
# crashed piece is rerun as is and a done one is skipped (-Force reruns it).
. D:\_scratch\e5-1p14\src\tests\precache\e4_box_env.ps1
$global:F2      = 'D:\_scratch\f2-2080'
$global:F2Src   = 'D:\SuperSLM\.worktrees\f2-2080'
$global:F2Logs  = Join-Path $F2 'logs'
$global:F2Out   = Join-Path $F2 'pkvg'
$global:F2Bin   = Join-Path $F2Out 'bin'
$global:F2FX    = 'D:\_artifacts\SuperSLM\t3035-precache-refs\fixtures'
$global:F2E0    = 'D:\_artifacts\SuperSLM\t3030-commission'
$global:F2M05   = 'D:\_artifacts\SuperSLM\te442-qwen2.5-0.5b-instruct-dg\qwen2.5-0.5b-instruct-dg.sslm'
$global:F2M15   = 'D:\_artifacts\SuperSLM\t2901-qwen2.5-1.5b-dg\qwen2.5-1.5b-instruct-dg-g5.sslm'
$global:F2Box   = 'D:\Wizard\Claude\Tooling\rel10-box.ps1'
New-Item -ItemType Directory -Force $F2Logs | Out-Null
function global:F2-Log([string]$n) { Join-Path $F2Logs "$n.log" }
function global:F2-Done([string]$n) { Test-Path (Join-Path $F2 "$n.done") }
function global:F2-Mark([string]$n, [string]$v) { Set-Content (Join-Path $F2 "$n.done") "$v $(Get-Date -Format s)" }
# The shared box lock (the UE build lock), so no Wizard build or editor run shares the GPU with a piece.
function global:F2-Lock {
    $env:SUPERSLM_BUILD_LOCK = 'D:\_ssu_build_lock'
    $out = & pwsh -NoProfile -File $F2Box -Action Lock -DriverPid $PID 2>&1 | ForEach-Object { "$_" }
    if ($LASTEXITCODE -ne 0) { throw "box lock not taken (exit $LASTEXITCODE): $($out -join ' | ')" }
    $env:SSU_BUILD_LOCK_TOKEN = (($out | Where-Object { $_ -match '^SSU_BUILD_LOCK_TOKEN=' } | Select-Object -First 1) -split '=', 2)[1]
}
function global:F2-Unlock { & pwsh -NoProfile -File $F2Box -Action Unlock 2>&1 | Out-Null }
function global:F2-Busy {
    $busy = @(Get-Process -Name 'UnrealEditor', 'UnrealEditor-Cmd', 'UnrealBuildTool', 'ExampleProject' -ErrorAction SilentlyContinue)
    if ($busy.Count -gt 0) { throw "box not clear: $(($busy | ForEach-Object { "$($_.ProcessName)#$($_.Id)" }) -join ', ')" }
}
# Runs one cell executable with a time bound, logging its output and exit code (the last line of the log).
function global:F2-Cell([string]$Log, [string]$Exe, [string[]]$Arguments, [int]$Minutes = 20) {
    E4-Append $Log ('[{0}] {1} {2}' -f (Get-Date -Format 's'), $Exe, ($Arguments -join ' '))
    $o = "$Log.out"; $e = "$Log.err"
    $p = Start-Process -FilePath $Exe -ArgumentList $Arguments -WorkingDirectory $F2Src -NoNewWindow -PassThru -RedirectStandardOutput $o -RedirectStandardError $e
    $null = $p.Handle
    if (-not $p.WaitForExit($Minutes * 60000)) { $p.Kill($true); E4-Append $Log "TIMEOUT after $Minutes min: killed"; return 124 }
    foreach ($f in @($o, $e)) { if (Test-Path $f) { Get-Content $f | ForEach-Object { E4-Append $Log "$_" }; Remove-Item $f } }
    E4-Append $Log ('[{0}] exit {1}' -f (Get-Date -Format 's'), $p.ExitCode)
    return $p.ExitCode
}
# The exit code E4-Run appended last to Log ("[time] exit N").
function global:F2-LastExit([string]$Log) {
    $l = Get-Content -LiteralPath $Log | Where-Object { $_ -match '^\[[^\]]+\] exit (-?\d+)$' } | Select-Object -Last 1
    if ($l -match 'exit (-?\d+)$') { return [int]$Matches[1] }
    return -1
}
