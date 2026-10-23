# E4 timing pieces (T1b, B8a-B8d): one 10.3/E4 slice, alone, with the U3 PAUSE file and the T1 re-run rule's evidence
# (PAUSE present at start and end; per-process CPU seconds before and after, excluding the runner and idle).
#   powershell -File timed.ps1 -Name <log name> -Slice 0-1 -Runner <exe> -Receipt <receipt file>
param([string]$Name, [string]$Slice, [string]$Runner, [string]$Receipt)
. D:\_scratch\e4\src\tests\precache\e4_box_env.ps1
$pause = 'D:\_scratch\TE-444\u3\PAUSE'
$log = E4-Log $Name
$env:SUPERSLM_PRECACHE_TIMING_PREFIXES = $Slice
$env:SUPERSLM_PRECACHE_TIMING_RECEIPT = $Receipt
function Snap { $h = @{}; Get-Process | ForEach-Object { if ($_.CPU -ne $null) { $h['{0}:{1}' -f $_.ProcessName, $_.Id] = [double]$_.CPU } }; return ,$h }
E4-Append $log ('pause present at start: {0}' -f (Test-Path -LiteralPath $pause))
E4-Append $log ('receipt {0} sha256 {1}' -f $Receipt, (Get-FileHash $Receipt).Hash.ToLower())
$before = Snap
E4-Run $log $Runner @('=10.3/E4') | Out-Null
$after = Snap
E4-Append $log ('pause present at end: {0}' -f (Test-Path -LiteralPath $pause))
$runnerName = [IO.Path]::GetFileNameWithoutExtension($Runner)
$busy = foreach ($k in $after.Keys) {
    $p = $k.Split(':')[0]
    if ($p -eq $runnerName -or $p -eq 'Idle' -or $k -eq ('powershell:{0}' -f $PID)) { continue }
    $d = $after[$k] - $(if ($before.ContainsKey($k)) { $before[$k] } else { 0 })
    if ($d -gt 5) { '{0} {1:N1}s' -f $k, $d }
}
E4-Append $log ('other processes over 5 CPU-s during the slice: {0}' -f $(if ($busy) { $busy -join '; ' } else { 'none' }))
Get-Content -LiteralPath $log | Select-String -Pattern '^10\.3|10\.3/E4|MUTANT|pause present|other processes|exit \d+$|FAIL|failed'
