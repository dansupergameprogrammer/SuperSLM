. D:\_scratch\e4\src\tests\precache\e4_box_env.ps1
$w = Split-Path $env:SUPERSLM_PRECACHE_WORST_BLOB          # D:\_scratch\e4\worst
New-Item -ItemType Directory -Force $w | Out-Null
Copy-Item F:\longleg\results\e4-w1\* $w -Force
$log = E4-Log 'B9-worst-restore'
Get-Content (Join-Path $w 'W1.txt') | ForEach-Object { E4-Append $log $_; $_ }
$want = ((Get-Content (Join-Path $w 'cap1p5.spx1.sha256')) -split '\s+')[0]
$got = (Get-FileHash $env:SUPERSLM_PRECACHE_WORST_BLOB).Hash.ToLower()
"blob sha256 $got; W1 printed $want; bytes $((Get-Item $env:SUPERSLM_PRECACHE_WORST_BLOB).Length)" | ForEach-Object { E4-Append $log $_; $_ }
if ($got -ne $want) { throw 'cap1p5.spx1 differs from what W1 recorded: copy it again' }
$env:SUPERSLM_PRECACHE_WORST_PHASE = 'restore'
E4-Run $log $E4Runner @('=10.3/E4:worst')
