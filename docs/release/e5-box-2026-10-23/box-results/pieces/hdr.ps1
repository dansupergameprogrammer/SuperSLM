. D:\_scratch\e5-1p14\src\tests\precache\e4_box_env.ps1
$E5 = 'D:\_scratch\e5-1p14'; $E5Src = "$E5\src"; $E5Logs = "$E5\logs"; New-Item -ItemType Directory -Force $E5Logs | Out-Null
$FX = 'D:\_artifacts\SuperSLM\t3035-precache-refs\fixtures'
function E5-Log([string]$n) { Join-Path $E5Logs "$n.log" }
