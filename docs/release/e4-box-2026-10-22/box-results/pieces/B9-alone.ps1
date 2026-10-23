New-Item -ItemType File -Force -Path 'D:\_scratch\TE-444\u3\PAUSE' | Out-Null
"pause present at start: " + (Test-Path 'D:\_scratch\TE-444\u3\PAUSE')
& powershell.exe -NoProfile -ExecutionPolicy Bypass -File 'D:\_scratch\e4\pieces\B9.ps1'
"B9 exit: $LASTEXITCODE"
"pause present at end: " + (Test-Path 'D:\_scratch\TE-444\u3\PAUSE')
Remove-Item -LiteralPath 'D:\_scratch\TE-444\u3\PAUSE' -Force
"pause removed: " + (-not (Test-Path 'D:\_scratch\TE-444\u3\PAUSE'))
