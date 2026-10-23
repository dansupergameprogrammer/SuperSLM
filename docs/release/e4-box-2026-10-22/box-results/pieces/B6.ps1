. D:\_scratch\e4\src\tests\precache\e4_box_env.ps1
Remove-Item -Recurse -Force $env:SUPERSLM_PRECACHE_BLOB_DIR -ErrorAction SilentlyContinue
Push-Location $E4Build
E4-Run (E4-Log 'B6-two-process') ctest @('-R', '^superslm_precache_real_two_process$', '-V', '--timeout', '1500')
Pop-Location
