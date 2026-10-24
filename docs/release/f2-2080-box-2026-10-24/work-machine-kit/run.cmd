@echo off
rem F2 GPU twins kit for the RX 7900 XTX. Double-click; leave it running; rerun after a crash (finished steps are skipped).
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0run.ps1" %*
echo.
pause
