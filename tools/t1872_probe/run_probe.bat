@echo off
REM T-1872 portable feasibility probe -- one entry point.
REM Double-click this file, or run it from a command prompt.
REM This extracts T1872_Probe.tar to a temporary local-disk folder, runs the
REM probe there, then deletes that temporary folder automatically -- see
REM run_probe.ps1 for the details and the cleanup guarantee.
setlocal
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0run_probe.ps1"
echo.
echo ============================================================
echo Done. Results are in RESULTS.txt in this same folder (on the drive).
echo ============================================================
pause
