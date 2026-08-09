@echo off
REM T-1872 portable feasibility probe -- one entry point.
REM Double-click this file, or run it from a command prompt. It reads and
REM writes only inside this bundle folder; it does not install anything on
REM this machine, does not touch PATH, the registry, or any driver, and
REM leaves nothing behind once the bundle folder is deleted.
setlocal
set BUNDLE=%~dp0
"%BUNDLE%python\python.exe" "%BUNDLE%probe\orchestrator.py"
echo.
echo ============================================================
echo Done. Results are in RESULTS.txt in this same folder.
echo ============================================================
pause
