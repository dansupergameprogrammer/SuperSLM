@echo off
rem TE-436 must-accept for the foreign-exception sweep (te436_foreign_sweep.cpp). Every leg is a
rem construction another seat found healthy, never one the sweep's builder chose:
rem   - the code reviewer's TE-431 prompt-route legs at c3b5412: the final sub-chunk's two finish sites
rem     of a 5-token prompt (k=96, k=97), where the escaping exception leaves the handle Idle and every
rem     next call clean;
rem   - the builder's TE-432 fix at c3b5412: both submission tails carry a catch (...) that waits the
rem     submission out (the decode step's tail, and both sub-chunk tails of a 5-token prompt), the
rem     construction TE433-TAIL-PIN was commissioned against as healthy (the gated legs).
rem Usage: fx_sweep_must_accept.bat <suite-bin-at-c3b5412> <qwen3-artifact>
rem Exits 0 only when every leg returned the sweep's GREEN verdict (exit 0); any other exit exits 1.
setlocal
if "%~2"=="" (echo usage: fx_sweep_must_accept.bat ^<bin-c3b5412^> ^<qwen3^> & exit /b 1)
set ALL=1
pushd "%~1"
.\te436_foreign_sweep.exe --route=prompt --site=97 --expect-phase=prefill "--qwen3=%~2"
if not "%ERRORLEVEL%"=="0" set ALL=0
.\te436_foreign_sweep.exe --route=prompt --site=98 --expect-phase=prefill "--qwen3=%~2"
if not "%ERRORLEVEL%"=="0" set ALL=0
.\te436_foreign_sweep.exe --route=step --select=outstanding "--qwen3=%~2"
if not "%ERRORLEVEL%"=="0" set ALL=0
.\te436_foreign_sweep.exe --route=prompt --select=outstanding "--qwen3=%~2"
if not "%ERRORLEVEL%"=="0" set ALL=0
popd
if "%ALL%"=="1" (echo MUST-ACCEPT: every leg GREEN & exit /b 0)
echo MUST-ACCEPT: not every leg GREEN
exit /b 1
