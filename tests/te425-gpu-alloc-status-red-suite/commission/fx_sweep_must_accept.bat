@echo off
rem TE-436 must-accept for the foreign-exception sweep (te436_foreign_sweep.cpp). Every leg is a
rem construction another seat found healthy, never one the sweep's builder chose:
rem   - the code reviewer's TE-431 prompt-route legs at c3b5412: the final sub-chunk's two finish sites of
rem     a 5-token prompt (after the fence wait, and after the readback Map; te425_cells r10 k=96 and k=97),
rem     the last two sites of the prefill phase, where the escaping exception leaves the handle Idle and
rem     every next call clean;
rem   - the builder's TE-432 fix at c3b5412: both submission tails carry a catch (...) that waits the
rem     submission out (every outstanding site of the decode step, and of both sub-chunks of a 5-token
rem     prompt), the construction TE433-TAIL-PIN was commissioned against as healthy (the gated legs).
rem Sites are named by what they are (--select), not by number.
rem Usage: fx_sweep_must_accept.bat <suite-bin-at-c3b5412> <qwen3-artifact>
rem Exits 0 only when every leg returned the sweep's GREEN verdict (exit 0); any other exit exits 1.
setlocal
if "%~2"=="" (echo usage: fx_sweep_must_accept.bat ^<bin-c3b5412^> ^<qwen3^> & exit /b 1)
set ALL=1
pushd "%~1"
.\te436_foreign_sweep.exe --route=prompt --select=phase:prefill --pick=-2 "--qwen3=%~2"
if not "%ERRORLEVEL%"=="0" set ALL=0
.\te436_foreign_sweep.exe --route=prompt --select=phase:prefill --pick=-1 "--qwen3=%~2"
if not "%ERRORLEVEL%"=="0" set ALL=0
.\te436_foreign_sweep.exe --route=step --select=outstanding "--qwen3=%~2"
if not "%ERRORLEVEL%"=="0" set ALL=0
.\te436_foreign_sweep.exe --route=prompt --select=outstanding "--qwen3=%~2"
if not "%ERRORLEVEL%"=="0" set ALL=0
popd
if "%ALL%"=="1" (echo MUST-ACCEPT: every leg GREEN & exit /b 0)
echo MUST-ACCEPT: not every leg GREEN
exit /b 1