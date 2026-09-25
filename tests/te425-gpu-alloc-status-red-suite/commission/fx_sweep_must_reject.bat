@echo off
rem TE-436 must-reject for the foreign-exception sweep (te436_foreign_sweep.cpp). Every leg is a
rem construction another seat found and executed, never one the sweep's builder chose:
rem   - the code reviewer's TE-431 S-1 legs: the per-token decode step's two finish-phase sites (after
rem     the fence wait, and after the readback Map; te425_cells r10dec k=26 and k=27) at c3b5412, and the
rem     first of them at v1.7.1. The finish lets a non-standard exception escape, the handle stays
rem     Submitted with a freed in-flight token, and its reset is refused;
rem   - the builder's TE-432 tail finding: a decode-step submission tail with no catch (...), here the
rem     v1.7.1 tail, where a foreign exception at the in-flight token's allocation (the first allocation
rem     after the step's Signal) leaves the call while its submission is still outstanding (the gated leg).
rem Sites are named by what they are (--select), not by number: the engine makes allocations that
rem depend on call history, which moves site numbers. One site per process: each of these legs leaves
rem the handle or the device unusable for the rest of its process.
rem Usage: fx_sweep_must_reject.bat <suite-bin-at-c3b5412> <suite-bin-at-v1.7.1> <qwen3-artifact>
rem   each bin: a build_red_suite.bat output directory built against that engine's seam build.
rem Exits 1 only when every leg returned the sweep's RED verdict (exit 1). Any other exit -- a crash, a
rem setup failure, an undriven leg, a green leg -- exits 0, so nothing but a real rejection reads as one.
setlocal
if "%~3"=="" (echo usage: fx_sweep_must_reject.bat ^<bin-c3b5412^> ^<bin-v1.7.1^> ^<qwen3^> & exit /b 0)
set ALL=1
pushd "%~1"
.\te436_foreign_sweep.exe --route=step --select=phase:ready --pick=0 "--qwen3=%~3"
if not "%ERRORLEVEL%"=="1" set ALL=0
.\te436_foreign_sweep.exe --route=step --select=phase:ready --pick=1 "--qwen3=%~3"
if not "%ERRORLEVEL%"=="1" set ALL=0
popd
pushd "%~2"
.\te436_foreign_sweep.exe --route=step --select=phase:ready --pick=0 "--qwen3=%~3"
if not "%ERRORLEVEL%"=="1" set ALL=0
.\te436_foreign_sweep.exe --route=step --select=outstanding --pick=0 "--qwen3=%~3"
if not "%ERRORLEVEL%"=="1" set ALL=0
popd
if "%ALL%"=="1" (echo MUST-REJECT: every leg RED & exit /b 1)
echo MUST-REJECT: not every leg RED
exit /b 0