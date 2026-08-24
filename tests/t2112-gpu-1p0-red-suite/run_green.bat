@echo off
rem T-2243 review finding 5 (D-SLM4113): the committed run recipe for this suite -- t2138's own
rem run_green.bat pattern (env-var-supplied model/adapter paths, SKIP when a cell's own real-
rem artifact argument is absent, one loop over the file list build_link_red.bat already
rem maintains). Before this file, s2_bind_red.cpp / cell_rebind_serial.cpp / dim8_composition_red.cpp
rem had no committed way to run with a real model and two real adapters, which left plan Sec9's
rem release gate item 4 (T-2244, every GPU-touching red cell re-run on the AMD RX 7900 XTX)
rem unexecutable as specified. `run_c1_o2_cells.bat` is untouched -- it self-describes as ad hoc,
rem session-scoped execution evidence with a hardcoded path and is not this recipe's replacement,
rem only its narrower, pre-existing sibling.
rem
rem Env vars (fixture_common.h's own argv convention, --flag=value on the built .exe):
rem   T2112_MODEL       -> --model1p5b=PATH   (required for any cell to do real work; absent =>
rem                        every real-artifact cell SKIPs, matching t2138's own required-arg
rem                        convention is NOT enforced here, since dim1-dim6 exercise host-only/
rem                        hostile-input paths that need no real model at all)
rem   T2112_MODEL_0P5B  -> --model0p5b=PATH   (optional, cross-model cells)
rem   T2112_MODEL_VARIANT -> --model1p5bvariant=PATH (optional)
rem   T2112_ADAPTER     -> --adapter=PATH     (optional, single-adapter cells)
rem   T2112_ADAPTER2    -> --adapter2=PATH    (optional, S2's own two-real-adapter rebind/
rem                        composition/serial-switching cells -- absent => their two-adapter
rem                        product half SKIPs, per fixture_common.h's own documented convention)
setlocal enabledelayedexpansion
set HEREDIR=%~dp0

call "%HEREDIR%build_link_red.bat"
if errorlevel 1 exit /b 1

set ARGS=
if defined T2112_MODEL set ARGS=!ARGS! --model1p5b=!T2112_MODEL!
if defined T2112_MODEL_0P5B set ARGS=!ARGS! --model0p5b=!T2112_MODEL_0P5B!
if defined T2112_MODEL_VARIANT set ARGS=!ARGS! --model1p5bvariant=!T2112_MODEL_VARIANT!
if defined T2112_ADAPTER set ARGS=!ARGS! --adapter=!T2112_ADAPTER!
if defined T2112_ADAPTER2 set ARGS=!ARGS! --adapter2=!T2112_ADAPTER2!

set OVERALL_OK=1
for %%f in (dim1_lifetime_red dim2_hostile_red dim3_concurrency_red dim4_shape_red dim5_failure_red dim6_determinism_red dim8_composition_red dim9_persistence_red dim10_functional_red dim11_guard_red s2_bind_red cell_rebind_serial) do (
    echo ===== %%f.cpp execution =====
    "%HEREDIR%obj\%%f.exe" !ARGS! > "%HEREDIR%obj\%%f.runlog" 2>&1
    set RUN_EC=!errorlevel!
    type "%HEREDIR%obj\%%f.runlog"
    set SUMMARY_LINE=
    for /f "delims=" %%s in ('findstr /R "^checks=[0-9]* failures=[0-9]*" "%HEREDIR%obj\%%f.runlog"') do set SUMMARY_LINE=%%s
    if "!SUMMARY_LINE!"=="" (
        echo T-2112 %%f produced no summary ^(exit !RUN_EC!^)
        set OVERALL_OK=0
    ) else (
        echo !SUMMARY_LINE! | findstr /R "failures=0" >nul
        if errorlevel 1 set OVERALL_OK=0
        if not "!RUN_EC!"=="0" set OVERALL_OK=0
    )
)

if "!OVERALL_OK!"=="1" (
    echo T-2112 SUITE STATUS: EXECUTED GREEN.
    exit /b 0
)
echo T-2112 SUITE STATUS: EXECUTION FAILURE -- see obj\*.runlog.
exit /b 1
