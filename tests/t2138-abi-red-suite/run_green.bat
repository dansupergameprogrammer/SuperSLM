@echo off
rem Builds and executes the shipped Layer-1 CPU ABI suite. The first argument is the required
rem hermetic base model. Optional T2138_ADAPTER, T2138_FOREIGN_MODEL, T2138_MODEL_VARIANT, and
rem T2138_MODEL_TOK variables activate cells whose real-artifact contracts need those shapes.
setlocal enabledelayedexpansion
set HEREDIR=%~dp0
if "%~1"=="" (
    echo T-2138 run_green: model path required
    exit /b 2
)

call "%HEREDIR%build_link_red.bat"
if errorlevel 1 exit /b 1

set ARGS=--model=%~1
if defined T2138_ADAPTER set ARGS=!ARGS! --adapter=!T2138_ADAPTER!
if defined T2138_FOREIGN_MODEL set ARGS=!ARGS! --foreignmodel=!T2138_FOREIGN_MODEL!
if defined T2138_MODEL_VARIANT set ARGS=!ARGS! --modelvariant=!T2138_MODEL_VARIANT!
if defined T2138_MODEL_TOK set ARGS=!ARGS! --modeltok=!T2138_MODEL_TOK!

set OVERALL_OK=1
for %%f in (dim1_lifetime_red dim2_hostile_red dim3_concurrency_red dim4_shape_red dim5_failure_red dim6_determinism_red dim7_contract_red dim8_composition_red dim9_persistence_red dim10_functional_red dim11_guard_red) do (
    echo ===== %%f.cpp execution =====
    "%HEREDIR%obj\%%f.exe" !ARGS! > "%HEREDIR%obj\%%f.runlog" 2>&1
    set RUN_EC=!errorlevel!
    type "%HEREDIR%obj\%%f.runlog"
    set SUMMARY_LINE=
    for /f "delims=" %%s in ('findstr /R "^checks=[0-9]* failures=[0-9]* skips=[0-9]*$" "%HEREDIR%obj\%%f.runlog"') do set SUMMARY_LINE=%%s
    if "!SUMMARY_LINE!"=="" (
        echo T-2138 %%f produced no summary ^(exit !RUN_EC!^)
        set OVERALL_OK=0
    ) else (
        echo !SUMMARY_LINE! | findstr /R "failures=0 " >nul
        if errorlevel 1 set OVERALL_OK=0
        if not "!RUN_EC!"=="0" set OVERALL_OK=0
    )
)

if "!OVERALL_OK!"=="1" (
    echo T-2138 SUITE STATUS: EXECUTED GREEN.
    exit /b 0
)
echo T-2138 SUITE STATUS: EXECUTION FAILURE -- see obj\*.runlog.
exit /b 1
