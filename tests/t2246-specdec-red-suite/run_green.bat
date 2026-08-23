@echo off
rem T-2246 (build seat): builds every C++ cell file in this directory against the REAL engine
rem -- the identical CPU source list build_link_red.bat uses, plus
rem SUPERSLM_ENABLE_T2246_SPECDEC_FAULT_INJECTION so the mid-verify fault seam (CM-G3) is
rem consulted -- and EXECUTES each cell against supplied artifacts. This is the green-state
rem runner; build_link_red.bat remains the pre-build red-state gate. Per-cell summaries are
rem reported honestly; any failures=N summary or nonzero exit marks the suite not-green.
rem
rem Usage: run_green.bat <model.sslm>   (or set T2246_MODEL)
rem Optional: T2246_MODEL_TOK (tokenizer-bearing .sslm), T2246_CORPUS (shopkeeper JSONL).
setlocal enabledelayedexpansion
set HEREDIR=%~dp0
set MODEL=%~1
if "%MODEL%"=="" set MODEL=%T2246_MODEL%
if "%MODEL%"=="" (
    echo T-2246 run_green: model path required ^(argument or T2246_MODEL^)
    exit /b 2
)

call "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 -no_logo
cd /d "%HEREDIR%"
if not exist obj mkdir obj

set ENG=%HEREDIR%..\..
set ARGS=--model=%MODEL%
if defined T2246_MODEL_TOK set ARGS=!ARGS! --modeltok=%T2246_MODEL_TOK%
if defined T2246_CORPUS set ARGS=!ARGS! --corpus=%T2246_CORPUS%

set OVERALL_OK=1
for %%f in (dim01_equivalence_red dim02_rollback_red dim03_retention_red dim04_drafter_red dim05_trust_red dim06_concurrency_red dim07_failure_red dim08_guard_red dim09_persistence_red dim10_composition_red dim11_matrix_red dim12_performance_red) do (
    echo ===== %%f =====
    cl /nologo /std:c++20 /O2 /W4 /fp:precise /EHsc /DSUPERSLM_ENABLE_T2246_SPECDEC_FAULT_INJECTION /I%ENG%\include /I%ENG%\tests\support /I. ^
        %ENG%\src\artifact.cpp %ENG%\src\sha256.cpp %ENG%\src\tokenizer.cpp %ENG%\src\model.cpp ^
        %ENG%\src\intmath.cpp %ENG%\src\silu_lut.cpp %ENG%\src\matmul.cpp %ENG%\src\proof_manifest.cpp ^
        %ENG%\src\trace_hook.cpp %ENG%\src\forward\checked_chain_funnel.cpp ^
        %ENG%\src\forward\forward_sites.cpp %ENG%\src\decode_digest.cpp %ENG%\src\sslm_abi.cpp ^
        %ENG%\src\damped_greedy_antilm.cpp %ENG%\src\damped_greedy_topk.cpp ^
        %ENG%\src\damped_greedy_phaseD.cpp %ENG%\src\damped_greedy_phaseD_loop.cpp ^
        "%%f.cpp" /Fo:"obj\\" /Fe:"obj\%%f.exe" > "obj\%%f.log" 2>&1
    if errorlevel 1 (
        echo    BUILD FAILURE -- see obj\%%f.log
        type "obj\%%f.log" | findstr /C:"error"
        set OVERALL_OK=0
    ) else (
        "obj\%%f.exe" !ARGS! > "obj\%%f.runlog" 2>&1
        set RUN_EC=!errorlevel!
        type "obj\%%f.runlog"
        set SUMMARY_LINE=
        for /f "delims=" %%s in ('findstr /R "^checks=[0-9]* failures=[0-9]* skips=[0-9]*$" "obj\%%f.runlog"') do set SUMMARY_LINE=%%s
        if "!SUMMARY_LINE!"=="" (
            echo T-2246 %%f produced no summary ^(exit !RUN_EC!^)
            set OVERALL_OK=0
        ) else (
            echo !SUMMARY_LINE! | findstr /R "failures=0 " >nul
            if errorlevel 1 set OVERALL_OK=0
            if not "!RUN_EC!"=="0" set OVERALL_OK=0
        )
    )
)

echo.
if "!OVERALL_OK!"=="1" (
    echo T-2246 SUITE STATUS: EXECUTED GREEN.
    exit /b 0
)
echo T-2246 SUITE STATUS: NOT GREEN -- see per-cell summaries above.
exit /b 1
