@echo off
rem T-2899 (Curie) -- builds every CPU-only cell in this suite against the tree AS IT STANDS
rem (Stage 1, curie/t2809-stage1-build) and runs it. Mirrors
rem tests/t2199-damped-greedy-red-suite/build_green_phaseD.bat's own source list (this suite's
rem cells call the identical public ABI, sslm_decode_step/sslm_seq_adopt_prefix/sslm_prefill).
rem Usage: build_red_suite.bat --model=PATH-TO-.sslm [--schema=NAME]
setlocal enabledelayedexpansion
set HEREDIR=%~dp0
set ENG=%HEREDIR%..\..
set TESTS=%ENG%\tests
set MODELARG=%*
call "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 -no_logo
cd /d "%HEREDIR%"
if not exist obj mkdir obj

set SRC=%ENG%\src\artifact.cpp %ENG%\src\sha256.cpp %ENG%\src\tokenizer.cpp %ENG%\src\model.cpp ^
    %ENG%\src\intmath.cpp %ENG%\src\silu_lut.cpp %ENG%\src\matmul.cpp %ENG%\src\proof_manifest.cpp ^
    %ENG%\src\trace_hook.cpp %ENG%\src\forward\checked_chain_funnel.cpp ^
    %ENG%\src\forward\forward_sites.cpp %ENG%\src\decode_digest.cpp %ENG%\src\sslm_abi.cpp ^
    %ENG%\src\damped_greedy_antilm.cpp %ENG%\src\damped_greedy_topk.cpp ^
    %ENG%\src\damped_greedy_phaseD.cpp %ENG%\src\damped_greedy_phaseD_loop.cpp

set OVERALL_OK=1
for %%f in (cell_adopt_prefix_census cell_cpu_deadend_retry_reset) do (
    echo ===== %%f.cpp =====
    rem T-2905: SUPERSLM_CPU_G5_FINISH_ROW_FAULT_INJECTION defined for every cell in this loop --
    rem it gates ArmCpuFinishDegenerateLogitRowInjection's own definition in sslm_abi.cpp (%SRC%,
    rem recompiled fresh below for every cell) and cell_cpu_deadend_retry_reset.cpp's own
    rem extern "C" reservation of it (CpuCell2DegenerateRowTwin). Harmless for
    rem cell_adopt_prefix_census, which does not reference the symbol.
    cl /nologo /std:c++20 /O2 /W4 /fp:precise /EHsc /DSUPERSLM_CPU_G5_FINISH_ROW_FAULT_INJECTION ^
        /I%ENG%\include /I%ENG%\src /I%TESTS% /I%TESTS%\t2199-damped-greedy-red-suite /I. %SRC% ^
        "%%f.cpp" /Fo:"obj\\" /Fe:"obj\%%f.exe" ^
        /link > "obj\%%f.buildlog" 2>&1
    if errorlevel 1 (
        echo    BUILD FAILED:
        type "obj\%%f.buildlog"
        set OVERALL_OK=0
    ) else (
        "obj\%%f.exe" %MODELARG% > "obj\%%f.runlog" 2>&1
        set RUN_EC=!errorlevel!
        type "obj\%%f.runlog"
        set SUMMARY_LINE=
        for /f "delims=" %%s in ('findstr /R "^checks=[0-9]* failures=[0-9]*" "obj\%%f.runlog"') do set SUMMARY_LINE=%%s
        if "!SUMMARY_LINE!"=="" (
            echo    CRASHED OR NO SUMMARY LINE -- exit code !RUN_EC!
            set OVERALL_OK=0
        )
    )
)

if "%OVERALL_OK%"=="1" (
    echo ===== build_red_suite: all cells built and ran =====
    exit /b 0
) else (
    echo ===== build_red_suite: FAILURES ABOVE =====
    exit /b 1
)
