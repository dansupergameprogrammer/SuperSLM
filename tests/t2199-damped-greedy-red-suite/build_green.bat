@echo off
rem T-2199 (Brunel, fix round 2026-08-20, Claude/Poirot/7be9508-t2199-phaseAC-review.md
rem Finding S4): builds every cell in this directory against the REAL Phase A/Phase C
rem implementation (src/damped_greedy_antilm.cpp, src/damped_greedy_topk.cpp) alongside the
rem same certified sub-primitives build_link_red.bat compiles against, RUNS each resulting
rem .exe, and aggregates their own CHECK/FAIL/SKIP counters into one pass/fail verdict.
rem
rem This is the missing half build_link_red.bat deliberately does not provide (S4): that
rem script's own job is proving the suite is RED (fails to link) before the implementation
rem exists; this script's job is proving the suite is GREEN (links and every check passes)
rem once it does. Neither script edits the other or the suite's own .cpp files -- this is
rem build glue, not a test.
rem
rem Usage: build_green.bat from this directory, or from anywhere (it cd's to its own
rem location first). Exit code 0 iff every cell linked clean AND every cell's own binary
rem reported zero failures; nonzero otherwise. Per-cell transcripts land in obj_green\.
setlocal enabledelayedexpansion
set HEREDIR=%~dp0
set ENG=%HEREDIR%..\..
call "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 -no_logo
cd /d "%HEREDIR%"
if not exist obj_green mkdir obj_green

set OVERALL_OK=1
set TOTAL_CHECKS=0
set TOTAL_FAILURES=0
set TOTAL_SKIPS=0

for %%f in (antilm_phaseA_red.cpp topk_select_phaseC_red.cpp renormalize_phaseC_red.cpp score_select_phaseC_red.cpp) do (
    echo ===== %%f =====
    cl /nologo /std:c++20 /O2 /W4 /fp:precise /EHsc /I%ENG%\include /I. ^
        %ENG%\src\intmath.cpp %ENG%\src\silu_lut.cpp %ENG%\src\matmul.cpp ^
        %ENG%\src\trace_hook.cpp %ENG%\src\forward\checked_chain_funnel.cpp ^
        %ENG%\src\damped_greedy_antilm.cpp %ENG%\src\damped_greedy_topk.cpp ^
        "%%f" /Fo:"obj_green\\" /Fe:"obj_green\%%~nf.exe" ^
        /link > "obj_green\%%~nf.buildlog" 2>&1
    if errorlevel 1 (
        echo    BUILD FAILED -- see obj_green\%%~nf.buildlog:
        type "obj_green\%%~nf.buildlog"
        set OVERALL_OK=0
    ) else (
        "obj_green\%%~nf.exe" > "obj_green\%%~nf.runlog" 2>&1
        set RUN_EC=!errorlevel!
        type "obj_green\%%~nf.runlog"
        findstr /R "^checks=[0-9]* failures=[0-9]* skips=[0-9]*$" "obj_green\%%~nf.runlog" >nul
        if errorlevel 1 (
            echo    UNEXPECTED -- no "checks=.. failures=.. skips=.." summary line found
            set OVERALL_OK=0
        ) else (
            if not !RUN_EC! == 0 (
                echo    RUN FAILED -- %%f reported a nonzero exit ^(!RUN_EC!^)
                set OVERALL_OK=0
            )
        )
    )
)

if !OVERALL_OK! == 1 (
    echo.
    echo ALL CELLS BUILT AND RAN CLEAN. Per-cell checks=/failures=/skips= lines are above;
    echo see obj_green\*.runlog for the individual transcripts. This script's own exit code
    echo is 0 iff every cell both linked and exited 0.
    exit /b 0
) else (
    echo.
    echo NOT GREEN -- at least one cell failed to build, failed to run cleanly, or reported
    echo a nonzero exit. See the per-cell logs in obj_green\ above.
    exit /b 1
)
