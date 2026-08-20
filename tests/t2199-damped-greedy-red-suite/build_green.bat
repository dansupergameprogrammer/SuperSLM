@echo off
rem T-2199: builds every cell in this directory against the REAL Phase A/Phase C
rem implementation (src/damped_greedy_antilm.cpp, src/damped_greedy_topk.cpp -- and, for the
rem O2 mock cell alone, a TEST-ONLY substitute for the former, see below) alongside the same
rem certified sub-primitives build_link_red.bat compiles against, RUNS each resulting .exe,
rem and aggregates their own CHECK/FAIL/SKIP counters into ONE numeric total and one pass/fail
rem verdict.
rem
rem This is the missing half build_link_red.bat deliberately does not provide (S4): that
rem script's own job is proving the suite is RED (fails to link) before the implementation
rem exists; this script's job is proving the suite is GREEN (links and every check passes)
rem once it does. Neither script edits the other or the suite's own .cpp files -- this is
rem build glue, not a test.
rem
rem M7, `Claude/Poirot/927bbda-t2199-confirmation.md`: this script previously covered only the
rem four main cell files, iterating a fixed list, and declared (but never summed) its own
rem TOTAL_CHECKS/TOTAL_FAILURES/TOTAL_SKIPS -- "run the green battery" produced 254/0/2 while
rem the board recorded 264/0/2 (the O2 mock cell's own 10 checks, built and run only by the
rem separate build_o2_mock.bat, which nothing invoked). Fixed: this script now calls
rem build_o2_mock.bat as its own fifth cell and folds its exit code AND its own
rem checks=/failures=/skips= line into the SAME aggregate this script already computes for the
rem other four -- one recipe, one number, matching what the board cites.
rem
rem Usage: build_green.bat from this directory, or from anywhere (it cd's to its own
rem location first). Exit code 0 iff every cell linked clean AND every cell's own binary
rem reported zero failures; nonzero otherwise. Per-cell transcripts land in obj_green\ (the
rem four main cells) and obj_o2\ (the O2 mock cell, build_o2_mock.bat's own directory).
setlocal enabledelayedexpansion
set HEREDIR=%~dp0
set ENG=%HEREDIR%..\..
call "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 -no_logo
cd /d "%HEREDIR%"
if not exist obj_green mkdir obj_green

set OVERALL_OK=1
set /a TOTAL_CHECKS=0
set /a TOTAL_FAILURES=0
set /a TOTAL_SKIPS=0

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
        set SUMMARY_LINE=
        for /f "tokens=*" %%s in ('findstr /R "^checks=[0-9]* failures=[0-9]* skips=[0-9]*$" "obj_green\%%~nf.runlog"') do set SUMMARY_LINE=%%s
        if not defined SUMMARY_LINE (
            echo    UNEXPECTED -- no "checks=.. failures=.. skips=.." summary line found
            set OVERALL_OK=0
        ) else (
            if not !RUN_EC! == 0 (
                echo    RUN FAILED -- %%f reported a nonzero exit ^(!RUN_EC!^)
                set OVERALL_OK=0
            )
            for /f "tokens=1,2,3 delims= " %%a in ("!SUMMARY_LINE!") do (
                set "C_TOK=%%a" & set "F_TOK=%%b" & set "S_TOK=%%c"
            )
            rem Fixed-offset substring, not search-replace: "checks="/"failures="/"skips=" each
            rem contain their own "=", which the %VAR:find=replace% syntax cannot express
            rem unambiguously (an earlier version of this script tried it and silently produced
            rem "=53" instead of "53", making every TOTAL_* accumulate garbage -- caught by
            rem executing this script and reading its own AGGREGATE line, not by inspection).
            set "C_VAL=!C_TOK:~7!" & set "F_VAL=!F_TOK:~9!" & set "S_VAL=!S_TOK:~6!"
            set /a TOTAL_CHECKS+=!C_VAL!
            set /a TOTAL_FAILURES+=!F_VAL!
            set /a TOTAL_SKIPS+=!S_VAL!
        )
    )
)

echo ===== o2_masked_queried_red.cpp (via build_o2_mock.bat) =====
call "%HEREDIR%build_o2_mock.bat"
set O2_EC=!errorlevel!
if not !O2_EC! == 0 (
    echo    build_o2_mock.bat reported a nonzero exit ^(!O2_EC!^)
    set OVERALL_OK=0
)
set O2_SUMMARY=
if exist "obj_o2\o2_masked_queried_red.runlog" (
    for /f "tokens=*" %%s in ('findstr /R "^checks=[0-9]* failures=[0-9]* skips=[0-9]*$" "obj_o2\o2_masked_queried_red.runlog"') do set O2_SUMMARY=%%s
)
if not defined O2_SUMMARY (
    echo    UNEXPECTED -- no "checks=.. failures=.. skips=.." summary line found for the O2 cell
    set OVERALL_OK=0
) else (
    for /f "tokens=1,2,3 delims= " %%a in ("!O2_SUMMARY!") do (
        set "C_TOK=%%a" & set "F_TOK=%%b" & set "S_TOK=%%c"
    )
    set "C_VAL=!C_TOK:~7!" & set "F_VAL=!F_TOK:~9!" & set "S_VAL=!S_TOK:~6!"
    set /a TOTAL_CHECKS+=!C_VAL!
    set /a TOTAL_FAILURES+=!F_VAL!
    set /a TOTAL_SKIPS+=!S_VAL!
)

echo.
echo ===== AGGREGATE: checks=!TOTAL_CHECKS! failures=!TOTAL_FAILURES! skips=!TOTAL_SKIPS! =====

if !OVERALL_OK! == 1 (
    echo.
    echo ALL CELLS BUILT AND RAN CLEAN, INCLUDING THE O2 MOCK CELL. Per-cell checks=/failures=/
    echo skips= lines are above; see obj_green\*.runlog and obj_o2\*.runlog for the individual
    echo transcripts. This script's own exit code is 0 iff every cell both linked and exited 0.
    exit /b 0
) else (
    echo.
    echo NOT GREEN -- at least one cell failed to build, failed to run cleanly, or reported
    echo a nonzero exit. See the per-cell logs in obj_green\ and obj_o2\ above.
    exit /b 1
)
