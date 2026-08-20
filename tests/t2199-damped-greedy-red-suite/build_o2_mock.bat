@echo off
rem T-2199 (Curie): builds and runs the ONE cell that needs a substituted AntiLmState
rem (o2_masked_queried_red.cpp, linked against the REAL, unmodified src/damped_greedy_topk.cpp
rem and this directory's own TEST-ONLY o2_counting_antilm_mock.cpp INSTEAD OF
rem src/damped_greedy_antilm.cpp). This is its own, separate build recipe -- not folded
rem directly into build_link_red.bat's own file list -- because its link line genuinely
rem differs (one production source swapped for a test double). build_green.bat's own
rem "everything" run calls this script and folds its exit code into its own OVERALL_OK (M7,
rem `Claude/Poirot/927bbda-t2199-confirmation.md`: "the committed recipe story" must produce
rem one honest count, not a board figure no single script reproduces) -- this script also
rem stays independently runnable on its own. Exit code 0 iff the cell links and reports zero
rem failures.
setlocal enabledelayedexpansion
set HEREDIR=%~dp0
set ENG=%HEREDIR%..\..
call "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 -no_logo
cd /d "%HEREDIR%"
if not exist obj_o2 mkdir obj_o2

cl /nologo /std:c++20 /O2 /W4 /fp:precise /EHsc /I%ENG%\include /I. ^
    %ENG%\src\intmath.cpp %ENG%\src\silu_lut.cpp %ENG%\src\matmul.cpp ^
    %ENG%\src\trace_hook.cpp %ENG%\src\forward\checked_chain_funnel.cpp ^
    %ENG%\src\damped_greedy_topk.cpp o2_counting_antilm_mock.cpp ^
    o2_masked_queried_red.cpp /Fo:"obj_o2\\" /Fe:"obj_o2\o2_masked_queried_red.exe" ^
    /link > "obj_o2\o2_masked_queried_red.buildlog" 2>&1
if errorlevel 1 (
    echo BUILD FAILED -- see obj_o2\o2_masked_queried_red.buildlog:
    type "obj_o2\o2_masked_queried_red.buildlog"
    exit /b 1
)
"obj_o2\o2_masked_queried_red.exe" > "obj_o2\o2_masked_queried_red.runlog" 2>&1
set RUN_EC=!errorlevel!
type "obj_o2\o2_masked_queried_red.runlog"
if not !RUN_EC! == 0 (
    echo RUN FAILED -- nonzero exit ^(!RUN_EC!^)
    exit /b 1
)
echo O2 MOCK CELL: BUILT AND RAN CLEAN.
exit /b 0
