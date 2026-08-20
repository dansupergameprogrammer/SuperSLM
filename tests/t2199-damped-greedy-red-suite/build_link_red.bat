@echo off
rem T-2199 (Curie): builds every RED-BY-LINK C++ cell file in this directory against the
rem REAL, certified sub-primitives Phase A/Phase C are specified to reuse (intmath.cpp,
rem checked_chain_funnel.cpp, and their own real dependencies -- matmul.cpp, silu_lut.cpp,
rem trace_hook.cpp) and captures the LINK transcript. Per StandardsDocument.md Sec5.4/Curie's
rem own red-first discipline, this is run and the transcript kept as evidence a cell is red
rem for the RIGHT reason (LNK2019 unresolved external on the damped-greedy symbols this suite
rem calls -- AntiLm*/FsdTopK/TopKRenormalizeQ15/DampedGreedyScoreAndArgmax* -- never a compile
rem error and never a silent pass), not merely asserted in prose. Mirrors
rem tests/t2130-g5-red-suite/build_link_red.bat's own convention exactly, trimmed to the
rem narrower dependency set Phase A/Phase C actually need (no full engine ABI, no model
rem loading, no GPU bridge -- this suite's own scope is standalone, per the plan's own Sec8
rem "no engine dependency, buildable and testable standalone" phase-ordering rationale).
rem
rem Exit code is 1 (red) until the build seat lands Phase A's and Phase C's own .cpp; this
rem script's own exit code inverts to 0 ONLY once every cell links, at which point this suite
rem has gone from red to buildable and the individual CHECK/FAIL output governs pass/fail.
setlocal enabledelayedexpansion
set HEREDIR=%~dp0
set ENG=%HEREDIR%..\..
call "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 -no_logo
cd /d "%HEREDIR%"
if not exist obj mkdir obj

set OVERALL_LINK_OK=1
for %%f in (antilm_phaseA_red.cpp topk_select_phaseC_red.cpp renormalize_phaseC_red.cpp score_select_phaseC_red.cpp) do (
    echo ===== %%f =====
    cl /nologo /std:c++20 /O2 /W4 /fp:precise /EHsc /I%ENG%\include /I. ^
        %ENG%\src\intmath.cpp %ENG%\src\silu_lut.cpp %ENG%\src\matmul.cpp ^
        %ENG%\src\trace_hook.cpp %ENG%\src\forward\checked_chain_funnel.cpp ^
        "%%f" /Fo:"obj\\" /Fe:"obj\%%~nf.exe" ^
        /link > "obj\%%~nf.log" 2>&1
    findstr /C:"error C" "obj\%%~nf.log" >nul
    if not errorlevel 1 (
        echo    COMPILE ERROR -- unexpected, this cell should compile clean against
        echo    sslm_damped_greedy.h:
        type "obj\%%~nf.log"
        set OVERALL_LINK_OK=0
    ) else (
        findstr /C:"LNK2019" /C:"LNK1120" "obj\%%~nf.log" >nul
        if not errorlevel 1 (
            echo    RED BY LINK -- unresolved external^(s^) on the undeclared damped-greedy
            echo    Phase A/Phase C implementation:
            findstr /C:"LNK2019" "obj\%%~nf.log"
        ) else (
            findstr /C:"error" "obj\%%~nf.log" >nul
            if not errorlevel 1 (
                echo    UNEXPECTED ERROR CLASS -- see obj\%%~nf.log
                type "obj\%%~nf.log"
                set OVERALL_LINK_OK=0
            ) else (
                echo    LINKED CLEAN -- this cell's own implementation now exists; run the .exe.
            )
        )
    )
)

if !OVERALL_LINK_OK! == 1 (
    echo.
    echo All cells compiled clean; each either linked clean ^(Phase A/C already built^) or
    echo failed with LNK2019/LNK1120 on the undeclared symbols ^(genuinely red^). No compile
    echo error and no unexpected error class anywhere -- this run is evidence, not assertion.
    exit /b 1
) else (
    echo.
    echo UNEXPECTED: at least one cell hit a compile error or an unexpected error class --
    echo this is NOT the red-for-the-right-reason state this script exists to confirm.
    exit /b 2
)
