@echo off
rem T-2199 Phase D (Brunel, build glue): builds every Phase D cell against the real
rem implementation (src/damped_greedy_phaseD.cpp, src/damped_greedy_phaseD_loop.cpp, plus the
rem widened src/sslm_abi.cpp/include/superslm/artifact.h) and runs each, mirroring
rem build_link_red_phaseD.bat's own source list plus the new production files. A real base
rem artifact is required for D2/D2a/D3's own product cells (--model=PATH); D1 needs none.
rem Usage: build_green_phaseD.bat [PATH-TO-.sslm]
setlocal enabledelayedexpansion
set HEREDIR=%~dp0
set ENG=%HEREDIR%..\..
set TESTS=%ENG%\tests
set MODELARG=
if not "%~1"=="" set MODELARG=--model=%~1
call "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 -no_logo
cd /d "%HEREDIR%"
if not exist obj_green_D mkdir obj_green_D

set SRC=%ENG%\src\artifact.cpp %ENG%\src\sha256.cpp %ENG%\src\tokenizer.cpp %ENG%\src\model.cpp ^
    %ENG%\src\intmath.cpp %ENG%\src\silu_lut.cpp %ENG%\src\matmul.cpp %ENG%\src\proof_manifest.cpp ^
    %ENG%\src\trace_hook.cpp %ENG%\src\forward\checked_chain_funnel.cpp ^
    %ENG%\src\forward\forward_sites.cpp %ENG%\src\decode_digest.cpp %ENG%\src\sslm_abi.cpp ^
    %ENG%\src\damped_greedy_antilm.cpp %ENG%\src\damped_greedy_topk.cpp ^
    %ENG%\src\damped_greedy_phaseD.cpp %ENG%\src\damped_greedy_phaseD_loop.cpp

set OVERALL_OK=1
for %%f in (phaseD1_artifact_flag_red phaseD2_wiring_red phaseD2a_cost_ratio_red phaseD3_teardown_red) do (
    echo ===== %%f.cpp =====
    cl /nologo /std:c++20 /O2 /W4 /fp:precise /EHsc /I%ENG%\include /I%TESTS% /I. %SRC% ^
        "%%f.cpp" /Fo:"obj_green_D\\" /Fe:"obj_green_D\%%f.exe" ^
        /link > "obj_green_D\%%f.buildlog" 2>&1
    if errorlevel 1 (
        echo    BUILD FAILED:
        type "obj_green_D\%%f.buildlog"
        set OVERALL_OK=0
    ) else (
        "obj_green_D\%%f.exe" %MODELARG% > "obj_green_D\%%f.runlog" 2>&1
        set RUN_EC=!errorlevel!
        type "obj_green_D\%%f.runlog"
        rem CURIE FIX, 2026-08-20 (conductor's dispute-resolution commission): this script
        rem previously always exited 0 regardless of a crashing .exe -- a nonzero RUN_EC with no
        rem "checks=.. failures=.. skips=.." summary line in the runlog is a CRASH (matching
        rem build_green.bat's own established convention for Phase A/C), and was silently
        rem swallowed here. Executed finding this exact gap would have hidden:
        rem TestD3_TeardownDuringFlight_ConcurrentReleaseDoesNotCorruptSurvivor crashes
        rem (0xC0000005/0xC0000409, reproduced 5/5 at kTrials=1) against a real checkpoint --
        rem see Claude/Curie/t2199-phaseD-red-2026-08-20.md for the full reproduction.
        set SUMMARY_LINE=
        for /f "delims=" %%s in ('findstr /R "^checks=[0-9]* failures=[0-9]* skips=[0-9]*$" "obj_green_D\%%f.runlog"') do set SUMMARY_LINE=%%s
        if "!SUMMARY_LINE!"=="" (
            if not "!RUN_EC!"=="0" (
                echo    CRASHED -- exit code !RUN_EC!, no checks=/failures=/skips= summary line
                set OVERALL_OK=0
            )
        ) else (
            rem BRUNEL FIX, T-2199 Phase D review S8 (Claude/Poirot/7a3b10a-t2199-phaseD-review.md):
            rem this script previously NEVER gated on a nonzero failures= count -- true even after
            rem every §4 fixture dispute this comment used to cite was resolved by the merged
            rem curie/t2199-phaseD-red@2a33e14 dispute round, so the "known, routed-not-fixed"
            rem justification no longer held and a real failures>0 count was passing silently.
            rem Now: any failures= value other than 0 fails this script exactly like a crash does.
            echo !SUMMARY_LINE! | findstr /R "failures=0 " >nul
            if errorlevel 1 (
                echo    FAILURES: !SUMMARY_LINE!
                set OVERALL_OK=0
            )
        )
    )
)

echo.
echo Per-cell checks=/failures=/skips= lines are above; obj_green_D\*.runlog carries each
echo transcript. This script gates on BOTH a crash (a nonzero process exit with no
echo checks=/failures=/skips= summary line) AND a nonzero failures= count in that summary line --
echo neither is a fixture dispute at this build's own tip (every §4 dispute this ticket routed was
echo resolved by the merged curie/t2199-phaseD-red@2a33e14/7a3b10a rounds; a failures^>0 count from
echo here forward is always a real regression).
if !OVERALL_OK! == 1 (
    exit /b 0
) else (
    exit /b 1
)
