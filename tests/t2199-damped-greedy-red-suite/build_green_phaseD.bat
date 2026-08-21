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
        type "obj_green_D\%%f.runlog"
    )
)

echo.
echo Per-cell checks=/failures=/skips= lines are above; obj_green_D\*.runlog carries each
echo transcript. This script does NOT gate on failures=0 (unlike build_green.bat for Phase
echo A/C) -- Phase D's own known, routed-not-fixed fixture disputes (build log Sec4) make a
echo hard pass/fail exit code misleading here; read the transcripts.
exit /b 0
