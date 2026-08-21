@echo off
rem T-2199 (Curie) -- Phase D red-by-link build. Links every Phase D cell file against the
rem REAL, already-shipped CPU ABI/engine (the same source list tests/t2138-abi-red-suite/
rem build_link_red.bat uses, CPU-only, no GPU symbols) PLUS Phase A/C's own already-built
rem src/damped_greedy_antilm.cpp/src/damped_greedy_topk.cpp (feature/t2199-damped-greedy@
rem c02b156). Captures the LINK transcript as evidence a cell is red for the RIGHT reason
rem (LNK2019 on the undeclared Phase D symbols this suite calls -- sslm_decode_step_damped_
rem greedy, RunGreedyOrDampedGreedyDecodeLoop, ArtifactHasDampedGreedyConstants,
rem ReadDampedGreedyScaleConstants, ValidateDampedGreedyParams -- never a compile error).
setlocal enabledelayedexpansion
set HEREDIR=%~dp0
set ENG=%HEREDIR%..\..
set TESTS=%ENG%\tests
call "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 -no_logo
cd /d "%HEREDIR%"
if not exist obj_phaseD mkdir obj_phaseD

set OVERALL_LINK_OK=1
for %%f in (phaseD1_artifact_flag_red.cpp phaseD2_wiring_red.cpp phaseD2a_cost_ratio_red.cpp phaseD3_teardown_red.cpp) do (
    echo ===== %%f =====
    cl /nologo /std:c++20 /O2 /W4 /fp:precise /EHsc /I%ENG%\include /I%TESTS% /I. ^
        %ENG%\src\artifact.cpp %ENG%\src\sha256.cpp %ENG%\src\tokenizer.cpp %ENG%\src\model.cpp ^
        %ENG%\src\intmath.cpp %ENG%\src\silu_lut.cpp %ENG%\src\matmul.cpp %ENG%\src\proof_manifest.cpp ^
        %ENG%\src\trace_hook.cpp %ENG%\src\forward\checked_chain_funnel.cpp ^
        %ENG%\src\forward\forward_sites.cpp %ENG%\src\decode_digest.cpp %ENG%\src\sslm_abi.cpp ^
        %ENG%\src\damped_greedy_antilm.cpp %ENG%\src\damped_greedy_topk.cpp ^
        "%%f" /Fo:"obj_phaseD\\" /Fe:"obj_phaseD\%%~nf.exe" ^
        /link > "obj_phaseD\%%~nf.log" 2>&1
    findstr /C:"error C" "obj_phaseD\%%~nf.log" >nul
    if not errorlevel 1 (
        echo    COMPILE ERROR -- unexpected:
        type "obj_phaseD\%%~nf.log"
        set OVERALL_LINK_OK=0
    ) else (
        findstr /C:"LNK2019" /C:"LNK1120" "obj_phaseD\%%~nf.log" >nul
        if not errorlevel 1 (
            echo    RED BY LINK -- unresolved external^(s^) on the undeclared Phase D symbols:
            findstr /C:"LNK2019" "obj_phaseD\%%~nf.log"
        ) else (
            findstr /C:"error" "obj_phaseD\%%~nf.log" >nul
            if not errorlevel 1 (
                echo    UNEXPECTED ERROR CLASS:
                type "obj_phaseD\%%~nf.log"
                set OVERALL_LINK_OK=0
            ) else (
                echo    LINKED CLEAN -- this cell's own implementation now exists; run the .exe.
            )
        )
    )
)
if !OVERALL_LINK_OK! == 1 (
    exit /b 1
) else (
    exit /b 2
)
