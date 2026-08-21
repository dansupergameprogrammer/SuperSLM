@echo off
rem T-2130 (Curie): builds every C++ cell file in this directory against the REAL engine (the
rem identical source list build.bat, this repo's own root, uses) and captures the LINK
rem transcript. Historical purpose: prove a cell is RED BY LINK for the right reason before
rem the G5 ABI existed.
rem
rem T-2132 harness fix (Curie, 2026-08-17): src\sslm_abi.cpp -- the file that already defines
rem sslm_model_map/sslm_seq_create/sslm_prefill/sslm_decode_step/etc, shipped under T-2139 --
rem is now on this link line, matching the source list the repo root build.bat's own C2/C3/...
rem tool builds use for the identical ABI file. Before this fix, every one of this suite's 12
rem binaries reported LNK2019 on ALREADY-SHIPPED pre-G5 symbols (sslm_seq_create, sslm_prefill,
rem sslm_decode_step, sslm_seq_save, sslm_seq_restore, ...) that src\sslm_abi.cpp defines --
rem captured, evidenced, in obj\*.log at main@aea6116 with NO G5 code written -- so this script
rem could never distinguish "G5's own verbs are unbuilt" from "the pre-G5 ABI this script never
rem linked". Landing every G5-2..G5-6 symbol perfectly would not have moved this script's exit
rem code without this fix, because the file that defines them was never on its own link line.
rem
rem N1 STRUCTURAL FIX (Claude/Poirot/a12bbdd-t2199-phaseD-closing.md): the IDENTICAL class
rem recurred a third time -- src/sslm_abi.cpp has called superslm::AntiLmCreate/AntiLmDestroy/
rem AntiLmUpdate/DampedGreedyScoreAndArgmax/ValidateDampedGreedyParams since T-2199 Phase D
rem landed (first reference at 5ced7a6), and this script's source list was never updated --
rem every one of these 12 cells LNK2019'd on all five symbols, zero .exe produced, and this
rem script's own tail logic reported that as "SUITE STATUS: RED BY LINK, as expected
rem pre-build," indistinguishable from a real link-list defect. Fixed two ways: (1) the four
rem damped_greedy_*.cpp translation units appended below, matching build_green_phaseD.bat's
rem own source list; (2) the tail logic itself, since the G5 ABI this suite links against is
rem real and shipped and has been for a long time -- per Curie's own build_green gating
rem precedent (build_green.bat/build_green_phaseD.bat), a link failure is now NEVER reported
rem as this suite's expected/red state. This script's own contract, going forward:
rem   - every cell links clean -> SUITE STATUS: LINKED CLEAN, exit /b 0.
rem   - ANY cell hits LNK2019/LNK1120 -> SUITE STATUS: LINK FAILURE (loud, per-cell,
rem     per-symbol), exit /b 1 -- a real defect, never "expected."
rem   - a compile error or any other unexpected error class -> exit /b 2, unchanged.
setlocal enabledelayedexpansion
set HEREDIR=%~dp0
set ENG=%HEREDIR%..\..
call "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 -no_logo
cd /d "%HEREDIR%"
if not exist obj mkdir obj

set ANY_COMPILE_ERROR=0
set ANY_LINK_FAILURE=0
for %%f in (dim1_lifetime_red.cpp dim2_hostile_red.cpp dim3_concurrency_red.cpp dim4_shape_red.cpp dim5_failure_red.cpp dim6_determinism_red.cpp dim7_contract_red.cpp dim8_composition_red.cpp dim9_persistence_red.cpp dim10_functional_red.cpp dim11_guard_red.cpp g5_collision_regression_red.cpp) do (
    echo ===== %%f =====
    cl /nologo /std:c++20 /O2 /W4 /fp:precise /EHsc /I%ENG%\include /I. ^
        %ENG%\src\artifact.cpp %ENG%\src\sha256.cpp %ENG%\src\tokenizer.cpp %ENG%\src\model.cpp ^
        %ENG%\src\intmath.cpp %ENG%\src\silu_lut.cpp %ENG%\src\matmul.cpp %ENG%\src\proof_manifest.cpp ^
        %ENG%\src\trace_hook.cpp %ENG%\src\forward\checked_chain_funnel.cpp ^
        %ENG%\src\forward\forward_sites.cpp %ENG%\src\decode_digest.cpp %ENG%\src\gpu\superslm_gpu.cpp ^
        %ENG%\src\sslm_abi.cpp ^
        %ENG%\src\damped_greedy_antilm.cpp %ENG%\src\damped_greedy_topk.cpp ^
        %ENG%\src\damped_greedy_phaseD.cpp %ENG%\src\damped_greedy_phaseD_loop.cpp ^
        "%%f" /Fo:"obj\\" /Fe:"obj\%%~nf.exe" ^
        /link d3d12.lib dxgi.lib dxguid.lib > "obj\%%~nf.log" 2>&1
    findstr /C:"error C" "obj\%%~nf.log" >nul
    if not errorlevel 1 (
        echo    COMPILE ERROR -- unexpected, this cell should compile clean against sslm_g5.h:
        type "obj\%%~nf.log"
        set ANY_COMPILE_ERROR=1
    ) else (
        findstr /C:"LNK2019" /C:"LNK1120" "obj\%%~nf.log" >nul
        if not errorlevel 1 (
            echo    *** LINK FAILURE *** unresolved external^(s^) -- this is a DEFECT, not an
            echo    expected pre-build state ^(the G5/CPU ABI this cell links against is real
            echo    and shipped^). Missing symbols:
            findstr /C:"LNK2019" "obj\%%~nf.log"
            set ANY_LINK_FAILURE=1
        ) else (
            findstr /C:"error" "obj\%%~nf.log" >nul
            if not errorlevel 1 (
                echo    UNEXPECTED ERROR CLASS -- see obj\%%~nf.log
                type "obj\%%~nf.log"
                set ANY_COMPILE_ERROR=1
            ) else (
                echo    LINKED CLEAN.
            )
        )
    )
)
echo.
if "%ANY_COMPILE_ERROR%"=="1" (
    echo SUITE STATUS: COMPILE ERROR -- an unexpected error class was found, see logs above.
    exit /b 2
) else if "%ANY_LINK_FAILURE%"=="1" (
    echo SUITE STATUS: LINK FAILURE -- one or more cells did not link against the real,
    echo shipped G5/CPU ABI. This is a real defect ^(a rotted source list, or a genuine missing
    echo production symbol^), never this suite's expected state -- see the per-cell LNK2019
    echo lines above.
    exit /b 1
) else (
    echo SUITE STATUS: LINKED CLEAN -- every cell links against the real, shipped G5/CPU ABI.
    exit /b 0
)
