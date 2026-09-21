@echo off
rem T-2899 (Curie) -- mutant runner for cell_cpu_deadend_retry_reset: builds the cell against
rem four scratch sslm_abi.cpp variants --
rem   FIXED         : Claude/Vitruvius/t2898-probe/sslm_abi_cpu_fixed_v5.cpp verbatim (cumulative
rem                   reference fix through T-2866/T-2894/T-2896/T-2897/T-2898).
rem   MUT_NOREARM   : the SAME file with BOTH miss sites' `ready_for_logits = true;` reverted --
rem                   must turn Cell 1(iii) red on every route.
rem   MUT_NORESET   : the SAME file with BOTH miss sites' `state.layer_index = 0;` (T-2894's own
rem                   line) reverted -- must turn Cell 1(iv) red on exactly routes D and D1
rem                   while Cell 1(iii) STAYS GREEN on every route (the two assertions are
rem                   independent -- TE-361's own finding).
rem Usage: run_mutants_cpu_deadend.bat <path-to-refs-dir> <path-to-model.sslm>
setlocal enabledelayedexpansion
set HEREDIR=%~dp0
set ENG=%HEREDIR%..\..
set TESTS=%ENG%\tests
set REFS=%~1
set MODELARG=--model=%~2
call "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 -no_logo
cd /d "%HEREDIR%"
if not exist obj_mutants_cpu mkdir obj_mutants_cpu

set SRC_NOABI=%ENG%\src\artifact.cpp %ENG%\src\sha256.cpp %ENG%\src\tokenizer.cpp %ENG%\src\model.cpp ^
    %ENG%\src\intmath.cpp %ENG%\src\silu_lut.cpp %ENG%\src\matmul.cpp %ENG%\src\proof_manifest.cpp ^
    %ENG%\src\trace_hook.cpp %ENG%\src\forward\checked_chain_funnel.cpp ^
    %ENG%\src\forward\forward_sites.cpp %ENG%\src\decode_digest.cpp ^
    %ENG%\src\damped_greedy_antilm.cpp %ENG%\src\damped_greedy_topk.cpp ^
    %ENG%\src\damped_greedy_phaseD.cpp %ENG%\src\damped_greedy_phaseD_loop.cpp

for %%v in (FIXED MUT_NOREARM MUT_NORESET) do (
    if "%%v"=="FIXED" set ABIFILE=%REFS%\sslm_abi_fixed.cpp
    if "%%v"=="MUT_NOREARM" set ABIFILE=%REFS%\sslm_abi_mutant_norearm.cpp
    if "%%v"=="MUT_NORESET" set ABIFILE=%REFS%\sslm_abi_mutant_noreset.cpp
    echo ===== %%v ^(!ABIFILE!^) =====
    if not exist "obj_mutants_cpu\%%v" mkdir "obj_mutants_cpu\%%v"
    cl /nologo /std:c++20 /O2 /W4 /fp:precise /EHsc ^
        /I%ENG%\include /I%ENG%\src /I%TESTS% /I. %SRC_NOABI% "!ABIFILE!" ^
        cell_cpu_deadend_retry_reset.cpp /Fo:"obj_mutants_cpu\%%v\\" /Fe:"obj_mutants_cpu\%%v.exe" ^
        /link > "obj_mutants_cpu\%%v.buildlog" 2>&1
    if errorlevel 1 (
        echo    BUILD FAILED:
        type "obj_mutants_cpu\%%v.buildlog"
    ) else (
        "obj_mutants_cpu\%%v.exe" %MODELARG% > "obj_mutants_cpu\%%v.runlog" 2>&1
        findstr /B "FAIL" "obj_mutants_cpu\%%v.runlog"
        findstr /B "checks=" "obj_mutants_cpu\%%v.runlog"
    )
)
echo ===== run_mutants_cpu_deadend: done =====
exit /b 0
