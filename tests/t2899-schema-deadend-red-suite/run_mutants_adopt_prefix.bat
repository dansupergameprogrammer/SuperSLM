@echo off
rem T-2899 (Curie) -- mutant runner for cell_adopt_prefix_census: builds the cell against three
rem scratch sslm_abi.cpp variants in place of the tree's own copy --
rem   FIXED         : Claude/Vitruvius/t2898-probe/sslm_abi_cpu_fixed_v5.cpp verbatim (Sec3.10.7's
rem                   reference fix, cumulative through T-2866/T-2894/T-2896/T-2897/T-2898).
rem   MUT_WALKRESET : the SAME file with T-2898's own reset line reverted to a no-op
rem                   (Claude/Vitruvius/t2898-probe/sslm_abi_cpu_mutant_v5.cpp verbatim) --
rem                   Sec3.10.3 row 11's own named mutant.
rem   MUT_FORCED    : the SAME fixed file with ONLY line 2018's `forced_token_count = 0;`
rem                   commented out -- the TE-364 mutant this cell exists to kill.
rem Usage: run_mutants_adopt_prefix.bat <path-to-refs-dir> <path-to-model.sslm> [schema-name]
setlocal enabledelayedexpansion
set HEREDIR=%~dp0
set ENG=%HEREDIR%..\..
set TESTS=%ENG%\tests
set REFS=%~1
set MODELARG=--model=%~2
if not "%~3"=="" set MODELARG=%MODELARG% --schema=%~3
call "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 -no_logo
cd /d "%HEREDIR%"
if not exist obj_mutants mkdir obj_mutants

set SRC_NOABI=%ENG%\src\artifact.cpp %ENG%\src\sha256.cpp %ENG%\src\tokenizer.cpp %ENG%\src\model.cpp ^
    %ENG%\src\intmath.cpp %ENG%\src\silu_lut.cpp %ENG%\src\matmul.cpp %ENG%\src\proof_manifest.cpp ^
    %ENG%\src\trace_hook.cpp %ENG%\src\forward\checked_chain_funnel.cpp ^
    %ENG%\src\forward\forward_sites.cpp %ENG%\src\decode_digest.cpp ^
    %ENG%\src\damped_greedy_antilm.cpp %ENG%\src\damped_greedy_topk.cpp ^
    %ENG%\src\damped_greedy_phaseD.cpp %ENG%\src\damped_greedy_phaseD_loop.cpp

set OVERALL_OK=1
for %%v in (FIXED MUT_WALKRESET MUT_FORCED) do (
    if "%%v"=="FIXED" set ABIFILE=%REFS%\sslm_abi_fixed.cpp
    if "%%v"=="MUT_WALKRESET" set ABIFILE=%REFS%\sslm_abi_mutant_walkreset.cpp
    if "%%v"=="MUT_FORCED" set ABIFILE=%REFS%\sslm_abi_mutant_forced.cpp
    echo ===== %%v ^(!ABIFILE!^) =====
    if not exist "obj_mutants\%%v" mkdir "obj_mutants\%%v"
    cl /nologo /std:c++20 /O2 /W4 /fp:precise /EHsc ^
        /I%ENG%\include /I%ENG%\src /I%TESTS% /I%TESTS%\t2199-damped-greedy-red-suite /I. %SRC_NOABI% "!ABIFILE!" ^
        cell_adopt_prefix_census.cpp /Fo:"obj_mutants\%%v\\" /Fe:"obj_mutants\%%v.exe" ^
        /link > "obj_mutants\%%v.buildlog" 2>&1
    if errorlevel 1 (
        echo    BUILD FAILED:
        type "obj_mutants\%%v.buildlog"
        set OVERALL_OK=0
    ) else (
        "obj_mutants\%%v.exe" %MODELARG% > "obj_mutants\%%v.runlog" 2>&1
        findstr /B "SUMMARY" "obj_mutants\%%v.runlog"
        findstr /B "checks=" "obj_mutants\%%v.runlog"
    )
)

echo ===== run_mutants_adopt_prefix: done =====
exit /b 0
