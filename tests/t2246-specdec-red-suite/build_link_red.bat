@echo off
rem T-2246 (test design): builds every C++ cell file in this directory against the REAL engine
rem (the identical CPU source list tests/t2138-abi-red-suite/build_link_red.bat uses) and
rem reports each cell's red status HONESTLY. This suite is a RED-FIRST PRE-IMPLEMENTATION
rem GATE: the speculative-decoding mechanism (plan r6 SS3/SS5 S-E) does not exist at pin
rem f409bda, so every cell is expected to fail at LINK on exactly the symbols recorded in
rem sslm_specdec_red_contract.h's EXPECTED-MISSING SYMBOL SET.
rem
rem Per-cell classification (the N1 lesson from t2138 applied symbol-scoped -- a link failure
rem is never silently absorbed, and an expected-red state is never faked):
rem   - COMPILE ERROR (error C...)                          -> DEFECT, exit 2.
rem   - LINK failure whose EVERY unresolved external names
rem     a recorded expected-missing symbol                  -> LINK RED (EXPECTED PRE-BUILD).
rem   - LINK failure naming ANYTHING ELSE                   -> DEFECT (stale source list or an
rem                                                            unexpected production gap), exit 1.
rem   - LINKED CLEAN                                        -> UNEXPECTED PRE-BUILD; the cell's
rem                                                            _v3 references resolved against
rem                                                            REAL symbols, so it is executed
rem                                                            with any supplied artifacts and
rem                                                            its pass/fail counts reported.
rem
rem Suite status exits 0 ONLY when every cell is link-red-as-expected (or clean-linked and
rem executed green). Anything else is loud and non-zero.
setlocal enabledelayedexpansion
set HEREDIR=%~dp0
set ENG=%HEREDIR%..\..
call "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 -no_logo
cd /d "%HEREDIR%"
if not exist obj mkdir obj

set ANY_COMPILE_ERROR=0
set ANY_UNEXPECTED_LINK=0
set CELLS_RED=0
set CELLS_CLEAN=0

for %%f in (dim01_equivalence_red.cpp dim02_rollback_red.cpp dim03_retention_red.cpp dim04_drafter_red.cpp dim05_trust_red.cpp dim06_concurrency_red.cpp dim07_failure_red.cpp dim08_guard_red.cpp dim09_persistence_red.cpp dim10_composition_red.cpp dim11_matrix_red.cpp dim12_performance_red.cpp) do (
    echo ===== %%f =====
    cl /nologo /std:c++20 /O2 /W4 /fp:precise /EHsc /I%ENG%\include /I%ENG%\tests\support /I. ^
        %ENG%\src\artifact.cpp %ENG%\src\sha256.cpp %ENG%\src\tokenizer.cpp %ENG%\src\model.cpp ^
        %ENG%\src\intmath.cpp %ENG%\src\silu_lut.cpp %ENG%\src\matmul.cpp %ENG%\src\proof_manifest.cpp ^
        %ENG%\src\trace_hook.cpp %ENG%\src\forward\checked_chain_funnel.cpp ^
        %ENG%\src\forward\forward_sites.cpp %ENG%\src\decode_digest.cpp %ENG%\src\sslm_abi.cpp ^
        %ENG%\src\damped_greedy_antilm.cpp %ENG%\src\damped_greedy_topk.cpp ^
        %ENG%\src\damped_greedy_phaseD.cpp %ENG%\src\damped_greedy_phaseD_loop.cpp ^
        "%%f" /Fo:"obj\\" /Fe:"obj\%%~nf.exe" ^
        /link > "obj\%%~nf.log" 2>&1
    findstr /C:"error C" "obj\%%~nf.log" >nul
    if not errorlevel 1 (
        echo    COMPILE ERROR -- unexpected, every cell must compile clean against the real
        echo    sslm_abi.h plus the promoted contract header:
        type "obj\%%~nf.log"
        set ANY_COMPILE_ERROR=1
    ) else (
        findstr /C:"LNK2019" /C:"LNK1120" /C:"LNK2001" "obj\%%~nf.log" >nul
        if not errorlevel 1 (
            rem Count unresolved externals total vs those naming expected-missing symbols.
            for /f %%t in ('findstr /C:"LNK2019" "obj\%%~nf.log" ^| find /c /v ""') do set TOTAL_MISS=%%t
            for /f %%e in ('findstr /C:"LNK2019" "obj\%%~nf.log" ^| findstr /C:"speculate_step_v3" /C:"speculate_params_init" /C:"committed_token_count" /C:"committed_tokens_peek" /C:"SpecdecDraftPropose" /C:"g_inject_specdec_fault" ^| find /c /v ""') do set EXPECTED_MISS=%%e
            if !TOTAL_MISS! EQU !EXPECTED_MISS! (
                if !TOTAL_MISS! GTR 0 (
                    echo    LINK RED (EXPECTED PRE-BUILD^) -- !TOTAL_MISS! unresolved external(s^), all recorded expectations:
                    findstr /C:"LNK2019" "obj\%%~nf.log"
                    set /a CELLS_RED+=1
                ) else (
                    echo    *** LINK FAILURE *** LNK1120 without per-symbol lines -- see log:
                    type "obj\%%~nf.log"
                    set ANY_UNEXPECTED_LINK=1
                )
            ) else (
                echo    *** UNEXPECTED MISSING SYMBOLS *** ^(a defect: stale source list or an
                echo    unexpected production-symbol gap -- never this suite's expected state^):
                findstr /C:"LNK2019" "obj\%%~nf.log"
                set ANY_UNEXPECTED_LINK=1
            )
        ) else (
            findstr /C:"error" "obj\%%~nf.log" >nul
            if not errorlevel 1 (
                echo    UNEXPECTED ERROR CLASS -- see obj\%%~nf.log
                type "obj\%%~nf.log"
                set ANY_COMPILE_ERROR=1
            ) else (
                echo    LINKED CLEAN PRE-BUILD -- UNEXPECTED: _v3 references resolved against
                echo    REAL symbols. Executing with any supplied artifacts to observe behavior:
                if exist "obj\%%~nf.exe" (
                    "obj\%%~nf.exe" !T2246_ARGS! >> "obj\%%~nf.runlog" 2>&1
                    set RUN_EC=!errorlevel!
                    findstr /R "^checks=[0-9]* failures=[0-9]* skips=[0-9]*$" "obj\%%~nf.runlog"
                    if not "!RUN_EC!"=="0" (
                        echo    EXECUTED WITH FAILURES ^(exit !RUN_EC!^) -- honest red-by-behavior.
                        set /a CELLS_RED+=1
                    ) else (
                        echo    EXECUTED GREEN pre-build -- investigate immediately.
                        set /a CELLS_CLEAN+=1
                    )
                )
            )
        )
    )
)

echo.
echo Cells link-red as expected: !CELLS_RED!   linked-clean-and-executed: !CELLS_CLEAN!
if "%ANY_COMPILE_ERROR%"=="1" (
    echo SUITE STATUS: COMPILE ERROR -- unexpected error class found, see logs above.
    exit /b 2
)
if "%ANY_UNEXPECTED_LINK%"=="1" (
    echo SUITE STATUS: UNEXPECTED LINK FAILURE -- one or more cells failed to resolve
    echo symbols OUTSIDE the recorded expectation. Real defect, see per-cell logs above.
    exit /b 1
)
echo SUITE STATUS: RED BY LINK AS EXPECTED (pre-build) -- every cell failed on exactly the
echo recorded expected-missing symbols in sslm_specdec_red_contract.h.
exit /b 0
