@echo off
rem T-2899 (Curie) -- mutant runner for cell_adopt_prefix_census: builds the cell against three
rem scratch sslm_abi.cpp variants in place of the tree's own copy --
rem   FIXED         : the LIVE TIP's own src/sslm_abi.cpp, verbatim (Sec3.10.7's reference fix,
rem                   cumulative through T-2866/T-2894/T-2896/T-2897/T-2898, is already landed in
rem                   the committed tree). Must be the green tip.
rem   MUT_WALKRESET : the SAME file with T-2898's own reset line reverted to a no-op --
rem                   Sec3.10.3 row 11's own named mutant. Must be killed.
rem   MUT_FORCED    : the SAME fixed file with ONLY the `forced_token_count = 0;` line inside
rem                   `adopt_prefix` commented out -- the TE-364 mutant this cell exists to kill.
rem                   Must be killed.
rem
rem T-2916 (TE-370 M1): all three are now generated from the live tip by
rem `make_mut_adopt_prefix.py`, anchored on unique surrounding comments (never by line number,
rem which drifts) -- no external scratch directory (`D:\_t2909\refs_adopt`) is read any more, so
rem this runner has no out-of-repo input. Confirmed equivalent to the retired external copies by
rem `diff --strip-trailing-cr` before this switch (139 changed lines, all prose/comment, none
rem semantic).
rem
rem T-2909 (TE-365 S1): this runner used to `echo` a summary line and then unconditionally
rem `exit /b 0` -- a build failure, a regressed FIXED variant, or a SURVIVING mutant (zero
rem failures where a kill was required) all read as a clean run. Every variant now has an
rem expected verdict, and a mismatch, a build failure, or a crashed/summary-less run all set
rem OVERALL_OK=0, gating the script's own exit code.
rem
rem Usage: run_mutants_adopt_prefix.bat <path-to-model.sslm> [schema-name]
rem T-2916: dropped the <path-to-refs-dir> argument this runner used to take.
setlocal enabledelayedexpansion
set HEREDIR=%~dp0
set ENG=%HEREDIR%..\..
set TESTS=%ENG%\tests
set REFS=obj_mutants
set MODELARG=--model=%~1
if not "%~2"=="" set MODELARG=%MODELARG% --schema=%~2
call "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 -no_logo
cd /d "%HEREDIR%"
if not exist obj_mutants mkdir obj_mutants

set SRC_NOABI=%ENG%\src\artifact.cpp %ENG%\src\sha256.cpp %ENG%\src\tokenizer.cpp %ENG%\src\model.cpp ^
    %ENG%\src\intmath.cpp %ENG%\src\silu_lut.cpp %ENG%\src\matmul.cpp %ENG%\src\proof_manifest.cpp ^
    %ENG%\src\trace_hook.cpp %ENG%\src\forward\checked_chain_funnel.cpp ^
    %ENG%\src\forward\forward_sites.cpp %ENG%\src\decode_digest.cpp ^
    %ENG%\src\damped_greedy_antilm.cpp %ENG%\src\damped_greedy_topk.cpp ^
    %ENG%\src\damped_greedy_phaseD.cpp %ENG%\src\damped_greedy_phaseD_loop.cpp

if not defined SSLM_PYTHON set SSLM_PYTHON=C:\Users\dansu\AppData\Local\Programs\Python\Python313\python.exe
set OVERALL_OK=1
for %%m in (FIXED:sslm_abi_fixed.cpp MUT_WALKRESET:sslm_abi_mutant_walkreset.cpp MUT_FORCED:sslm_abi_mutant_forced.cpp) do (
    for /f "tokens=1,2 delims=:" %%a in ("%%m") do (
        "%SSLM_PYTHON%" make_mut_adopt_prefix.py %%a "%ENG%\src\sslm_abi.cpp" "obj_mutants\%%b" ^
            > "obj_mutants\make_mut_adopt_prefix_%%a.log" 2>&1
        if errorlevel 1 (echo BUILD FAILED: make_mut_adopt_prefix.py %%a & type "obj_mutants\make_mut_adopt_prefix_%%a.log" & set OVERALL_OK=0)
    )
)
for %%v in (FIXED MUT_WALKRESET MUT_FORCED) do (
    if "%%v"=="FIXED" (set ABIFILE=%REFS%\sslm_abi_fixed.cpp& set EXPECT=PASS)
    if "%%v"=="MUT_WALKRESET" (set ABIFILE=%REFS%\sslm_abi_mutant_walkreset.cpp& set EXPECT=KILL)
    if "%%v"=="MUT_FORCED" (set ABIFILE=%REFS%\sslm_abi_mutant_forced.cpp& set EXPECT=KILL)
    echo ===== %%v ^(!ABIFILE!^) expect=!EXPECT! =====
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
        set RUN_EC=!errorlevel!
        findstr /B "SUMMARY" "obj_mutants\%%v.runlog"
        set SUMMARY_LINE=
        for /f "delims=" %%s in ('findstr /R "^checks=[0-9]* failures=[0-9]*" "obj_mutants\%%v.runlog"') do set SUMMARY_LINE=%%s
        echo    !SUMMARY_LINE! ^(exit !RUN_EC!^)
        if "!SUMMARY_LINE!"=="" (
            echo    CRASHED OR NO SUMMARY LINE -- exit code !RUN_EC!
            set OVERALL_OK=0
        ) else (
            set TOK_CHECKS=
            set TOK_FAILURES=
            set TOK_SKIPS=
            for /f "tokens=1,2,3 delims= " %%a in ("!SUMMARY_LINE!") do (
                set TOK_CHECKS=%%a
                set TOK_FAILURES=%%b
                set TOK_SKIPS=%%c
            )
            set "FAILN=!TOK_FAILURES:~9!"
            set "SKIPN=!TOK_SKIPS:~6!"
            rem T-2909 (TE-365 S1 residual): cell_adopt_prefix_census's own summary line has a
            rem SEPARATE acceptance contract -- a row mismatch is reported in `mismatch=`/
            rem `acceptance=`, never asserted through CHECK/CHECK_MSG (that cell's own header
            rem comment: "the mutant runner checks THOSE, not GFailures"), so `failures=0` stays
            rem true under every mismatch. `ACCEPT0` is true whenever the line carries
            rem `acceptance=` and it is not `acceptance=1`.
            set ACCEPT0=0
            echo !SUMMARY_LINE! | findstr /C:"acceptance=" >nul
            if not errorlevel 1 (
                echo !SUMMARY_LINE! | findstr /C:"acceptance=1" >nul
                if errorlevel 1 set ACCEPT0=1
            )
            if "!EXPECT!"=="PASS" (
                if not "!FAILN!"=="0" (
                    echo    EXPECTED THE GREEN TIP, GOT FAILURES=!FAILN!
                    set OVERALL_OK=0
                )
                if not "!SKIPN!"=="0" (
                    echo    EXPECTED THE GREEN TIP, GOT SKIPS=!SKIPN!
                    set OVERALL_OK=0
                )
                if not "!RUN_EC!"=="0" (
                    echo    EXPECTED THE GREEN TIP, GOT NONZERO EXIT !RUN_EC!
                    set OVERALL_OK=0
                )
                if "!ACCEPT0!"=="1" (
                    echo    EXPECTED THE GREEN TIP, GOT ACCEPTANCE=0
                    set OVERALL_OK=0
                )
            ) else (
                if "!FAILN!"=="0" if "!ACCEPT0!"=="0" (
                    echo    SURVIVING MUTANT -- expected a kill, got failures=0 and no acceptance=0
                    set OVERALL_OK=0
                )
            )
        )
    )
)

if "%OVERALL_OK%"=="1" (
    echo ===== run_mutants_adopt_prefix: FIXED is the green tip, every mutant was killed =====
    exit /b 0
) else (
    echo ===== run_mutants_adopt_prefix: FAILURES ABOVE =====
    exit /b 1
)
