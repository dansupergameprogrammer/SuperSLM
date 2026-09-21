@echo off
rem T-2899 (Curie) -- builds every CPU-only cell in this suite against the tree AS IT STANDS
rem (Stage 1, curie/t2809-stage1-build) and runs it. Mirrors
rem tests/t2199-damped-greedy-red-suite/build_green_phaseD.bat's own source list (this suite's
rem cells call the identical public ABI, sslm_decode_step/sslm_seq_adopt_prefix/sslm_prefill).
rem
rem T-2909 (TE-365 S1): a build/link failure was the only way this script could fail before --
rem a nonzero exit code, a failures>0 count, or a skips>0 count in a cell's own summary line
rem were never consulted, only the summary line's PRESENCE (`findstr` alone). That is how C2
rem (a cell that printed `checks=0 failures=0 skips=0` and exited 0, asserting nothing) read as
rem a passing run. Every cell's exit code and its own parsed failures=/skips= counts now gate
rem OVERALL_OK; a skip fails the run exactly as a failure does (plan Sec3.5 step 1, T-2806 M2) --
rem an acceptance run supplies every fixture flag, so skips=0 is a claim this script checks
rem rather than a courtesy the runner prints and nobody reads. T-2909 also generates and wires
rem in the string-schema fixture TE-365 C2 needs (`make_t2909_string_schema_fixture.py`).
rem
rem Usage: build_red_suite.bat --model=PATH-TO-.sslm [--schema=NAME]
setlocal enabledelayedexpansion
set HEREDIR=%~dp0
set ENG=%HEREDIR%..\..
set TESTS=%ENG%\tests
call "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 -no_logo
cd /d "%HEREDIR%"
if not exist obj mkdir obj

set OVERALL_OK=1

rem T-2909: the string-schema fixture TE-365 C2's CPU twin needs (a real schema with a
rem string field; the C39 synthetic's own schema has no interior state -- see that
rem generator's own header comment). Regenerated fresh every run, never committed as a
rem binary blob, matching this suite's own S8-fixture discipline.
if not defined SSLM_PYTHON set SSLM_PYTHON=C:\Users\dansu\AppData\Local\Programs\Python\Python313\python.exe
"%SSLM_PYTHON%" make_t2909_string_schema_fixture.py "obj\t2909_string_schema_fixture.sslm" ^
    > obj\make_t2909_string_schema_fixture.log 2>&1
if errorlevel 1 (
    echo BUILD FAILED: make_t2909_string_schema_fixture.py
    type obj\make_t2909_string_schema_fixture.log
    set OVERALL_OK=0
)
set MODELARG=%* --stringschema=obj\t2909_string_schema_fixture.sslm

set SRC=%ENG%\src\artifact.cpp %ENG%\src\sha256.cpp %ENG%\src\tokenizer.cpp %ENG%\src\model.cpp ^
    %ENG%\src\intmath.cpp %ENG%\src\silu_lut.cpp %ENG%\src\matmul.cpp %ENG%\src\proof_manifest.cpp ^
    %ENG%\src\trace_hook.cpp %ENG%\src\forward\checked_chain_funnel.cpp ^
    %ENG%\src\forward\forward_sites.cpp %ENG%\src\decode_digest.cpp %ENG%\src\sslm_abi.cpp ^
    %ENG%\src\damped_greedy_antilm.cpp %ENG%\src\damped_greedy_topk.cpp ^
    %ENG%\src\damped_greedy_phaseD.cpp %ENG%\src\damped_greedy_phaseD_loop.cpp

for %%f in (cell_adopt_prefix_census cell_cpu_deadend_retry_reset) do (
    echo ===== %%f.cpp =====
    rem T-2905: SUPERSLM_CPU_G5_FINISH_ROW_FAULT_INJECTION defined for every cell in this loop --
    rem it gates ArmCpuFinishDegenerateLogitRowInjection's own definition in sslm_abi.cpp (%SRC%,
    rem recompiled fresh below for every cell) and cell_cpu_deadend_retry_reset.cpp's own
    rem extern "C" reservation of it (CpuCell2DegenerateRowTwin). Harmless for
    rem cell_adopt_prefix_census, which does not reference the symbol.
    cl /nologo /std:c++20 /O2 /W4 /fp:precise /EHsc /DSUPERSLM_CPU_G5_FINISH_ROW_FAULT_INJECTION ^
        /I%ENG%\include /I%ENG%\src /I%TESTS% /I%TESTS%\t2199-damped-greedy-red-suite /I. %SRC% ^
        "%%f.cpp" /Fo:"obj\\" /Fe:"obj\%%f.exe" ^
        /link > "obj\%%f.buildlog" 2>&1
    if errorlevel 1 (
        echo    BUILD FAILED:
        type "obj\%%f.buildlog"
        set OVERALL_OK=0
    ) else (
        "obj\%%f.exe" %MODELARG% > "obj\%%f.runlog" 2>&1
        set RUN_EC=!errorlevel!
        type "obj\%%f.runlog"
        set SUMMARY_LINE=
        for /f "delims=" %%s in ('findstr /R "^checks=[0-9]* failures=[0-9]*" "obj\%%f.runlog"') do set SUMMARY_LINE=%%s
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
            if not "!RUN_EC!"=="0" (
                echo    NONZERO EXIT CODE !RUN_EC! -- !SUMMARY_LINE!
                set OVERALL_OK=0
            )
            if not "!FAILN!"=="0" (
                echo    FAILURES=!FAILN! -- a red cell
                set OVERALL_OK=0
            )
            if not "!SKIPN!"=="0" (
                echo    SKIPS=!SKIPN! -- a skipped cell fails the run
                set OVERALL_OK=0
            )
            rem T-2909 (TE-365 S1 residual): cell_adopt_prefix_census's own summary line carries
            rem a SEPARATE acceptance contract -- its own header comment (Sec3.10.7's promise):
            rem "the mutant runner checks THOSE [rows/mismatch/acceptance], not GFailures" -- a
            rem row mismatch is reported, never asserted through CHECK/CHECK_MSG, so failures=0
            rem stays true under every mismatch. A summary line carrying `acceptance=` is only
            rem genuinely green at `acceptance=1`; harmless no-op for every cell that carries no
            rem such field.
            echo !SUMMARY_LINE! | findstr /C:"acceptance=" >nul
            if not errorlevel 1 (
                echo !SUMMARY_LINE! | findstr /C:"acceptance=1" >nul
                if errorlevel 1 (
                    echo    ACCEPTANCE=0 -- !SUMMARY_LINE!
                    set OVERALL_OK=0
                )
            )
        )
    )
)

if "%OVERALL_OK%"=="1" (
    echo ===== build_red_suite: all cells built and ran, zero failures, zero skips =====
    exit /b 0
) else (
    echo ===== build_red_suite: FAILURES ABOVE =====
    exit /b 1
)
