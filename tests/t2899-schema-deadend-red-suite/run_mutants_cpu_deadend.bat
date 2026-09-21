@echo off
rem T-2899 (Curie) -- mutant runner for cell_cpu_deadend_retry_reset: builds the cell against
rem scratch sslm_abi.cpp variants --
rem   FIXED         : the LIVE TIP's own src/sslm_abi.cpp, verbatim (every fold this once needed
rem                   an external reference for -- T-2866/T-2894/T-2896/T-2897/T-2898 -- is
rem                   already landed in the committed tree). Every assertion in the cell,
rem                   including TE-365 C2's own degenerate-row twin, must be GREEN.
rem   MUT_NOREARM   : the SAME file with BOTH miss sites' `ready_for_logits = true;` reverted --
rem                   must turn Cell 1(iii) red on every route.
rem   MUT_NORESET   : the SAME file with BOTH miss sites' `state.layer_index = 0;` (T-2894's own
rem                   line) reverted -- must turn Cell 1(iv) red on exactly routes D and D1
rem                   while Cell 1(iii) STAYS GREEN on every route (the two assertions are
rem                   independent -- TE-361's own finding).
rem   T-2916 (TE-370 M1): FIXED/MUT_NOREARM/MUT_NORESET are now generated from the live tip by
rem   `make_mut_cpu_deadend.py` (T-2909 built this generator already; this runner had simply
rem   never been switched over to call it) -- no external scratch directory (`D:\_t2909\refs`)
rem   is read any more, so this runner has no out-of-repo input.
rem   MUT_NOSEAM    : T-2909's own guard-vitality mutant for TE-365 C2 -- a fresh copy of the
rem                   LIVE TIP's sslm_abi.cpp (%ENG%\src\sslm_abi.cpp, generated here by
rem                   make_mut_noseam.py, never staged by the planner) with ONLY the seam's own
rem                   consumption call deleted (`MaybeInjectCpuFinishDegenerateLogitRow`, the
rem                   greedy branch's masked-argmax site). Must turn Cell 2(i)/(ii) red while
rem                   every other cell in the same binary stays green -- the CPU-side stand-in
rem                   for plan Sec3.10.3 row 11's "checked return" mutant (CPU's own
rem                   `has_transition` check was already correct pre-fix, Sec3.10.4's own cost
rem                   table; the seam's consumption call is the only thing this cell's own
rem                   assertions depend on that a regression could silently drop).
rem
rem T-2909 (TE-365 S1): this runner used to `echo` a summary line and then unconditionally
rem `exit /b 0` -- a build failure, a FIXED variant that regressed, or a mutant that SURVIVED
rem (zero failures where a kill was required) all read as a clean run. Every variant now has an
rem expected verdict -- FIXED must be the green tip (checks>0, failures=0, skips=0); every
rem MUT_* must be KILLED (failures>0) -- and a mismatch, a build failure, or a crashed/summary-
rem less run all set OVERALL_OK=0, gating the script's own exit code.
rem
rem Usage: run_mutants_cpu_deadend.bat <path-to-model.sslm> <path-to-stringschema.sslm>
rem T-2916 (TE-370 M1): dropped the <path-to-refs-dir> argument this runner used to take --
rem every source configuration below is now generated from the tracked tree, no scratch
rem directory input needed.
setlocal enabledelayedexpansion
set HEREDIR=%~dp0
set ENG=%HEREDIR%..\..
set TESTS=%ENG%\tests
set MODELARG=--model=%~1 --stringschema=%~2
call "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 -no_logo
cd /d "%HEREDIR%"
if not exist obj_mutants_cpu mkdir obj_mutants_cpu

set SRC_NOABI=%ENG%\src\artifact.cpp %ENG%\src\sha256.cpp %ENG%\src\tokenizer.cpp %ENG%\src\model.cpp ^
    %ENG%\src\intmath.cpp %ENG%\src\silu_lut.cpp %ENG%\src\matmul.cpp %ENG%\src\proof_manifest.cpp ^
    %ENG%\src\trace_hook.cpp %ENG%\src\forward\checked_chain_funnel.cpp ^
    %ENG%\src\forward\forward_sites.cpp %ENG%\src\decode_digest.cpp ^
    %ENG%\src\damped_greedy_antilm.cpp %ENG%\src\damped_greedy_topk.cpp ^
    %ENG%\src\damped_greedy_phaseD.cpp %ENG%\src\damped_greedy_phaseD_loop.cpp

set REFS=obj_mutants_cpu
if not defined SSLM_PYTHON set SSLM_PYTHON=C:\Users\dansu\AppData\Local\Programs\Python\Python313\python.exe
set OVERALL_OK=1
"%SSLM_PYTHON%" make_mut_noseam.py "%ENG%\src\sslm_abi.cpp" "obj_mutants_cpu\sslm_abi_mut_noseam.cpp" ^
    > obj_mutants_cpu\make_mut_noseam.log 2>&1
if errorlevel 1 (echo BUILD FAILED: make_mut_noseam.py & type obj_mutants_cpu\make_mut_noseam.log & set OVERALL_OK=0)
rem T-2916: FIXED/MUT_NOREARM/MUT_NORESET generated from the live tip (make_mut_cpu_deadend.py
rem already existed for this purpose; this runner had simply never been switched over to call it
rem instead of an external scratch refs directory).
for %%m in (FIXED:sslm_abi_fixed.cpp NOREARM:sslm_abi_mutant_norearm.cpp NORESET:sslm_abi_mutant_noreset.cpp) do (
    for /f "tokens=1,2 delims=:" %%a in ("%%m") do (
        "%SSLM_PYTHON%" make_mut_cpu_deadend.py "%ENG%\src\sslm_abi.cpp" %%a "obj_mutants_cpu\%%b" ^
            > "obj_mutants_cpu\make_mut_cpu_deadend_%%a.log" 2>&1
        if errorlevel 1 (echo BUILD FAILED: make_mut_cpu_deadend.py %%a & type "obj_mutants_cpu\make_mut_cpu_deadend_%%a.log" & set OVERALL_OK=0)
    )
)

for %%v in (FIXED MUT_NOREARM MUT_NORESET MUT_NOSEAM) do (
    if "%%v"=="FIXED" (set ABIFILE=%REFS%\sslm_abi_fixed.cpp& set EXPECT=PASS)
    if "%%v"=="MUT_NOREARM" (set ABIFILE=%REFS%\sslm_abi_mutant_norearm.cpp& set EXPECT=KILL)
    if "%%v"=="MUT_NORESET" (set ABIFILE=%REFS%\sslm_abi_mutant_noreset.cpp& set EXPECT=KILL)
    if "%%v"=="MUT_NOSEAM" (set ABIFILE=obj_mutants_cpu\sslm_abi_mut_noseam.cpp& set EXPECT=KILL)
    echo ===== %%v ^(!ABIFILE!^) expect=!EXPECT! =====
    if not exist "obj_mutants_cpu\%%v" mkdir "obj_mutants_cpu\%%v"
    cl /nologo /std:c++20 /O2 /W4 /fp:precise /EHsc /DSUPERSLM_CPU_G5_FINISH_ROW_FAULT_INJECTION ^
        /I%ENG%\include /I%ENG%\src /I%TESTS% /I. %SRC_NOABI% "!ABIFILE!" ^
        cell_cpu_deadend_retry_reset.cpp /Fo:"obj_mutants_cpu\%%v\\" /Fe:"obj_mutants_cpu\%%v.exe" ^
        /link > "obj_mutants_cpu\%%v.buildlog" 2>&1
    if errorlevel 1 (
        echo    BUILD FAILED:
        type "obj_mutants_cpu\%%v.buildlog"
        set OVERALL_OK=0
    ) else (
        "obj_mutants_cpu\%%v.exe" %MODELARG% > "obj_mutants_cpu\%%v.runlog" 2>&1
        set RUN_EC=!errorlevel!
        findstr /B "FAIL" "obj_mutants_cpu\%%v.runlog"
        set SUMMARY_LINE=
        for /f "delims=" %%s in ('findstr /R "^checks=[0-9]* failures=[0-9]*" "obj_mutants_cpu\%%v.runlog"') do set SUMMARY_LINE=%%s
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
            ) else (
                if "!FAILN!"=="0" (
                    echo    SURVIVING MUTANT -- expected a kill, got failures=0
                    set OVERALL_OK=0
                )
            )
        )
    )
)

if "%OVERALL_OK%"=="1" (
    echo ===== run_mutants_cpu_deadend: FIXED is the green tip, every mutant was killed =====
    exit /b 0
) else (
    echo ===== run_mutants_cpu_deadend: FAILURES ABOVE =====
    exit /b 1
)
