@echo off
rem T-2916 (Curie) -- TE-370 S3: builds and runs cell_gpu_slm5_bounds_tamper against the tree's
rem own gpu_1p0.cpp/superslm_gpu.cpp (ASBUILT -- the checks exist at 0062c99, so this must be
rem GREEN), then against three mutants, each with exactly one of the three bounds checks deleted
rem from a scratch copy of gpu_1p0.cpp (make_mut_slm5_bounds.py) -- each MUST be killed (at least
rem one CHECK failure). Self-contained: builds its own CPU_COMMON and ASBUILT GPU object sets
rem fresh, independent of this suite's own build_red_suite_gpu.bat (which this cell does not need
rem any of the FIXED/MUT_CHECKEDRETURN/refs machinery from -- the checks under test already exist
rem in the live tree).
rem
rem Usage: build_t2916_s3_bounds_tamper.bat <path-to-C39.sslm>
setlocal enabledelayedexpansion
set HEREDIR=%~dp0
set ENG=%HEREDIR%..\..
set TESTS=%ENG%\tests
set MODELARG=--model=%~1
set MUTDIR=D:\_t2916\mutants
call "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 -no_logo
cd /d "%HEREDIR%"
if not exist obj_t2916_s3 mkdir obj_t2916_s3
if not exist bin_t2916_s3 mkdir bin_t2916_s3
if not exist bin_t2916_s3\shaders xcopy /E /I /Q /Y "%ENG%\build\gpu-shaders-staged" bin_t2916_s3\shaders >nul
if not exist %MUTDIR% mkdir %MUTDIR%

set SRC_NOABI=%ENG%\src\artifact.cpp %ENG%\src\sha256.cpp %ENG%\src\tokenizer.cpp %ENG%\src\model.cpp ^
    %ENG%\src\intmath.cpp %ENG%\src\silu_lut.cpp %ENG%\src\matmul.cpp %ENG%\src\proof_manifest.cpp ^
    %ENG%\src\trace_hook.cpp %ENG%\src\forward\checked_chain_funnel.cpp ^
    %ENG%\src\forward\forward_sites.cpp %ENG%\src\decode_digest.cpp ^
    %ENG%\src\damped_greedy_antilm.cpp %ENG%\src\damped_greedy_topk.cpp ^
    %ENG%\src\damped_greedy_phaseD.cpp %ENG%\src\damped_greedy_phaseD_loop.cpp

set GPUINC=%TESTS%\t2791-gpu-prefill-read-red-suite
set STOCKINC=/I%ENG%\include /I%ENG%\src /I%TESTS% /I%GPUINC% /I.
set SYSLIBS=d3d12.lib dxgi.lib dxguid.lib
set OVERALL_OK=1

echo ================= T-2916 S3: shared CPU_COMMON + ASBUILT GPU objects =================
if not exist obj_t2916_s3\cpu_common mkdir obj_t2916_s3\cpu_common
cl /nologo /std:c++20 /O2 /W3 /fp:precise /EHsc /I%ENG%\include /I%ENG%\src ^
    /c %SRC_NOABI% /Fo"obj_t2916_s3\cpu_common\\" ^
    > obj_t2916_s3\cpu_common.buildlog 2>&1 || (echo BUILD FAILED: cpu_common & type obj_t2916_s3\cpu_common.buildlog & set OVERALL_OK=0)
set CPU_COMMON_OBJS=
for %%f in (%SRC_NOABI%) do (
    for %%n in (%%f) do set CPU_COMMON_OBJS=!CPU_COMMON_OBJS! obj_t2916_s3\cpu_common\%%~nn.obj
)
cl /nologo /std:c++20 /O2 /W3 /fp:precise /EHsc /I%ENG%\include /I%ENG%\src /c %ENG%\src\sslm_abi.cpp ^
    /Fo"obj_t2916_s3\sslm_abi.obj" > obj_t2916_s3\sslm_abi.buildlog 2>&1 || (echo BUILD FAILED: sslm_abi & type obj_t2916_s3\sslm_abi.buildlog & set OVERALL_OK=0)

if not exist obj_t2916_s3\gpu_asbuilt mkdir obj_t2916_s3\gpu_asbuilt
cl /nologo /std:c++20 /O2 /W3 /fp:precise /EHsc /I%ENG%\include /I%ENG%\src\gpu ^
    /c %ENG%\src\gpu\gpu_1p0.cpp %ENG%\src\gpu\superslm_gpu.cpp /Fo"obj_t2916_s3\gpu_asbuilt\\" ^
    > obj_t2916_s3\gpu_asbuilt.buildlog 2>&1 || (echo BUILD FAILED: gpu_asbuilt & type obj_t2916_s3\gpu_asbuilt.buildlog & set OVERALL_OK=0)

echo ================= Generating three MUT_SLM5 variants of gpu_1p0.cpp =================
if not defined SSLM_PYTHON set SSLM_PYTHON=C:\Users\dansu\AppData\Local\Programs\Python\Python313\python.exe
for %%c in (INDEXCOUNT WALKSTATE UNBOUNDWALK) do (
    "%SSLM_PYTHON%" make_mut_slm5_bounds.py %%c "%ENG%\src\gpu\gpu_1p0.cpp" "%MUTDIR%\gpu_1p0_mut_%%c.cpp" ^
        > "obj_t2916_s3\make_mut_%%c.log" 2>&1
    if errorlevel 1 (
        echo GENERATE FAILED: mut_%%c & type "obj_t2916_s3\make_mut_%%c.log" & set OVERALL_OK=0
    ) else (
        if not exist "obj_t2916_s3\gpu_mut_%%c" mkdir "obj_t2916_s3\gpu_mut_%%c"
        cl /nologo /std:c++20 /O2 /W3 /fp:precise /EHsc /I%ENG%\include /I%ENG%\src\gpu ^
            /c "%MUTDIR%\gpu_1p0_mut_%%c.cpp" /Fo"obj_t2916_s3\gpu_mut_%%c\\" ^
            > "obj_t2916_s3\gpu_mut_%%c.buildlog" 2>&1 || (echo BUILD FAILED: gpu_mut_%%c & type "obj_t2916_s3\gpu_mut_%%c.buildlog" & set OVERALL_OK=0)
    )
)

echo ================= Building cell_gpu_slm5_bounds_tamper =================
cl /nologo /std:c++20 /O2 /W3 /fp:precise /EHsc %STOCKINC% /c cell_gpu_slm5_bounds_tamper.cpp ^
    /Fo"obj_t2916_s3\cell.obj" > obj_t2916_s3\cell.buildlog 2>&1 || (echo BUILD FAILED: cell.obj & type obj_t2916_s3\cell.buildlog & set OVERALL_OK=0)

rem MUT_INDEXCOUNT's own EXPECT is SURVIVE_REDUNDANT, not KILL: executed (see this run's own
rem history), deleting ONLY the index>=schemas.Count() check does not change any observable
rem status, because `SchemaMasksTable::ByIndex(size_t)` (include/superslm/schema_masks.h) is
rem itself bounds-safe (`index < entries_.size() ? &entries_[index] : nullptr`) and every
rem consumer of `bound_schema_index` in this file (lines 2258/2475/3214, plus this restore path)
rem routes through it and null-checks the result -- so ANY out-of-range index that check 1 would
rem catch is, by construction, ALSO caught by check 2's own `!resolved_entry` guard. This is a
rem genuine, provable redundancy (an index >= Count() can never make ByIndex return non-null), not
rem a gap in this cell -- documented as a finding in Claude/Curie/t2916-te370-red-cells-2026-09-21.md,
rem not silently forced to a fabricated KILL.
for %%v in (ASBUILT MUT_INDEXCOUNT MUT_WALKSTATE MUT_UNBOUNDWALK) do (
    if "%%v"=="ASBUILT" (set GPUOBJS=obj_t2916_s3\gpu_asbuilt\gpu_1p0.obj obj_t2916_s3\gpu_asbuilt\superslm_gpu.obj& set EXPECT=PASS)
    if "%%v"=="MUT_INDEXCOUNT" (set GPUOBJS=obj_t2916_s3\gpu_mut_INDEXCOUNT\gpu_1p0_mut_INDEXCOUNT.obj obj_t2916_s3\gpu_asbuilt\superslm_gpu.obj& set EXPECT=SURVIVE_REDUNDANT)
    if "%%v"=="MUT_WALKSTATE" (set GPUOBJS=obj_t2916_s3\gpu_mut_WALKSTATE\gpu_1p0_mut_WALKSTATE.obj obj_t2916_s3\gpu_asbuilt\superslm_gpu.obj& set EXPECT=KILL)
    if "%%v"=="MUT_UNBOUNDWALK" (set GPUOBJS=obj_t2916_s3\gpu_mut_UNBOUNDWALK\gpu_1p0_mut_UNBOUNDWALK.obj obj_t2916_s3\gpu_asbuilt\superslm_gpu.obj& set EXPECT=KILL)
    echo ===== cell_gpu_slm5_bounds_tamper [%%v] expect=!EXPECT! =====
    link /nologo /OUT:"bin_t2916_s3\%%v.exe" obj_t2916_s3\cell.obj !GPUOBJS! !CPU_COMMON_OBJS! obj_t2916_s3\sslm_abi.obj %SYSLIBS% ^
        > "obj_t2916_s3\%%v.linklog" 2>&1
    if errorlevel 1 (
        echo    LINK FAILED: & type "obj_t2916_s3\%%v.linklog" & set OVERALL_OK=0
    ) else (
        "bin_t2916_s3\%%v.exe" %MODELARG% > "obj_t2916_s3\%%v.runlog" 2>&1
        type "obj_t2916_s3\%%v.runlog"
        set SUMMARY_LINE=
        for /f "delims=" %%s in ('findstr /R "^checks=[0-9]* failures=[0-9]*" "obj_t2916_s3\%%v.runlog"') do set SUMMARY_LINE=%%s
        if "!SUMMARY_LINE!"=="" (
            echo    CRASHED OR NO SUMMARY LINE
            set OVERALL_OK=0
        ) else (
            set TOK_CHECKS=
            set TOK_FAILURES=
            set TOK_SKIPS=
            for /f "tokens=1,2,3 delims= " %%a in ("!SUMMARY_LINE!") do (set TOK_CHECKS=%%a& set TOK_FAILURES=%%b& set TOK_SKIPS=%%c)
            set "FAILN=!TOK_FAILURES:~9!"
            set "SKIPN=!TOK_SKIPS:~6!"
            if not "!SKIPN!"=="0" (echo    SKIPS=!SKIPN! -- a skipped cell fails the run ^("obj_t2916_s3\%%v.runlog"^)& set OVERALL_OK=0)
            if "!EXPECT!"=="PASS" (
                if not "!FAILN!"=="0" (echo    FAILURES=!FAILN! -- the tree's own checks must be green ^("obj_t2916_s3\%%v.runlog"^)& set OVERALL_OK=0)
            )
            if "!EXPECT!"=="KILL" (
                if "!FAILN!"=="0" (echo    SURVIVING MUTANT -- expected a kill, got failures=0 ^("obj_t2916_s3\%%v.runlog"^)& set OVERALL_OK=0)
            )
            if "!EXPECT!"=="SURVIVE_REDUNDANT" (
                if not "!FAILN!"=="0" (echo    UNEXPECTED KILL -- MUT_INDEXCOUNT was expected to survive as a documented redundancy; it killed instead, meaning the redundancy claim needs re-checking ^("obj_t2916_s3\%%v.runlog"^)& set OVERALL_OK=0) else (echo    confirmed: MUT_INDEXCOUNT survives as expected -- the index^>=Count^(^) check is redundant with check 2's own null-guard, per ByIndex's bounds safety)
            )
        )
    )
)

if "%OVERALL_OK%"=="1" (
    echo ===== build_t2916_s3_bounds_tamper: ASBUILT is green; MUT_WALKSTATE/MUT_UNBOUNDWALK killed; MUT_INDEXCOUNT survives as the documented redundancy =====
    exit /b 0
) else (
    echo ===== build_t2916_s3_bounds_tamper: FAILURES ABOVE =====
    exit /b 1
)
