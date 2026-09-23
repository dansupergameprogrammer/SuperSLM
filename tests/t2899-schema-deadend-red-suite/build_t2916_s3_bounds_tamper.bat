@echo off
rem T-2916/T-2917 (Curie/Brunel) -- TE-370 S3: builds and runs cell_gpu_slm5_bounds_tamper against
rem the tree's own gpu_1p0.cpp/superslm_gpu.cpp (ASBUILT, must be GREEN), then against two
rem mutants, each with one of the two REMAINING bounds checks deleted from a scratch copy of
rem gpu_1p0.cpp (make_mut_slm5_bounds.py) -- each MUST be killed (at least one CHECK failure).
rem T-2917 (D-SLM7600): the third check this runner used to test as MUT_INDEXCOUNT (its own
rem deletion mutant provably survives -- SchemaMasksTable::ByIndex is bounds-safe and the
rem adjacent null-check catches everything it would have) is REMOVED from the tree, replaced by a
rem comment naming the null-check as the bound; there is no longer a check to mutate. Check 1's
rem own tamper input (schema_index=999999) is still exercised below, as part of ASBUILT, and is
rem still refused, now by check 2's own null-guard alone. Self-contained: builds its own
rem CPU_COMMON and ASBUILT GPU object sets fresh, independent of this suite's own
rem build_red_suite_gpu.bat (which this cell does not need any of the
rem FIXED/MUT_CHECKEDRETURN/refs machinery from -- the checks under test already exist in the
rem live tree).
rem
rem Usage: build_t2916_s3_bounds_tamper.bat <path-to-C39.sslm> [<shaders-dir>]
setlocal enabledelayedexpansion
set HEREDIR=%~dp0
set ENG=%HEREDIR%..\..
set TESTS=%ENG%\tests
set MODELARG=--model=%~1
set MUTDIR=D:\_t2961-mutants\bounds
set "SHADERDIR=%ENG%\out\shaders"
if not "%~2"=="" set "SHADERDIR=%~2"
call "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 -no_logo
cd /d "%HEREDIR%"
if not exist obj_t2916_s3 mkdir obj_t2916_s3
if not exist bin_t2916_s3 mkdir bin_t2916_s3
if not exist "%SHADERDIR%\*.cso" (echo SHADERS MISSING: "%SHADERDIR%" & exit /b 1)
if exist bin_t2916_s3\shaders rmdir /s /q bin_t2916_s3\shaders
mkdir bin_t2916_s3\shaders
xcopy /I /Q /Y "%SHADERDIR%\*.cso" bin_t2916_s3\shaders\ >nul
if errorlevel 1 (echo SHADER COPY FAILED: "%SHADERDIR%" & exit /b 1)
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

echo ================= Generating two MUT_SLM5 variants of gpu_1p0.cpp =================
if not defined SSLM_PYTHON set SSLM_PYTHON=C:\Users\dansu\AppData\Local\Programs\Python\Python313\python.exe
for %%c in (WALKSTATE UNBOUNDWALK) do (
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

rem check 1's own tamper input (schema_index=999999, below in ASBUILT) is refused by check 2's
rem own `!resolved_entry` null-guard now that the redundant early check is removed from the tree
rem (T-2917, D-SLM7600) -- SchemaMasksTable::ByIndex(size_t) (include/superslm/schema_masks.h) is
rem itself bounds-safe (`index < entries_.size() ? &entries_[index] : nullptr`), so an
rem out-of-range index can never make it return non-null. No MUT_INDEXCOUNT variant exists to run.
for %%v in (ASBUILT MUT_WALKSTATE MUT_UNBOUNDWALK) do (
    if "%%v"=="ASBUILT" (set GPUOBJS=obj_t2916_s3\gpu_asbuilt\gpu_1p0.obj obj_t2916_s3\gpu_asbuilt\superslm_gpu.obj& set EXPECT=PASS)
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
        )
    )
)

if "%OVERALL_OK%"=="1" (
    echo ===== build_t2916_s3_bounds_tamper: ASBUILT is green [including check 1's own tamper, now refused by check 2's null-guard]; MUT_WALKSTATE/MUT_UNBOUNDWALK killed =====
    exit /b 0
) else (
    echo ===== build_t2916_s3_bounds_tamper: FAILURES ABOVE =====
    exit /b 1
)
