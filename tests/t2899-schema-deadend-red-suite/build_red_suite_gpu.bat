@echo off
rem T-2900 (Curie) -- builds and runs every GPU-side Sec3.10 cell in this suite, against
rem the live source and targeted mutants: AS_BUILT (the tree as it stands,
rem compiled fresh from %ENG%\src -- no dependency on any external scratch checkout), FIXED
rem (T-2916, TE-370 M1: an alias for AS_BUILT -- the planner's own cumulative reference fix is
rem already landed in the live tree, confirmed by execution against the scratch copies this
rem configuration used to compile separately), MUT_CHECKEDRETURN and MUT_NOREARM (single-line
rem reverts of the live tip, generated fresh by `make_mut_gpu_checkedreturn_seam.py`/
rem `make_mut_gpu_norearm_seam.py`, plan Sec3.10.3 row 11's own guard-vitality mutants -- T-2916
rem replaced the out-of-repo `D:\_t2900\refs\gpu_1p0_mut_*.cpp` static copies these generators
rem superseded). Mirrors this suite's own `build_red_suite.bat`/`run_mutants_cpu_
rem deadend.bat` conventions: skip-fails-the-run, a bare `checks=/failures=/skips=` summary line
rem per binary, one build log per failed step.
rem
rem Usage: build_red_suite_gpu.bat <path-to-C39.sslm> [<path-to-1.5B-G5.sslm>] [<shaders-dir>]
rem   arg 1 -- the C39 synthetic (t2199_s8_fixture.sslm) -- required for every cell but the
rem           real-schema generalization.
rem   arg 2 -- the real, production-scale G5 fixture (t2132_g5_fixture_1p5b.sslm) -- only
rem           cell_gpu_cell1_realschema needs it; every other cell ignores it. Omit to SKIP that
rem           cell alone (a plain positional argument, not a --flag, since cmd.exe's own
rem           tokenizer splits a `--flag=value` argument at the `=` before this script ever sees
rem           it -- confirmed by execution, not assumed).
setlocal enabledelayedexpansion
set HEREDIR=%~dp0
set ENG=%HEREDIR%..\..
set TESTS=%ENG%\tests
set MODELARG=
set G5ARG=
if not "%~1"=="" set MODELARG=--model=%~1
if not "%~2"=="" set G5ARG=--g5fixture=%~2
set "SHADERDIR=%ENG%\out\shaders"
if not "%~3"=="" set "SHADERDIR=%~3"
call "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 -no_logo
cd /d "%HEREDIR%"
if not exist obj_gpu mkdir obj_gpu
if not exist bin_gpu mkdir bin_gpu
if not exist "%SHADERDIR%\*.cso" (echo SHADERS MISSING: "%SHADERDIR%" & exit /b 1)
if exist bin_gpu\shaders rmdir /s /q bin_gpu\shaders
mkdir bin_gpu\shaders
xcopy /I /Q /Y "%SHADERDIR%\*.cso" bin_gpu\shaders\ >nul
if errorlevel 1 (echo SHADER COPY FAILED: "%SHADERDIR%" & exit /b 1)

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

rem T-2909 (TE-365 C2): the string-schema fixture cell_gpu_cell2_degenerate needs (a real
rem schema with a string field; the C39 synthetic's own schema has no interior state -- see
rem that generator's own header comment). Regenerated fresh every run, never committed as a
rem binary blob, matching this suite's own S8-fixture discipline.
if not defined SSLM_PYTHON set SSLM_PYTHON=C:\Users\dansu\AppData\Local\Programs\Python\Python313\python.exe
"%SSLM_PYTHON%" make_t2909_string_schema_fixture.py "obj_gpu\t2909_string_schema_fixture.sslm" ^
    > obj_gpu\make_t2909_string_schema_fixture.log 2>&1
if errorlevel 1 (
    echo BUILD FAILED: make_t2909_string_schema_fixture.py
    type obj_gpu\make_t2909_string_schema_fixture.log
    set OVERALL_OK=0
)
set STRINGSCHEMAARG=--stringschema=obj_gpu\t2909_string_schema_fixture.sslm

echo ================= Compiling shared GPU object sets =================
rem AS_BUILT GPU: the two translation units, unmodified, exactly as the live tree carries them --
rem plus SUPERSLM_GPU_G5_FINISH_ROW_FAULT_INJECTION (T-2905: the production seam now lives here,
rem gated by this macro exactly like every other test-only injection seam in this file; inert to
rem every cell but cell_gpu_cell2_degenerate, since nothing else in this suite ever arms it).
if not exist obj_gpu\gpu_asbuilt mkdir obj_gpu\gpu_asbuilt
cl /nologo /std:c++20 /O2 /W3 /fp:precise /EHsc /DSUPERSLM_GPU_G5_FINISH_ROW_FAULT_INJECTION /I%ENG%\include /I%ENG%\src\gpu ^
    /c %ENG%\src\gpu\gpu_1p0.cpp %ENG%\src\gpu\superslm_gpu.cpp /Fo"obj_gpu\gpu_asbuilt\\" ^
    > obj_gpu\gpu_asbuilt.buildlog 2>&1 || (echo BUILD FAILED: gpu_asbuilt & type obj_gpu\gpu_asbuilt.buildlog & set OVERALL_OK=0)

rem FIXED GPU (T-2916, TE-370 M1): T-2895's own v5 design (checked return + SLM5 blob format) is
rem already landed in the live tree -- this configuration used to compile a separate scratch copy
rem (`D:\_t2900\refs\gpu_1p0_v5.cpp`/`superslm_gpu_v5.cpp`, plus an `include_override\gpu_port.h`
rem for the struct/signature changes those now-landed folds needed). Confirmed by execution
rem (Poirot's own TE-370 M1 evidence, re-verified here): the scratch copies differ from the
rem shipped `gpu_1p0.cpp`/`superslm_gpu.cpp` only by test-injection seams, and the override
rem header is a strict subset of the tracked `include/superslm/gpu_port.h` (one comment block
rem short). FIXED is therefore an alias for ASBUILT (`obj_gpu\gpu_asbuilt\*.obj`, built above) --
rem no separate compile, no scratch directory read.

rem GPU_FIXED_NOSLM5 (T-2903 fix, corrected T-2905): the checked-return + ready_for_logits re-arm
rem ALONE, no SLM5 blob format, paired with a genuinely PRISTINE, pre-T-2895 superslm_gpu.cpp +
rem gpu_port.h. T-2903's own comment said this pairing came from "the engine's own pristine
rem superslm_gpu.cpp, reused unchanged from the gpu_asbuilt step below" -- true only until T-2905
rem landed T-2895's own SLM5 fix into the live tree, which is what gpu_asbuilt now compiles from
rem (SUPERSLM_GPU_G5_FINISH_ROW_FAULT_INJECTION above is a second, independent reason gpu_asbuilt
rem is no longer "unmodified"). Reusing it here would both fail to compile (the v5 tail
rem parameters gpu_port.h now declares) and, if it somehow linked, would write 'SLM5' regardless
rem of the source file's own intent -- defeating the one property this configuration exists for.
rem The pristine base is `git show`'d fresh into a SCRATCH root outside this repo
rem (`D:\_t2961-mutants\pristine_pre_slm5\`, a build-time OUTPUT cache, not a source dependency -- every
rem byte in it is regenerated from tracked git history on every run, never read back as an input
rem to anything else), from commit `ccf87c1` on this same branch (the tip immediately before
rem T-2905's GPU fold; confirmed zero 'SLM5'/kGpuSeqBlobMagicV5 occurrences) -- NOT checked into
rem the tracked tree: a first attempt did check it in, and tests\ci\test_geometry_site_census.py's
rem own tree-wide GS-marker sweep (tools\geometry_site_census.py, unnarrowed by design outside
rem `_SKIP_DIR_NAMES`) then double-counted every GS marker superslm_gpu.cpp carries, since a
rem byte-identical copy of a production file inside the tracked test tree is exactly what that
rem census exists to catch. Regenerated every run so it always tracks the pinned commit, never a
rem stale local copy.
rem T-2916 (TE-370 M1): the checked-return+rearm fix itself, previously `D:\_te338\
rem gpu_1p0_fixed.cpp` (a hand-patched, out-of-repo, unversioned single file), is now applied to
rem the SAME `ccf87c1` pristine base by `make_gpu_fixed_noslm5.py`, anchored on the unique
rem unconditional-Transition block that commit still carries -- confirmed by execution to produce
rem code byte-identical to the retired external file (comments only differ).
set "PRISTINE=D:\_t2961-mutants\pristine_pre_slm5"
if not exist "%PRISTINE%\superslm" mkdir "%PRISTINE%\superslm"
git -C "%ENG%" show ccf87c1:src/gpu/superslm_gpu.cpp > "%PRISTINE%\superslm_gpu_pristine.cpp"
git -C "%ENG%" show ccf87c1:src/gpu/d3d12_harness.h > "%PRISTINE%\d3d12_harness.h"
git -C "%ENG%" show ccf87c1:include/superslm/gpu_port.h > "%PRISTINE%\superslm\gpu_port.h"
git -C "%ENG%" show ccf87c1:src/gpu/gpu_1p0.cpp > "%PRISTINE%\gpu_1p0_pristine.cpp"
"%SSLM_PYTHON%" make_gpu_fixed_noslm5.py "%PRISTINE%\gpu_1p0_pristine.cpp" "%PRISTINE%\gpu_1p0_fixed.cpp" ^
    > obj_gpu\make_gpu_fixed_noslm5.log 2>&1 || (echo GENERATE FAILED: gpu_fixed_noslm5 & type obj_gpu\make_gpu_fixed_noslm5.log & set OVERALL_OK=0)
if not exist obj_gpu\gpu_fixed_noslm5 mkdir obj_gpu\gpu_fixed_noslm5
cl /nologo /std:c++20 /O2 /W3 /fp:precise /EHsc /I"%PRISTINE%" /I%ENG%\include /I%ENG%\src\gpu ^
    /c "%PRISTINE%\gpu_1p0_fixed.cpp" "%PRISTINE%\superslm_gpu_pristine.cpp" /Fo"obj_gpu\gpu_fixed_noslm5\\" ^
    > obj_gpu\gpu_fixed_noslm5.buildlog 2>&1 || (echo BUILD FAILED: gpu_fixed_noslm5 & type obj_gpu\gpu_fixed_noslm5.buildlog & set OVERALL_OK=0)

rem MUT_CHECKEDRETURN / MUT_NOREARM (T-2916, TE-370 M1): single-line reverts, generated fresh
rem from the LIVE TIP's own gpu_1p0.cpp by `make_mut_gpu_checkedreturn_seam.py`/
rem `make_mut_gpu_norearm_seam.py` (structural anchor match, refuses on drift) -- replacing the
rem out-of-repo `D:\_t2900\refs\gpu_1p0_mut_*.cpp` static copies. Pair with `gpu_asbuilt`'s own
rem `superslm_gpu.obj` unchanged (the mutants touch gpu_1p0.cpp only).
if not defined SSLM_PYTHON set SSLM_PYTHON=C:\Users\dansu\AppData\Local\Programs\Python\Python313\python.exe
if not exist obj_gpu\gpu_mut_cr mkdir obj_gpu\gpu_mut_cr
"%SSLM_PYTHON%" make_mut_gpu_checkedreturn_seam.py "%ENG%\src\gpu\gpu_1p0.cpp" "obj_gpu\gpu_mut_cr\gpu_1p0_mut_checkedreturn.cpp" ^
    > obj_gpu\make_mut_checkedreturn.log 2>&1 || (echo GENERATE FAILED: gpu_mut_cr & type obj_gpu\make_mut_checkedreturn.log & set OVERALL_OK=0)
cl /nologo /std:c++20 /O2 /W3 /fp:precise /EHsc /DSUPERSLM_GPU_G5_FINISH_ROW_FAULT_INJECTION /I%ENG%\include /I%ENG%\src\gpu ^
    /c obj_gpu\gpu_mut_cr\gpu_1p0_mut_checkedreturn.cpp /Fo"obj_gpu\gpu_mut_cr\\" ^
    > obj_gpu\gpu_mut_cr.buildlog 2>&1 || (echo BUILD FAILED: gpu_mut_cr & type obj_gpu\gpu_mut_cr.buildlog & set OVERALL_OK=0)
if not exist obj_gpu\gpu_mut_nr mkdir obj_gpu\gpu_mut_nr
"%SSLM_PYTHON%" make_mut_gpu_norearm_seam.py "%ENG%\src\gpu\gpu_1p0.cpp" "obj_gpu\gpu_mut_nr\gpu_1p0_mut_norearm.cpp" ^
    > obj_gpu\make_mut_norearm.log 2>&1 || (echo GENERATE FAILED: gpu_mut_nr & type obj_gpu\make_mut_norearm.log & set OVERALL_OK=0)
cl /nologo /std:c++20 /O2 /W3 /fp:precise /EHsc /DSUPERSLM_GPU_G5_FINISH_ROW_FAULT_INJECTION /I%ENG%\include /I%ENG%\src\gpu ^
    /c obj_gpu\gpu_mut_nr\gpu_1p0_mut_norearm.cpp /Fo"obj_gpu\gpu_mut_nr\\" ^
    > obj_gpu\gpu_mut_nr.buildlog 2>&1 || (echo BUILD FAILED: gpu_mut_nr & type obj_gpu\gpu_mut_nr.buildlog & set OVERALL_OK=0)
for %%v in (LATE RESTORE) do (
    if not exist "obj_gpu\gpu_mut_%%v" mkdir "obj_gpu\gpu_mut_%%v"
    "%SSLM_PYTHON%" make_mut_gpu_oldbind.py %%v "%ENG%\src\gpu\gpu_1p0.cpp" "obj_gpu\gpu_mut_%%v\gpu_1p0_mut_%%v.cpp" > "obj_gpu\make_mut_%%v.log" 2>&1
    if errorlevel 1 (echo GENERATE FAILED: gpu_mut_%%v & type "obj_gpu\make_mut_%%v.log" & set OVERALL_OK=0)
    cl /nologo /std:c++20 /O2 /W3 /fp:precise /EHsc /DSUPERSLM_GPU_G5_FINISH_ROW_FAULT_INJECTION /I%ENG%\include /I%ENG%\src\gpu ^
        /c "obj_gpu\gpu_mut_%%v\gpu_1p0_mut_%%v.cpp" /Fo"obj_gpu\gpu_mut_%%v\\" > "obj_gpu\gpu_mut_%%v.buildlog" 2>&1
    if errorlevel 1 (echo BUILD FAILED: gpu_mut_%%v & type "obj_gpu\gpu_mut_%%v.buildlog" & set OVERALL_OK=0)
)

rem CPU_COMMON: backend-agnostic sources shared by every configuration (unaffected by either fix).
if not exist obj_gpu\cpu_common mkdir obj_gpu\cpu_common
cl /nologo /std:c++20 /O2 /W3 /fp:precise /EHsc /I%ENG%\include /I%ENG%\src ^
    /c %SRC_NOABI% /Fo"obj_gpu\cpu_common\\" ^
    > obj_gpu\cpu_common.buildlog 2>&1 || (echo BUILD FAILED: cpu_common & type obj_gpu\cpu_common.buildlog & set OVERALL_OK=0)
for %%f in (%SRC_NOABI%) do (
    for %%n in (%%f) do set CPU_COMMON_OBJS=!CPU_COMMON_OBJS! obj_gpu\cpu_common\%%~nn.obj
)

rem CPU sslm_abi variants (AS_BUILT and FIXED).
if not exist obj_gpu\cpu_asbuilt mkdir obj_gpu\cpu_asbuilt
cl /nologo /std:c++20 /O2 /W3 /fp:precise /EHsc /I%ENG%\include /I%ENG%\src /c %ENG%\src\sslm_abi.cpp ^
    /Fo"obj_gpu\cpu_asbuilt\sslm_abi.obj" > obj_gpu\cpu_asbuilt.buildlog 2>&1 || (echo BUILD FAILED: cpu_asbuilt & type obj_gpu\cpu_asbuilt.buildlog & set OVERALL_OK=0)

echo ================= Compiling and linking cells =================

rem ---- cell_gpu_cell1_shortschema: GPU-only, four linkage variants (AS_BUILT/FIXED/MUT_CR/MUT_NR) ----
cl /nologo /std:c++20 /O2 /W3 /fp:precise /EHsc %STOCKINC% /c cell_gpu_cell1_shortschema.cpp ^
    /Fo"obj_gpu\cell1_short_stock.obj" > obj_gpu\cell1_short_stock.buildlog 2>&1 || (echo BUILD FAILED: cell1_short_stock.obj & type obj_gpu\cell1_short_stock.buildlog & set OVERALL_OK=0)
cl /nologo /std:c++20 /O2 /W3 /fp:precise /EHsc %STOCKINC% /c cell_gpu_cell1_shortschema.cpp ^
    /Fo"obj_gpu\cell1_short_ovr.obj" > obj_gpu\cell1_short_ovr.buildlog 2>&1 || (echo BUILD FAILED: cell1_short_ovr.obj & type obj_gpu\cell1_short_ovr.buildlog & set OVERALL_OK=0)

for %%v in (ASBUILT FIXED MUT_CHECKEDRETURN MUT_NOREARM) do (
    if "%%v"=="ASBUILT" (
        set CELLOBJ=obj_gpu\cell1_short_stock.obj
        set GPUOBJS=obj_gpu\gpu_asbuilt\gpu_1p0.obj obj_gpu\gpu_asbuilt\superslm_gpu.obj
        set EXPECT=PASS
    )
    if "%%v"=="FIXED" (
        set CELLOBJ=obj_gpu\cell1_short_ovr.obj
        set GPUOBJS=obj_gpu\gpu_asbuilt\gpu_1p0.obj obj_gpu\gpu_asbuilt\superslm_gpu.obj
        set EXPECT=PASS
    )
    if "%%v"=="MUT_CHECKEDRETURN" (
        set CELLOBJ=obj_gpu\cell1_short_ovr.obj
        set GPUOBJS=obj_gpu\gpu_mut_cr\gpu_1p0_mut_checkedreturn.obj obj_gpu\gpu_asbuilt\superslm_gpu.obj
        set EXPECT=KILL
    )
    if "%%v"=="MUT_NOREARM" (
        set CELLOBJ=obj_gpu\cell1_short_ovr.obj
        set GPUOBJS=obj_gpu\gpu_mut_nr\gpu_1p0_mut_norearm.obj obj_gpu\gpu_asbuilt\superslm_gpu.obj
        set EXPECT=KILL
    )
    echo ===== cell_gpu_cell1_shortschema [%%v] expect=!EXPECT! =====
    link /nologo /OUT:"bin_gpu\cell1_short_%%v.exe" !CELLOBJ! !GPUOBJS! !CPU_COMMON_OBJS! %SYSLIBS% ^
        > "obj_gpu\cell1_short_%%v.linklog" 2>&1
    if errorlevel 1 (
        echo    LINK FAILED: & type "obj_gpu\cell1_short_%%v.linklog" & set OVERALL_OK=0
    ) else (
        "bin_gpu\cell1_short_%%v.exe" %MODELARG% > "obj_gpu\cell1_short_%%v.runlog" 2>&1
        type "obj_gpu\cell1_short_%%v.runlog"
        set SUMMARY_LINE=
        for /f "delims=" %%s in ('findstr /R "^checks=[0-9]* failures=[0-9]*" "obj_gpu\cell1_short_%%v.runlog"') do set SUMMARY_LINE=%%s
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
            if not "!SKIPN!"=="0" (echo    SKIPS=!SKIPN! -- a skipped cell fails the run ^("obj_gpu\cell1_short_%%v.runlog"^)& set OVERALL_OK=0)
            if "!EXPECT!"=="PASS" (
                if not "!FAILN!"=="0" (echo    FAILURES=!FAILN! -- the green tip must be green ^("obj_gpu\cell1_short_%%v.runlog"^)& set OVERALL_OK=0)
            ) else (
                if "!FAILN!"=="0" (echo    SURVIVING MUTANT -- expected a kill, got failures=0 ^("obj_gpu\cell1_short_%%v.runlog"^)& set OVERALL_OK=0)
            )
        )
    )
)

rem ---- cell_gpu_cell1_realschema (+ its CPU-side helper): AS_BUILT and FIXED ----
if not exist obj_gpu\cell1_real_stock_ mkdir obj_gpu\cell1_real_stock_
if not exist obj_gpu\cell1_real_ovr_ mkdir obj_gpu\cell1_real_ovr_
cl /nologo /std:c++20 /O2 /W3 /fp:precise /EHsc %STOCKINC% /c cell_gpu_cell1_realschema.cpp cell_gpu_cell1_realschema_cpu_side.cpp ^
    /Fo"obj_gpu\cell1_real_stock_\\" > obj_gpu\cell1_real_stock.buildlog 2>&1 || (echo BUILD FAILED: cell1_real stock & type obj_gpu\cell1_real_stock.buildlog & set OVERALL_OK=0)
cl /nologo /std:c++20 /O2 /W3 /fp:precise /EHsc %STOCKINC% /c cell_gpu_cell1_realschema.cpp cell_gpu_cell1_realschema_cpu_side.cpp ^
    /Fo"obj_gpu\cell1_real_ovr_\\" > obj_gpu\cell1_real_ovr.buildlog 2>&1 || (echo BUILD FAILED: cell1_real ovr & type obj_gpu\cell1_real_ovr.buildlog & set OVERALL_OK=0)

for %%v in (ASBUILT FIXED) do (
    if "%%v"=="ASBUILT" (
        set CELLMAIN=obj_gpu\cell1_real_stock_\cell_gpu_cell1_realschema.obj
        set CELLCPU=obj_gpu\cell1_real_stock_\cell_gpu_cell1_realschema_cpu_side.obj
        set GPUOBJS=obj_gpu\gpu_asbuilt\gpu_1p0.obj obj_gpu\gpu_asbuilt\superslm_gpu.obj
        set ABIOBJ=obj_gpu\cpu_asbuilt\sslm_abi.obj
    )
    if "%%v"=="FIXED" (
        set CELLMAIN=obj_gpu\cell1_real_ovr_\cell_gpu_cell1_realschema.obj
        set CELLCPU=obj_gpu\cell1_real_ovr_\cell_gpu_cell1_realschema_cpu_side.obj
        set GPUOBJS=obj_gpu\gpu_asbuilt\gpu_1p0.obj obj_gpu\gpu_asbuilt\superslm_gpu.obj
        set ABIOBJ=obj_gpu\cpu_asbuilt\sslm_abi.obj
    )
    echo ===== cell_gpu_cell1_realschema [%%v] =====
    link /nologo /OUT:"bin_gpu\cell1_real_%%v.exe" !CELLMAIN! !CELLCPU! !GPUOBJS! !CPU_COMMON_OBJS! !ABIOBJ! %SYSLIBS% ^
        > "obj_gpu\cell1_real_%%v.linklog" 2>&1
    if errorlevel 1 (
        echo    LINK FAILED: & type "obj_gpu\cell1_real_%%v.linklog" & set OVERALL_OK=0
    ) else (
        "bin_gpu\cell1_real_%%v.exe" %G5ARG% > "obj_gpu\cell1_real_%%v.runlog" 2>&1
        type "obj_gpu\cell1_real_%%v.runlog"
        set SUMMARY_LINE=
        for /f "delims=" %%s in ('findstr /R "^checks=[0-9]* failures=[0-9]*" "obj_gpu\cell1_real_%%v.runlog"') do set SUMMARY_LINE=%%s
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
            if not "!FAILN!"=="0" (echo    FAILURES=!FAILN! -- a red cell ^("obj_gpu\cell1_real_%%v.runlog"^)& set OVERALL_OK=0)
            if not "!SKIPN!"=="0" (echo    SKIPS=!SKIPN! -- a skipped cell fails the run ^("obj_gpu\cell1_real_%%v.runlog"^)& set OVERALL_OK=0)
        )
    )
)

rem ---- cell_gpu_cell2_degenerate: written for real, T-2909 (TE-365 C2) ----
rem T-2905 built the production seam (ArmGpuFinishDegenerateLogitRowInjection, gpu_1p0.cpp,
rem compiled under SUPERSLM_GPU_G5_FINISH_ROW_FAULT_INJECTION); T-2909 writes the cell that
rem actually fires it -- against a real schema with a string field (STRINGSCHEMAARG, generated
rem above), driving to S_e via a real backslash transition. Two link configurations, matching
rem plan Sec3.10.3 row 11's own requirement ("a single-point mutant that reverts the checked
rem return... must turn both Cell 1 and Cell 2 red"): ASBUILT (the live tree, checked return in
rem place -- must be GREEN) and MUT_CHECKEDRETURN (RED). The EXISTING gpu_mut_cr object
rem (D:\_t2900\refs\gpu_1p0_mut_checkedreturn.cpp) predates T-2905's own seam entirely --
rem `ArmGpuFinishDegenerateLogitRowInjection` is undefined there. The current generator
rem creates the checked-return mutant from the live tip, compiled with the fault seam; Cell 2
rem also runs MUT_LATE, which restores admission of a bind after prompt prefill (D-SLM7625).
if not exist obj_gpu\cell2_stock mkdir obj_gpu\cell2_stock
cl /nologo /std:c++20 /O2 /W3 /fp:precise /EHsc %STOCKINC% /DSUPERSLM_GPU_G5_FINISH_ROW_FAULT_INJECTION /c cell_gpu_cell2_degenerate.cpp ^
    /Fo"obj_gpu\cell2_stock\\" > obj_gpu\cell2_stock.buildlog 2>&1 || (echo BUILD FAILED: cell2_stock & type obj_gpu\cell2_stock.buildlog & set OVERALL_OK=0)

for %%v in (ASBUILT MUT_CHECKEDRETURN MUT_LATE) do (
    set CELLOBJ=obj_gpu\cell2_stock\cell_gpu_cell2_degenerate.obj
    if "%%v"=="ASBUILT" (
        set GPUOBJS=obj_gpu\gpu_asbuilt\gpu_1p0.obj obj_gpu\gpu_asbuilt\superslm_gpu.obj
    )
    if "%%v"=="MUT_CHECKEDRETURN" (
        set GPUOBJS=obj_gpu\gpu_mut_cr\gpu_1p0_mut_checkedreturn.obj obj_gpu\gpu_asbuilt\superslm_gpu.obj
    )
    if "%%v"=="MUT_LATE" (
        set GPUOBJS=obj_gpu\gpu_mut_LATE\gpu_1p0_mut_LATE.obj obj_gpu\gpu_asbuilt\superslm_gpu.obj
    )
    echo ===== cell_gpu_cell2_degenerate [%%v] =====
    link /nologo /OUT:"bin_gpu\cell2_%%v.exe" !CELLOBJ! !GPUOBJS! !CPU_COMMON_OBJS! %SYSLIBS% ^
        > "obj_gpu\cell2_%%v.linklog" 2>&1
    if errorlevel 1 (
        echo    LINK FAILED: & type "obj_gpu\cell2_%%v.linklog" & set OVERALL_OK=0
    ) else (
        "bin_gpu\cell2_%%v.exe" %STRINGSCHEMAARG% > "obj_gpu\cell2_%%v.runlog" 2>&1
        type "obj_gpu\cell2_%%v.runlog"
        set SUMMARY_LINE=
        for /f "delims=" %%s in ('findstr /R "^checks=[0-9]* failures=[0-9]*" "obj_gpu\cell2_%%v.runlog"') do set SUMMARY_LINE=%%s
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
            if not "!SKIPN!"=="0" (echo    SKIPS=!SKIPN! -- a skipped cell fails the run ^(obj_gpu\cell2_%%v.runlog^)& set OVERALL_OK=0)
            if "%%v"=="ASBUILT" (
                if not "!FAILN!"=="0" (echo    FAILURES=!FAILN! -- the green tip must be green ^(obj_gpu\cell2_%%v.runlog^)& set OVERALL_OK=0)
            ) else (
                if "!FAILN!"=="0" (echo    SURVIVING MUTANT -- expected a kill, got failures=0 ^(obj_gpu\cell2_%%v.runlog^)& set OVERALL_OK=0)
            )
        )
    )
)

rem ---- cell_gpu_cell3_agreement (+ CPU-side helper): AS_BUILT, FIXED, and FIXED+--mutant ----
if not exist obj_gpu\cell3_stock_ mkdir obj_gpu\cell3_stock_
if not exist obj_gpu\cell3_ovr_ mkdir obj_gpu\cell3_ovr_
cl /nologo /std:c++20 /O2 /W3 /fp:precise /EHsc %STOCKINC% /c cell_gpu_cell3_agreement.cpp cell_gpu_cell3_cpu_side.cpp ^
    /Fo"obj_gpu\cell3_stock_\\" > obj_gpu\cell3_stock.buildlog 2>&1 || (echo BUILD FAILED: cell3 stock & type obj_gpu\cell3_stock.buildlog & set OVERALL_OK=0)
cl /nologo /std:c++20 /O2 /W3 /fp:precise /EHsc %STOCKINC% /c cell_gpu_cell3_agreement.cpp cell_gpu_cell3_cpu_side.cpp ^
    /Fo"obj_gpu\cell3_ovr_\\" > obj_gpu\cell3_ovr.buildlog 2>&1 || (echo BUILD FAILED: cell3 ovr & type obj_gpu\cell3_ovr.buildlog & set OVERALL_OK=0)

for %%v in (ASBUILT FIXED) do (
    if "%%v"=="ASBUILT" (
        set CELLMAIN=obj_gpu\cell3_stock_\cell_gpu_cell3_agreement.obj
        set CELLCPU=obj_gpu\cell3_stock_\cell_gpu_cell3_cpu_side.obj
        set GPUOBJS=obj_gpu\gpu_asbuilt\gpu_1p0.obj obj_gpu\gpu_asbuilt\superslm_gpu.obj
        set ABIOBJ=obj_gpu\cpu_asbuilt\sslm_abi.obj
    )
    if "%%v"=="FIXED" (
        set CELLMAIN=obj_gpu\cell3_ovr_\cell_gpu_cell3_agreement.obj
        set CELLCPU=obj_gpu\cell3_ovr_\cell_gpu_cell3_cpu_side.obj
        set GPUOBJS=obj_gpu\gpu_asbuilt\gpu_1p0.obj obj_gpu\gpu_asbuilt\superslm_gpu.obj
        set ABIOBJ=obj_gpu\cpu_asbuilt\sslm_abi.obj
    )
    echo ===== cell_gpu_cell3_agreement [%%v] =====
    link /nologo /OUT:"bin_gpu\cell3_%%v.exe" !CELLMAIN! !CELLCPU! !GPUOBJS! !CPU_COMMON_OBJS! !ABIOBJ! %SYSLIBS% ^
        > "obj_gpu\cell3_%%v.linklog" 2>&1
    if errorlevel 1 (
        echo    LINK FAILED: & type "obj_gpu\cell3_%%v.linklog" & set OVERALL_OK=0
    ) else (
        "bin_gpu\cell3_%%v.exe" %MODELARG% > "obj_gpu\cell3_%%v.runlog" 2>&1
        type "obj_gpu\cell3_%%v.runlog"
        set SUMMARY_LINE=
        for /f "delims=" %%s in ('findstr /R "^checks=[0-9]* failures=[0-9]*" "obj_gpu\cell3_%%v.runlog"') do set SUMMARY_LINE=%%s
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
            if not "!FAILN!"=="0" (echo    FAILURES=!FAILN! -- a red cell ^("obj_gpu\cell3_%%v.runlog"^)& set OVERALL_OK=0)
            if not "!SKIPN!"=="0" (echo    SKIPS=!SKIPN! -- a skipped cell fails the run ^("obj_gpu\cell3_%%v.runlog"^)& set OVERALL_OK=0)
        )
        if "%%v"=="FIXED" (
            "bin_gpu\cell3_%%v.exe" %MODELARG% --mutant > "obj_gpu\cell3_%%v_mutant.runlog" 2>&1
            type "obj_gpu\cell3_%%v_mutant.runlog"
            set SUMMARY_LINE=
            for /f "delims=" %%s in ('findstr /R "^checks=[0-9]* failures=[0-9]*" "obj_gpu\cell3_%%v_mutant.runlog"') do set SUMMARY_LINE=%%s
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
                rem T-2909: --mutant is TE-361's own divergent-input construction (plan Sec3.10.2
                rem Cell 3's own guard-vitality mutant) -- it MUST be killed (c) failing on every
                rem route while (a)/(b) stay green under the same run), so failures=0 here is a
                rem surviving mutant, not a green run.
                if not "!SKIPN!"=="0" (echo    SKIPS=!SKIPN! -- a skipped cell fails the run ^("obj_gpu\cell3_%%v_mutant.runlog"^)& set OVERALL_OK=0)
                if "!FAILN!"=="0" (echo    SURVIVING MUTANT -- expected the divergent-input mutant to be killed, got failures=0 ^("obj_gpu\cell3_%%v_mutant.runlog"^)& set OVERALL_OK=0)
            )
        )
    )
)

rem ---- cell_gpu_slm5_saverestore: GPU-only, AS_BUILT and FIXED ----
cl /nologo /std:c++20 /O2 /W3 /fp:precise /EHsc %STOCKINC% /c cell_gpu_slm5_saverestore.cpp /Fo"obj_gpu\slm5_stock.obj" ^
    > obj_gpu\slm5_stock.buildlog 2>&1 || (echo BUILD FAILED: slm5_stock.obj & type obj_gpu\slm5_stock.buildlog & set OVERALL_OK=0)
cl /nologo /std:c++20 /O2 /W3 /fp:precise /EHsc %STOCKINC% /c cell_gpu_slm5_saverestore.cpp /Fo"obj_gpu\slm5_ovr.obj" ^
    > obj_gpu\slm5_ovr.buildlog 2>&1 || (echo BUILD FAILED: slm5_ovr.obj & type obj_gpu\slm5_ovr.buildlog & set OVERALL_OK=0)
for %%v in (ASBUILT FIXED MUT_RESTORE) do (
    if "%%v"=="ASBUILT" (
        set CELLOBJ=obj_gpu\slm5_stock.obj
        set GPUOBJS=obj_gpu\gpu_asbuilt\gpu_1p0.obj obj_gpu\gpu_asbuilt\superslm_gpu.obj
    )
    if "%%v"=="FIXED" (
        set CELLOBJ=obj_gpu\slm5_ovr.obj
        set GPUOBJS=obj_gpu\gpu_asbuilt\gpu_1p0.obj obj_gpu\gpu_asbuilt\superslm_gpu.obj
    )
    if "%%v"=="MUT_RESTORE" (
        set CELLOBJ=obj_gpu\slm5_stock.obj
        set GPUOBJS=obj_gpu\gpu_mut_RESTORE\gpu_1p0_mut_RESTORE.obj obj_gpu\gpu_asbuilt\superslm_gpu.obj
    )
    echo ===== cell_gpu_slm5_saverestore [%%v] =====
    link /nologo /OUT:"bin_gpu\slm5_%%v.exe" !CELLOBJ! !GPUOBJS! !CPU_COMMON_OBJS! %SYSLIBS% ^
        > "obj_gpu\slm5_%%v.linklog" 2>&1
    if errorlevel 1 (
        echo    LINK FAILED: & type "obj_gpu\slm5_%%v.linklog" & set OVERALL_OK=0
    ) else (
        "bin_gpu\slm5_%%v.exe" %MODELARG% > "obj_gpu\slm5_%%v.runlog" 2>&1
        type "obj_gpu\slm5_%%v.runlog"
        set SUMMARY_LINE=
        for /f "delims=" %%s in ('findstr /R "^checks=[0-9]* failures=[0-9]*" "obj_gpu\slm5_%%v.runlog"') do set SUMMARY_LINE=%%s
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
            if "%%v"=="MUT_RESTORE" (
                if "!FAILN!"=="0" (echo    SURVIVING MUTANT -- restored-bind regression was not detected & set OVERALL_OK=0)
            ) else (
                if not "!FAILN!"=="0" (echo    FAILURES=!FAILN! -- a red cell ^("obj_gpu\slm5_%%v.runlog"^)& set OVERALL_OK=0)
            )
            if not "!SKIPN!"=="0" (echo    SKIPS=!SKIPN! -- a skipped cell fails the run ^("obj_gpu\slm5_%%v.runlog"^)& set OVERALL_OK=0)
        )
    )
)

rem ---- cell_gpu_slm4_dump (checked-return-FIXED/no-SLM5 -- must genuinely dead-end AND write a
rem real legacy 'SLM4' blob; T-2903 fix, see gpu_fixed_noslm5 above -- gpu_asbuilt alone cannot
rem dead-end at all, so linking against it made the pre-save assertion below unwinnable) /
rem cell_gpu_slm4_restore (FIXED only) ----
cl /nologo /std:c++20 /O2 /W3 /fp:precise /EHsc %STOCKINC% /c cell_gpu_slm4_dump.cpp /Fo"obj_gpu\slm4_dump.obj" ^
    > obj_gpu\slm4_dump.buildlog 2>&1 || (echo BUILD FAILED: slm4_dump.obj & type obj_gpu\slm4_dump.buildlog & set OVERALL_OK=0)
link /nologo /OUT:"bin_gpu\slm4_dump.exe" obj_gpu\slm4_dump.obj obj_gpu\gpu_fixed_noslm5\gpu_1p0_fixed.obj obj_gpu\gpu_fixed_noslm5\superslm_gpu_pristine.obj !CPU_COMMON_OBJS! %SYSLIBS% ^
    > obj_gpu\slm4_dump.linklog 2>&1 || (echo LINK FAILED: slm4_dump & type obj_gpu\slm4_dump.linklog & set OVERALL_OK=0)
echo ===== cell_gpu_slm4_dump [FIXED, pre-SLM5 -- checked-return fix only, T-2903] =====
"bin_gpu\slm4_dump.exe" %MODELARG% --out=obj_gpu\slm4_blob.bin > obj_gpu\slm4_dump.runlog 2>&1
type obj_gpu\slm4_dump.runlog
set SUMMARY_LINE=
for /f "delims=" %%s in ('findstr /R "^checks=[0-9]* failures=[0-9]*" obj_gpu\slm4_dump.runlog') do set SUMMARY_LINE=%%s
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
    if not "!FAILN!"=="0" (echo    FAILURES=!FAILN! -- a red cell ^(obj_gpu\slm4_dump.runlog^)& set OVERALL_OK=0)
    if not "!SKIPN!"=="0" (echo    SKIPS=!SKIPN! -- a skipped cell fails the run ^(obj_gpu\slm4_dump.runlog^)& set OVERALL_OK=0)
)

cl /nologo /std:c++20 /O2 /W3 /fp:precise /EHsc %STOCKINC% /c cell_gpu_slm4_restore.cpp /Fo"obj_gpu\slm4_restore.obj" ^
    > obj_gpu\slm4_restore.buildlog 2>&1 || (echo BUILD FAILED: slm4_restore.obj & type obj_gpu\slm4_restore.buildlog & set OVERALL_OK=0)
link /nologo /OUT:"bin_gpu\slm4_restore.exe" obj_gpu\slm4_restore.obj obj_gpu\gpu_asbuilt\gpu_1p0.obj obj_gpu\gpu_asbuilt\superslm_gpu.obj !CPU_COMMON_OBJS! %SYSLIBS% ^
    > obj_gpu\slm4_restore.linklog 2>&1 || (echo LINK FAILED: slm4_restore & type obj_gpu\slm4_restore.linklog & set OVERALL_OK=0)
echo ===== cell_gpu_slm4_restore [FIXED, reading the ASBUILT-written blob] =====
"bin_gpu\slm4_restore.exe" %MODELARG% --in=obj_gpu\slm4_blob.bin > obj_gpu\slm4_restore.runlog 2>&1
type obj_gpu\slm4_restore.runlog
set SUMMARY_LINE=
for /f "delims=" %%s in ('findstr /R "^checks=[0-9]* failures=[0-9]*" obj_gpu\slm4_restore.runlog') do set SUMMARY_LINE=%%s
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
    if not "!FAILN!"=="0" (echo    FAILURES=!FAILN! -- a red cell ^(obj_gpu\slm4_restore.runlog^)& set OVERALL_OK=0)
    if not "!SKIPN!"=="0" (echo    SKIPS=!SKIPN! -- a skipped cell fails the run ^(obj_gpu\slm4_restore.runlog^)& set OVERALL_OK=0)
)

echo DEBUG_OVERALL_OK=[%OVERALL_OK%]
if "%OVERALL_OK%"=="1" (
    echo ===== build_red_suite_gpu: all cells built and ran =====
    exit /b 0
) else (
    echo ===== build_red_suite_gpu: FAILURES ABOVE =====
    exit /b 1
)
