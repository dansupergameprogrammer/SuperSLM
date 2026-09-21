@echo off
rem T-2900 (Curie) -- builds and runs every GPU-side Sec3.10 cell in this suite, against FOUR
rem source configurations: AS_BUILT (the tree as it stands, `curie/t2809-stage1-build` HEAD,
rem compiled fresh from %ENG%\src -- no dependency on any external scratch checkout), FIXED (the
rem planner's own cumulative reference fix, `Claude/Vitruvius/t2895-probe/gpu_1p0_v5.cpp`/
rem `superslm_gpu_v5.cpp` plus `Claude/Vitruvius/t2898-probe/sslm_abi_cpu_fixed_v5.cpp`, staged at
rem D:\_t2900\refs by this ticket), MUT_CHECKEDRETURN and MUT_NOREARM (T-2900's own single-line
rem reverts of the reference fix, `D:\_t2900\refs\gpu_1p0_mut_*.cpp`, plan Sec3.10.3 row 11's own
rem guard-vitality mutants). Mirrors this suite's own `build_red_suite.bat`/`run_mutants_cpu_
rem deadend.bat` conventions: skip-fails-the-run, a bare `checks=/failures=/skips=` summary line
rem per binary, one build log per failed step.
rem
rem Usage: build_red_suite_gpu.bat <path-to-C39.sslm> [<path-to-1.5B-G5.sslm>]
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
set REFS=D:\_t2900\refs
set MODELARG=
set G5ARG=
if not "%~1"=="" set MODELARG=--model=%~1
if not "%~2"=="" set G5ARG=--g5fixture=%~2
call "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 -no_logo
cd /d "%HEREDIR%"
if not exist obj_gpu mkdir obj_gpu
if not exist bin_gpu mkdir bin_gpu
if not exist bin_gpu\shaders xcopy /E /I /Q /Y "%ENG%\build\gpu-shaders-staged" bin_gpu\shaders >nul

set SRC_NOABI=%ENG%\src\artifact.cpp %ENG%\src\sha256.cpp %ENG%\src\tokenizer.cpp %ENG%\src\model.cpp ^
    %ENG%\src\intmath.cpp %ENG%\src\silu_lut.cpp %ENG%\src\matmul.cpp %ENG%\src\proof_manifest.cpp ^
    %ENG%\src\trace_hook.cpp %ENG%\src\forward\checked_chain_funnel.cpp ^
    %ENG%\src\forward\forward_sites.cpp %ENG%\src\decode_digest.cpp ^
    %ENG%\src\damped_greedy_antilm.cpp %ENG%\src\damped_greedy_topk.cpp ^
    %ENG%\src\damped_greedy_phaseD.cpp %ENG%\src\damped_greedy_phaseD_loop.cpp

set GPUINC=%TESTS%\t2791-gpu-prefill-read-red-suite
set STOCKINC=/I%ENG%\include /I%ENG%\src /I%TESTS% /I%GPUINC% /I.
set OVRINC=/I%REFS%\include_override /I%ENG%\include /I%ENG%\src /I%TESTS% /I%GPUINC% /I.
set SYSLIBS=d3d12.lib dxgi.lib dxguid.lib
set OVERALL_OK=1

echo ================= Compiling shared GPU object sets =================
rem AS_BUILT GPU: the two translation units, unmodified, exactly as the live tree carries them --
rem plus SUPERSLM_GPU_G5_FINISH_ROW_FAULT_INJECTION (T-2905: the production seam now lives here,
rem gated by this macro exactly like every other test-only injection seam in this file; inert to
rem every cell but cell_gpu_cell2_degenerate, since nothing else in this suite ever arms it).
if not exist obj_gpu\gpu_asbuilt mkdir obj_gpu\gpu_asbuilt
cl /nologo /std:c++20 /O2 /W3 /fp:precise /EHsc /DSUPERSLM_GPU_G5_FINISH_ROW_FAULT_INJECTION /I%ENG%\include /I%ENG%\src\gpu ^
    /c %ENG%\src\gpu\gpu_1p0.cpp %ENG%\src\gpu\superslm_gpu.cpp /Fo"obj_gpu\gpu_asbuilt\\" ^
    > obj_gpu\gpu_asbuilt.buildlog 2>&1 || (echo BUILD FAILED: gpu_asbuilt & type obj_gpu\gpu_asbuilt.buildlog & set OVERALL_OK=0)

rem FIXED GPU: T-2895's own v5 reference (checked return + SLM5 blob format), needs the patched
rem gpu_port.h override for the struct/signature changes.
if not exist obj_gpu\gpu_fixed mkdir obj_gpu\gpu_fixed
cl /nologo /std:c++20 /O2 /W3 /fp:precise /EHsc /I%REFS%\include_override /I%ENG%\include /I%ENG%\src\gpu ^
    /c %REFS%\gpu_1p0_v5.cpp %REFS%\superslm_gpu_v5.cpp /Fo"obj_gpu\gpu_fixed\\" ^
    > obj_gpu\gpu_fixed.buildlog 2>&1 || (echo BUILD FAILED: gpu_fixed & type obj_gpu\gpu_fixed.buildlog & set OVERALL_OK=0)

rem GPU_FIXED_NOSLM5 (T-2903 fix, corrected T-2905): the checked-return + ready_for_logits re-arm
rem ALONE, no SLM5 blob format -- T-2866/T-2864's own single-file patch
rem (`D:\_te338\gpu_1p0_fixed.cpp`, read-only, still present on the dev box per the plan's own
rem reproduction recipe), paired with a genuinely PRISTINE, pre-T-2895 superslm_gpu.cpp +
rem gpu_port.h. T-2903's own comment said this pairing came from "the engine's own pristine
rem superslm_gpu.cpp, reused unchanged from the gpu_asbuilt step below" -- true only until T-2905
rem landed T-2895's own SLM5 fix into the live tree, which is what gpu_asbuilt now compiles from
rem (SUPERSLM_GPU_G5_FINISH_ROW_FAULT_INJECTION above is a second, independent reason gpu_asbuilt
rem is no longer "unmodified"). Reusing it here would both fail to compile (gpu_1p0_fixed.cpp's
rem own Save/RestoreGpuSequenceState calls predate the v5 tail parameters gpu_port.h now
rem declares) and, if it somehow linked, would write 'SLM5' regardless of the source file's own
rem intent -- defeating the one property this configuration exists for. `pristine_pre_slm5\`
rem (checked in alongside this script) is `git show`'d from this same branch's own commit
rem immediately before T-2905's GPU fold (confirmed zero 'SLM5'/kGpuSeqBlobMagicV5 occurrences),
rem so this configuration keeps meaning what its own name says regardless of what the live tree
rem goes on to carry. Needed because a genuine legacy 'SLM4' blob (cell_gpu_slm4_dump) requires
rem the GPU to actually dead-end (gpu_asbuilt PRE-T-2905 could not: the unfixed finish bridge
rem never returned -2) while still saving through the OLD, pre-SLM5 format (gpu_fixed/v5 cannot:
rem it always writes 'SLM5').
if not exist obj_gpu\gpu_fixed_noslm5 mkdir obj_gpu\gpu_fixed_noslm5
cl /nologo /std:c++20 /O2 /W3 /fp:precise /EHsc /Ipristine_pre_slm5 /I%ENG%\include /I%ENG%\src\gpu ^
    /c D:\_te338\gpu_1p0_fixed.cpp pristine_pre_slm5\superslm_gpu_pristine.cpp /Fo"obj_gpu\gpu_fixed_noslm5\\" ^
    > obj_gpu\gpu_fixed_noslm5.buildlog 2>&1 || (echo BUILD FAILED: gpu_fixed_noslm5 & type obj_gpu\gpu_fixed_noslm5.buildlog & set OVERALL_OK=0)

rem MUT_CHECKEDRETURN / MUT_NOREARM: single-line reverts of gpu_1p0_v5.cpp (Claude/Curie/t2900-
rem probe/ carries the two make_mut_*.py generators); reuse gpu_fixed's own superslm_gpu_v5.obj
rem unchanged (the mutants touch gpu_1p0.cpp only).
if not exist obj_gpu\gpu_mut_cr mkdir obj_gpu\gpu_mut_cr
cl /nologo /std:c++20 /O2 /W3 /fp:precise /EHsc /I%REFS%\include_override /I%ENG%\include /I%ENG%\src\gpu ^
    /c %REFS%\gpu_1p0_mut_checkedreturn.cpp /Fo"obj_gpu\gpu_mut_cr\\" ^
    > obj_gpu\gpu_mut_cr.buildlog 2>&1 || (echo BUILD FAILED: gpu_mut_cr & type obj_gpu\gpu_mut_cr.buildlog & set OVERALL_OK=0)
if not exist obj_gpu\gpu_mut_nr mkdir obj_gpu\gpu_mut_nr
cl /nologo /std:c++20 /O2 /W3 /fp:precise /EHsc /I%REFS%\include_override /I%ENG%\include /I%ENG%\src\gpu ^
    /c %REFS%\gpu_1p0_mut_norearm.cpp /Fo"obj_gpu\gpu_mut_nr\\" ^
    > obj_gpu\gpu_mut_nr.buildlog 2>&1 || (echo BUILD FAILED: gpu_mut_nr & type obj_gpu\gpu_mut_nr.buildlog & set OVERALL_OK=0)

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
if not exist obj_gpu\cpu_fixed mkdir obj_gpu\cpu_fixed
cl /nologo /std:c++20 /O2 /W3 /fp:precise /EHsc /I%ENG%\include /I%ENG%\src /c %REFS%\sslm_abi_cpu_fixed_v5.cpp ^
    /Fo"obj_gpu\cpu_fixed\sslm_abi.obj" > obj_gpu\cpu_fixed.buildlog 2>&1 || (echo BUILD FAILED: cpu_fixed & type obj_gpu\cpu_fixed.buildlog & set OVERALL_OK=0)

echo ================= Compiling and linking cells =================

rem ---- cell_gpu_cell1_shortschema: GPU-only, four linkage variants (AS_BUILT/FIXED/MUT_CR/MUT_NR) ----
cl /nologo /std:c++20 /O2 /W3 /fp:precise /EHsc %STOCKINC% /c cell_gpu_cell1_shortschema.cpp ^
    /Fo"obj_gpu\cell1_short_stock.obj" > obj_gpu\cell1_short_stock.buildlog 2>&1 || (echo BUILD FAILED: cell1_short_stock.obj & type obj_gpu\cell1_short_stock.buildlog & set OVERALL_OK=0)
cl /nologo /std:c++20 /O2 /W3 /fp:precise /EHsc %OVRINC% /c cell_gpu_cell1_shortschema.cpp ^
    /Fo"obj_gpu\cell1_short_ovr.obj" > obj_gpu\cell1_short_ovr.buildlog 2>&1 || (echo BUILD FAILED: cell1_short_ovr.obj & type obj_gpu\cell1_short_ovr.buildlog & set OVERALL_OK=0)

for %%v in (ASBUILT FIXED MUT_CHECKEDRETURN MUT_NOREARM) do (
    if "%%v"=="ASBUILT" (
        set CELLOBJ=obj_gpu\cell1_short_stock.obj
        set GPUOBJS=obj_gpu\gpu_asbuilt\gpu_1p0.obj obj_gpu\gpu_asbuilt\superslm_gpu.obj
    )
    if "%%v"=="FIXED" (
        set CELLOBJ=obj_gpu\cell1_short_ovr.obj
        set GPUOBJS=obj_gpu\gpu_fixed\gpu_1p0_v5.obj obj_gpu\gpu_fixed\superslm_gpu_v5.obj
    )
    if "%%v"=="MUT_CHECKEDRETURN" (
        set CELLOBJ=obj_gpu\cell1_short_ovr.obj
        set GPUOBJS=obj_gpu\gpu_mut_cr\gpu_1p0_mut_checkedreturn.obj obj_gpu\gpu_fixed\superslm_gpu_v5.obj
    )
    if "%%v"=="MUT_NOREARM" (
        set CELLOBJ=obj_gpu\cell1_short_ovr.obj
        set GPUOBJS=obj_gpu\gpu_mut_nr\gpu_1p0_mut_norearm.obj obj_gpu\gpu_fixed\superslm_gpu_v5.obj
    )
    echo ===== cell_gpu_cell1_shortschema [%%v] =====
    link /nologo /OUT:"bin_gpu\cell1_short_%%v.exe" !CELLOBJ! !GPUOBJS! !CPU_COMMON_OBJS! %SYSLIBS% ^
        > "obj_gpu\cell1_short_%%v.linklog" 2>&1
    if errorlevel 1 (
        echo    LINK FAILED: & type "obj_gpu\cell1_short_%%v.linklog" & set OVERALL_OK=0
    ) else (
        "bin_gpu\cell1_short_%%v.exe" %MODELARG% > "obj_gpu\cell1_short_%%v.runlog" 2>&1
        type "obj_gpu\cell1_short_%%v.runlog"
        findstr /R "^checks=[0-9]* failures=[0-9]*" "obj_gpu\cell1_short_%%v.runlog" >nul || (echo    CRASHED OR NO SUMMARY LINE & set OVERALL_OK=0)
    )
)

rem ---- cell_gpu_cell1_realschema (+ its CPU-side helper): AS_BUILT and FIXED ----
if not exist obj_gpu\cell1_real_stock_ mkdir obj_gpu\cell1_real_stock_
if not exist obj_gpu\cell1_real_ovr_ mkdir obj_gpu\cell1_real_ovr_
cl /nologo /std:c++20 /O2 /W3 /fp:precise /EHsc %STOCKINC% /c cell_gpu_cell1_realschema.cpp cell_gpu_cell1_realschema_cpu_side.cpp ^
    /Fo"obj_gpu\cell1_real_stock_\\" > obj_gpu\cell1_real_stock.buildlog 2>&1 || (echo BUILD FAILED: cell1_real stock & type obj_gpu\cell1_real_stock.buildlog & set OVERALL_OK=0)
cl /nologo /std:c++20 /O2 /W3 /fp:precise /EHsc %OVRINC% /c cell_gpu_cell1_realschema.cpp cell_gpu_cell1_realschema_cpu_side.cpp ^
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
        set GPUOBJS=obj_gpu\gpu_fixed\gpu_1p0_v5.obj obj_gpu\gpu_fixed\superslm_gpu_v5.obj
        set ABIOBJ=obj_gpu\cpu_fixed\sslm_abi.obj
    )
    echo ===== cell_gpu_cell1_realschema [%%v] =====
    link /nologo /OUT:"bin_gpu\cell1_real_%%v.exe" !CELLMAIN! !CELLCPU! !GPUOBJS! !CPU_COMMON_OBJS! !ABIOBJ! %SYSLIBS% ^
        > "obj_gpu\cell1_real_%%v.linklog" 2>&1
    if errorlevel 1 (
        echo    LINK FAILED: & type "obj_gpu\cell1_real_%%v.linklog" & set OVERALL_OK=0
    ) else (
        "bin_gpu\cell1_real_%%v.exe" %G5ARG% > "obj_gpu\cell1_real_%%v.runlog" 2>&1
        type "obj_gpu\cell1_real_%%v.runlog"
        findstr /R "^checks=[0-9]* failures=[0-9]*" "obj_gpu\cell1_real_%%v.runlog" >nul || (echo    CRASHED OR NO SUMMARY LINE & set OVERALL_OK=0)
    )
)

rem ---- cell_gpu_cell2_degenerate: no-macro (always green/skip) and macro-defined ----
rem T-2905: the production seam (ArmGpuFinishDegenerateLogitRowInjection, gpu_1p0.cpp, compiled
rem under SUPERSLM_GPU_G5_FINISH_ROW_FAULT_INJECTION) now exists in obj_gpu\gpu_asbuilt -- this
rem block no longer expects LNK2019. It links against the live tree's own gpu_asbuilt objects
rem (built above from %ENG%\src\gpu, this ticket's own fix) and runs the placeholder body a real
rem cell is still owed (T-2900's own header comment: "exercise once the seam exists" -- driving a
rem real schema to a reachable interior state, arming the seam, asserting -2/SSLM_OK, is the test
rem author's to author, not this build script's).
cl /nologo /std:c++20 /O2 /W3 /fp:precise /EHsc /I. /c cell_gpu_cell2_degenerate.cpp /Fo"obj_gpu\cell2_nomacro.obj" ^
    > obj_gpu\cell2_nomacro.buildlog 2>&1 || (echo BUILD FAILED: cell2_nomacro & type obj_gpu\cell2_nomacro.buildlog & set OVERALL_OK=0)
link /nologo /OUT:"bin_gpu\cell2_nomacro.exe" obj_gpu\cell2_nomacro.obj > obj_gpu\cell2_nomacro.linklog 2>&1
if errorlevel 1 (
    echo LINK FAILED unexpectedly: cell2_nomacro
    type obj_gpu\cell2_nomacro.linklog
    set OVERALL_OK=0
) else (
    "bin_gpu\cell2_nomacro.exe" > obj_gpu\cell2_nomacro.runlog 2>&1
    type obj_gpu\cell2_nomacro.runlog
)
cl /nologo /std:c++20 /O2 /W3 /fp:precise /EHsc /I. /DSUPERSLM_GPU_G5_FINISH_ROW_FAULT_INJECTION /c cell_gpu_cell2_degenerate.cpp ^
    /Fo"obj_gpu\cell2_macro.obj" > obj_gpu\cell2_macro.buildlog 2>&1 || (echo BUILD FAILED: cell2_macro.obj & type obj_gpu\cell2_macro.buildlog & set OVERALL_OK=0)
echo ===== cell_gpu_cell2_degenerate [macro defined, seam now landed -- T-2905] =====
link /nologo /OUT:"bin_gpu\cell2_macro.exe" obj_gpu\cell2_macro.obj obj_gpu\gpu_asbuilt\gpu_1p0.obj obj_gpu\gpu_asbuilt\superslm_gpu.obj !CPU_COMMON_OBJS! %SYSLIBS% ^
    > obj_gpu\cell2_macro.linklog 2>&1
if errorlevel 1 (
    echo    LINK FAILED unexpectedly, now that the seam exists:
    type obj_gpu\cell2_macro.linklog
    set OVERALL_OK=0
) else (
    "bin_gpu\cell2_macro.exe" > obj_gpu\cell2_macro.runlog 2>&1
    type obj_gpu\cell2_macro.runlog
    findstr /R "^checks=[0-9]* failures=[0-9]*" obj_gpu\cell2_macro.runlog >nul || (echo    CRASHED OR NO SUMMARY LINE & set OVERALL_OK=0)
)

rem ---- cell_gpu_cell3_agreement (+ CPU-side helper): AS_BUILT, FIXED, and FIXED+--mutant ----
if not exist obj_gpu\cell3_stock_ mkdir obj_gpu\cell3_stock_
if not exist obj_gpu\cell3_ovr_ mkdir obj_gpu\cell3_ovr_
cl /nologo /std:c++20 /O2 /W3 /fp:precise /EHsc %STOCKINC% /c cell_gpu_cell3_agreement.cpp cell_gpu_cell3_cpu_side.cpp ^
    /Fo"obj_gpu\cell3_stock_\\" > obj_gpu\cell3_stock.buildlog 2>&1 || (echo BUILD FAILED: cell3 stock & type obj_gpu\cell3_stock.buildlog & set OVERALL_OK=0)
cl /nologo /std:c++20 /O2 /W3 /fp:precise /EHsc %OVRINC% /c cell_gpu_cell3_agreement.cpp cell_gpu_cell3_cpu_side.cpp ^
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
        set GPUOBJS=obj_gpu\gpu_fixed\gpu_1p0_v5.obj obj_gpu\gpu_fixed\superslm_gpu_v5.obj
        set ABIOBJ=obj_gpu\cpu_fixed\sslm_abi.obj
    )
    echo ===== cell_gpu_cell3_agreement [%%v] =====
    link /nologo /OUT:"bin_gpu\cell3_%%v.exe" !CELLMAIN! !CELLCPU! !GPUOBJS! !CPU_COMMON_OBJS! !ABIOBJ! %SYSLIBS% ^
        > "obj_gpu\cell3_%%v.linklog" 2>&1
    if errorlevel 1 (
        echo    LINK FAILED: & type "obj_gpu\cell3_%%v.linklog" & set OVERALL_OK=0
    ) else (
        "bin_gpu\cell3_%%v.exe" %MODELARG% > "obj_gpu\cell3_%%v.runlog" 2>&1
        type "obj_gpu\cell3_%%v.runlog"
        findstr /R "^checks=[0-9]* failures=[0-9]*" "obj_gpu\cell3_%%v.runlog" >nul || (echo    CRASHED OR NO SUMMARY LINE & set OVERALL_OK=0)
        if "%%v"=="FIXED" (
            "bin_gpu\cell3_%%v.exe" %MODELARG% --mutant > "obj_gpu\cell3_%%v_mutant.runlog" 2>&1
            type "obj_gpu\cell3_%%v_mutant.runlog"
            findstr /R "^checks=[0-9]* failures=[0-9]*" "obj_gpu\cell3_%%v_mutant.runlog" >nul || (echo    CRASHED OR NO SUMMARY LINE & set OVERALL_OK=0)
        )
    )
)

rem ---- cell_gpu_slm5_saverestore: GPU-only, AS_BUILT and FIXED ----
cl /nologo /std:c++20 /O2 /W3 /fp:precise /EHsc %STOCKINC% /c cell_gpu_slm5_saverestore.cpp /Fo"obj_gpu\slm5_stock.obj" ^
    > obj_gpu\slm5_stock.buildlog 2>&1 || (echo BUILD FAILED: slm5_stock.obj & type obj_gpu\slm5_stock.buildlog & set OVERALL_OK=0)
cl /nologo /std:c++20 /O2 /W3 /fp:precise /EHsc %OVRINC% /c cell_gpu_slm5_saverestore.cpp /Fo"obj_gpu\slm5_ovr.obj" ^
    > obj_gpu\slm5_ovr.buildlog 2>&1 || (echo BUILD FAILED: slm5_ovr.obj & type obj_gpu\slm5_ovr.buildlog & set OVERALL_OK=0)
for %%v in (ASBUILT FIXED) do (
    if "%%v"=="ASBUILT" (
        set CELLOBJ=obj_gpu\slm5_stock.obj
        set GPUOBJS=obj_gpu\gpu_asbuilt\gpu_1p0.obj obj_gpu\gpu_asbuilt\superslm_gpu.obj
    )
    if "%%v"=="FIXED" (
        set CELLOBJ=obj_gpu\slm5_ovr.obj
        set GPUOBJS=obj_gpu\gpu_fixed\gpu_1p0_v5.obj obj_gpu\gpu_fixed\superslm_gpu_v5.obj
    )
    echo ===== cell_gpu_slm5_saverestore [%%v] =====
    link /nologo /OUT:"bin_gpu\slm5_%%v.exe" !CELLOBJ! !GPUOBJS! !CPU_COMMON_OBJS! %SYSLIBS% ^
        > "obj_gpu\slm5_%%v.linklog" 2>&1
    if errorlevel 1 (
        echo    LINK FAILED: & type "obj_gpu\slm5_%%v.linklog" & set OVERALL_OK=0
    ) else (
        "bin_gpu\slm5_%%v.exe" %MODELARG% > "obj_gpu\slm5_%%v.runlog" 2>&1
        type "obj_gpu\slm5_%%v.runlog"
        findstr /R "^checks=[0-9]* failures=[0-9]*" "obj_gpu\slm5_%%v.runlog" >nul || (echo    CRASHED OR NO SUMMARY LINE & set OVERALL_OK=0)
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
findstr /R "^checks=[0-9]* failures=[0-9]*" obj_gpu\slm4_dump.runlog >nul || (echo    CRASHED OR NO SUMMARY LINE & set OVERALL_OK=0)

cl /nologo /std:c++20 /O2 /W3 /fp:precise /EHsc %OVRINC% /c cell_gpu_slm4_restore.cpp /Fo"obj_gpu\slm4_restore.obj" ^
    > obj_gpu\slm4_restore.buildlog 2>&1 || (echo BUILD FAILED: slm4_restore.obj & type obj_gpu\slm4_restore.buildlog & set OVERALL_OK=0)
link /nologo /OUT:"bin_gpu\slm4_restore.exe" obj_gpu\slm4_restore.obj obj_gpu\gpu_fixed\gpu_1p0_v5.obj obj_gpu\gpu_fixed\superslm_gpu_v5.obj !CPU_COMMON_OBJS! %SYSLIBS% ^
    > obj_gpu\slm4_restore.linklog 2>&1 || (echo LINK FAILED: slm4_restore & type obj_gpu\slm4_restore.linklog & set OVERALL_OK=0)
echo ===== cell_gpu_slm4_restore [FIXED, reading the ASBUILT-written blob] =====
"bin_gpu\slm4_restore.exe" %MODELARG% --in=obj_gpu\slm4_blob.bin > obj_gpu\slm4_restore.runlog 2>&1
type obj_gpu\slm4_restore.runlog
findstr /R "^checks=[0-9]* failures=[0-9]*" obj_gpu\slm4_restore.runlog >nul || (echo    CRASHED OR NO SUMMARY LINE & set OVERALL_OK=0)

echo DEBUG_OVERALL_OK=[%OVERALL_OK%]
if "%OVERALL_OK%"=="1" (
    echo ===== build_red_suite_gpu: all cells built and ran =====
    exit /b 0
) else (
    echo ===== build_red_suite_gpu: FAILURES ABOVE =====
    exit /b 1
)
