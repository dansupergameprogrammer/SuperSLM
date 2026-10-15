@echo off
rem Paged-KV plan (rev 16.2) step G1: the GPU red suite's ONE box build (MSVC). It replaces the groups' drafts
rem (g1b_build_draft.bat, g1d_build_draft.bat, g1e_build_draft.bat) and builds the same executables as
rem build_paged_kv_gpu_mingw.sh (the cloud build); the two tables below must stay in step.
rem
rem Usage: tests\paged-kv-gpu\build_paged_kv_gpu.bat ["--out=DIR"] ["--shaders=DIR"] ["--gpu-dir=DIR"]
rem                                                  ["--old-tree=DIR"] ["--old-cpu-lib=FILE"] [TARGET ...]
rem   TARGET     g2 | g3 | g4 | g5 | g6: every executable whose bracket is that step (default: all five);
rem              or one executable's name (pkvg_g3, g1d_cell_3_6f, ...); oldreader: 9.8's v1.12.0 reader only
rem   --out      default <repo>\build\pkvg (executables in <out>\bin, shaders copied to <out>\bin\shaders)
rem   --shaders  default <repo>\out\shaders (build.bat's), else <repo>\build\Release\shaders (CMake's)
rem   --gpu-dir  a whole copy of src\gpu: how a mutant engine is built (3.12's and 11.4's commissioning runs)
rem   --old-tree the v1.12.0 checkout (5d5ca58) for 9.8's leg 2; --old-cpu-lib its CPU library
rem              (default <old-tree>\build\Release\superslm.lib)
rem   --state-tracking  build engine and cells with /DSUPERSLM_GPU_STATE_TRACKING (src\gpu\gpu_resource_state_tracker.h,
rem              as build_paged_kv_gpu_mingw.sh --state-tracking): "# GPU STATE VIOLATION" lines and an exit summary.
rem              Default --out becomes <repo>\build\pkvg-track, so tracking and plain objects never mix.
rem
rem Each executable is reported LINKED, RED BY LINK (its LNK2019/LNK2001 lines), or RED BY COMPILE. RED BY
rem LINK is the expected state of every executable of a step that has not landed: a step's executables link
rem only the verbs and seams that step defines (see g1-unify-notes.md for the exact lists). Exit 0 unless an
rem engine library failed to build.
rem
rem Engine: build.bat's superslm_tests source list restricted to src\ (CPU library once; GPU library per
rem configuration). Configurations, and the cells compiled with the same definitions:
rem   plain  (none)                                                pkvg_g2..g5, g1d 3.12, g1e, G6 timing
rem   alloc  /DSUPERSLM_GPU_ALLOC_FAULT_INJECTION /DSUPERSLM_GPU_TEST_SEAMS  11.4, 3.6d first touch, g1d 3.6e/f/g
rem   t2169  /DSUPERSLM_T2169_CHUNK_RECORDING_FAULT_INJECTION       g1d 3.6g recovered
rem   bench  /DSUPERSLM_ENABLE_GPU_BENCH_PRE_BATCHING               3.6d bench prefill
rem
rem Then run the whole suite at a step, once per GPU (see g1-unify-notes.md for the full command block):
rem   set SSLM_GPU_ADAPTER_INDEX=0
rem   <out>\bin\pkvg_suite.exe "--bin=<out>\bin" --step=G3 "--pkv-fixtures=%FX%" "--gpu-ref=%REFS%" ...
rem Debug layer: every job runs with it ON except the te425 hook jobs (3.6d first touch, 3.6f stranded/waitnoarm,
rem 3.6g stranded/percontext, the nine pins), which run with it OFF; each job line and the summary say which.
setlocal enabledelayedexpansion
set HEREDIR=%~dp0
for %%I in ("%HEREDIR%..\..") do set ENG=%%~fI
set OUT=%ENG%\build\pkvg
set SHADERS=
set GPUDIR=%ENG%\src\gpu
set OLDTREE=
set OLDCPULIB=
set TARGETS=
set TRACK=
set OUTGIVEN=
:parse_args
if "%~1"=="" goto :args_done
set "ARG=%~1"
set "TAKEN="
if "!ARG:~0,6!"=="--out=" ( set "OUT=!ARG:~6!" & set "OUTGIVEN=1" & set "TAKEN=1" )
if "!ARG!"=="--state-tracking" ( set "TRACK=1" & set "TAKEN=1" )
if "!ARG:~0,10!"=="--shaders=" ( set "SHADERS=!ARG:~10!" & set "TAKEN=1" )
if "!ARG:~0,10!"=="--gpu-dir=" ( set "GPUDIR=!ARG:~10!" & set "TAKEN=1" )
if "!ARG:~0,11!"=="--old-tree=" ( set "OLDTREE=!ARG:~11!" & set "TAKEN=1" )
if "!ARG:~0,14!"=="--old-cpu-lib=" ( set "OLDCPULIB=!ARG:~14!" & set "TAKEN=1" )
if not defined TAKEN set "TARGETS=!TARGETS! %~1"
shift
goto :parse_args
:args_done
if "%TARGETS%"=="" set TARGETS= g2 g3 g4 g5 g6
if defined TRACK if not defined OUTGIVEN set OUT=%ENG%\build\pkvg-track
if "%SHADERS%"=="" set SHADERS=%ENG%\out\shaders
if not exist "%SHADERS%\*.cso" set SHADERS=%ENG%\build\Release\shaders
call "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 -no_logo
if exist "%OUT%\cells" rmdir /S /Q "%OUT%\cells"
for %%D in (cpu gpu_plain gpu_alloc gpu_t2169 gpu_bench cells bin) do if not exist "%OUT%\%%D" mkdir "%OUT%\%%D"
set T2791=%ENG%\tests\t2791-gpu-prefill-read-red-suite
set TE425=%ENG%\tests\te425-gpu-alloc-status-red-suite
set CLF=/nologo /std:c++20 /O2 /W4 /fp:precise /EHsc /I"%ENG%\include" /I"%ENG%\src" /I"%GPUDIR%" /I"%ENG%\tests" ^
 /I"%ENG%\tests\paged-kv" /I"%HEREDIR%." /I"%T2791%" /I"%TE425%"
if defined TRACK set CLF=%CLF% /DSUPERSLM_GPU_STATE_TRACKING
set DEF_plain=
set DEF_alloc=/DSUPERSLM_GPU_ALLOC_FAULT_INJECTION /DSUPERSLM_GPU_TEST_SEAMS
set DEF_t2169=/DSUPERSLM_T2169_CHUNK_RECORDING_FAULT_INJECTION
set DEF_bench=/DSUPERSLM_ENABLE_GPU_BENCH_PRE_BATCHING
set NLINKED=0
set NRED=0
set NCOMPILE=0

if "%TARGETS%"==" oldreader" goto :only_old

echo ===== engine: CPU sources (build.bat's list) =====
cl /c %CLF% /Fo"%OUT%\cpu\\" "%ENG%\src\artifact.cpp" "%ENG%\src\sha256.cpp" "%ENG%\src\tokenizer.cpp" ^
    "%ENG%\src\model.cpp" "%ENG%\src\intmath.cpp" "%ENG%\src\silu_lut.cpp" "%ENG%\src\matmul.cpp" ^
    "%ENG%\src\proof_manifest.cpp" "%ENG%\src\trace_hook.cpp" "%ENG%\src\forward\checked_chain_funnel.cpp" ^
    "%ENG%\src\forward\forward_sites.cpp" "%ENG%\src\forward\parallel_split.cpp" "%ENG%\src\gpu\gpu_inflight_registry.cpp" "%ENG%\src\decode_digest.cpp" ^
    "%ENG%\src\kv_pages.cpp" "%ENG%\src\sslm_abi.cpp" "%ENG%\src\damped_greedy_antilm.cpp" "%ENG%\src\damped_greedy_topk.cpp" ^
    "%ENG%\src\damped_greedy_phaseD.cpp" "%ENG%\src\damped_greedy_phaseD_loop.cpp" > "%OUT%\cpu\build.log" 2>&1
if errorlevel 1 ( type "%OUT%\cpu\build.log" & echo ENGINE CPU BUILD FAILED & exit /b 3 )
call :mklib cpu
if errorlevel 1 exit /b 3
call :gpulib plain
if errorlevel 1 exit /b 3
call :gpulib alloc
if errorlevel 1 exit /b 3
call :gpulib t2169
if errorlevel 1 exit /b 3
call :gpulib bench
if errorlevel 1 exit /b 3

echo ===== te425 hooks (InstallHooks), main renamed, allocation-fault definitions =====
cl /c %CLF% %DEF_alloc% /Dmain=te425_cells_main /Fo"%OUT%\cells\te425_cells.obj" "%TE425%\te425_cells.cpp" > "%OUT%\cells\te425_cells.log" 2>&1
if errorlevel 1 ( type "%OUT%\cells\te425_cells.log" & exit /b 3 )
cl /c %CLF% %DEF_alloc% /Fo"%OUT%\cells\te425_getdevice.obj" "%TE425%\te425_getdevice.cpp" > "%OUT%\cells\te425_getdevice.log" 2>&1
if errorlevel 1 ( type "%OUT%\cells\te425_getdevice.log" & exit /b 3 )
set HOOKS="%OUT%\cells\te425_cells.obj" "%OUT%\cells\te425_getdevice.obj"

if exist "%SHADERS%\*.cso" (
    if not exist "%OUT%\bin\shaders" mkdir "%OUT%\bin\shaders"
    xcopy /Y /Q "%SHADERS%\*.cso" "%OUT%\bin\shaders\" >nul
) else (
    echo WARNING: no shaders at %SHADERS% -- run build.bat first; the GPU cells will fail at context creation
)

echo ===== host: the suite's process runner =====
cl %CLF% /Fe"%OUT%\bin\pkvg_suite.exe" /Fo"%OUT%\cells\pkvg_suite.obj" "%HEREDIR%pkvg_suite.cpp" > "%OUT%\cells\pkvg_suite.log" 2>&1
if errorlevel 1 ( type "%OUT%\cells\pkvg_suite.log" & set /a NCOMPILE+=1 )

rem ---- the suite's executables (keep in step with build_paged_kv_gpu_mingw.sh's TABLE) ---------------------------
rem  name | configuration | step | sources (NAME@DEF compiles NAME with /DDEF; '#' stands for '=') | hooks
call :exe pkvg_g2 plain G2 "pkvg_main pkvg_group_g1a g1a_g2_legacy_view" ""
call :exe pkvg_g3 plain G3 "pkvg_main pkvg_group_g1a g1a_g2_legacy_view g1a_g3_pool pkvg_group_g1b g1b_g3_cells pkvg_group_g1c@PKVG_STEP#3 g1c_cell_3_6@PKVG_STEP#3 g1c_cell_3_6b@PKVG_STEP#3 g1c_cell_3_6c@PKVG_STEP#3 g1c_cell_3_6d@PKVG_STEP#3 g1c_debug_layer@PKVG_STEP#3" ""
call :exe pkvg_alloc_g3 alloc G3 "pkvg_main pkvg_group_g1b g1b_g3_pool_ceiling" ""
call :exe pkvg_bench_g3 bench G3 "pkvg_main pkvg_group_g1c@PKVG_STEP#3 g1c_cell_3_6d_bench@PKVG_STEP#3 g1c_debug_layer@PKVG_STEP#3" ""
call :exe pkvg_firsttouch_g3 alloc G3 "pkvg_main pkvg_group_g1c@PKVG_STEP#3 g1c_cell_3_6d_first_touch@PKVG_STEP#3 g1c_debug_layer@PKVG_STEP#3" hooks
call :exe g1d_cell_3_6e_g3 alloc G3 "g1d_cell_3_6e@G1D_3_6E_G3_ONLY" ""
call :exe g1d_cell_3_6f alloc G3 "g1d_cell_3_6f" hooks
call :exe g1d_cell_3_6g_recovered t2169 G3 "g1d_cell_3_6g_recovered" ""
call :exe g1d_cell_3_6g_stranded alloc G3 "g1d_cell_3_6g_stranded" hooks
call :exe pkvg_g4 plain G4 "pkvg_main pkvg_group_g1a g1a_g2_legacy_view g1a_g3_pool g1a_g4_prefix pkvg_group_g1b g1b_g3_cells g1b_g4_cells pkvg_group_g1c@PKVG_STEP#4 g1c_cell_3_6@PKVG_STEP#4 g1c_cell_3_6b@PKVG_STEP#4 g1c_cell_3_6c@PKVG_STEP#4 g1c_cell_3_6d@PKVG_STEP#4 g1c_debug_layer@PKVG_STEP#4" ""
call :exe g1d_cell_3_6e alloc G4 "g1d_cell_3_6e" ""
call :exe g1d_cell_3_12_g4 plain G4 "g1d_cell_3_12@G1D_3_12_G4_ONLY" ""
call :exe pkvg_g5 plain G5 "pkvg_main pkvg_group_g1a g1a_g2_legacy_view g1a_g3_pool g1a_g4_prefix g1a_g5_slm6 pkvg_group_g1b g1b_g3_cells g1b_g4_cells g1b_g5_cells pkvg_group_g1c@PKVG_STEP#5 g1c_cell_3_6@PKVG_STEP#5 g1c_cell_3_6b@PKVG_STEP#5 g1c_cell_3_6c@PKVG_STEP#5 g1c_cell_3_6d@PKVG_STEP#5 g1c_debug_layer@PKVG_STEP#5" ""
call :exe g1d_cell_3_12 plain G5 "g1d_cell_3_12" ""
call :exe g1e_cell_9_6 plain G5 "g1e_cell_9_6" ""
call :exe g1e_cell_9_8 plain G5 "g1e_cell_9_8" ""
call :exe g1e_cell_9_12 plain G5 "g1e_cell_9_12" ""
call :exe g1e_cell_9_13 plain G5 "g1e_cell_9_13" ""
call :exe g1e_cell_8_17 plain G5 "g1e_cell_8_17" ""
call :exe g1e_cell_slm6_cap plain G5 "g1e_cell_slm6_cap" ""
rem 1.13.0 contract fold: the same-handle decode race for every decode entry point (allocation-fault engine: its claim seam).
call :exe pkvg_same_handle_race alloc G5 "pkvg_same_handle_race" ""
rem G6 (plan §7 10.3, 10.5, 10.7; §10 R1): the GPU cohort and admission cells, and the decode timing harness
rem (plain configuration: the harness refuses any seam definition at compile time).
call :exe g1e_cell_10_3 plain G6 "g1e_cell_10_3" ""
call :exe g1e_cell_10_5 plain G6 "g1e_cell_10_5" ""
call :exe g1e_cell_10_7 plain G6 "g1e_cell_10_7" ""
call :exe pkvg_g6_decode_timing plain G6 "pkvg_g6_decode_timing" ""
:only_old
if defined OLDTREE call :oldreader
echo PKVG BUILD: %NLINKED% linked, %NRED% red by link, %NCOMPILE% red by compile or unexpected link failure
exit /b 0

rem ---- :exe NAME CONFIG STEP "SOURCES" HOOKS ------------------------------------------------------------------------
:exe
set EXE=%~1
set CFG=%~2
set STEP=%~3
set SRCS=%~4
set XHOOKS=%~5
set WANT=0
for %%T in (%TARGETS%) do (
    if /I "%%T"=="%EXE%" set WANT=1
    if /I "%%T"=="%STEP%" set WANT=1
)
if "%WANT%"=="0" exit /b 0
set OBJS=
set CELLFAIL=
for %%S in (%SRCS%) do (
    set "E=%%S"
    call :cell !E:@= !
)
if defined CELLFAIL exit /b 0
set EXTRA=
if "%XHOOKS%"=="hooks" set EXTRA=%HOOKS%
link /nologo /OUT:"%OUT%\bin\%EXE%.exe" %OBJS% %EXTRA% "%OUT%\cpu.lib" "%OUT%\gpu_%CFG%.lib" ^
    d3d12.lib dxgi.lib dxguid.lib shlwapi.lib psapi.lib > "%OUT%\cells\%EXE%.link.log" 2>&1
if not errorlevel 1 ( echo [%STEP%] %EXE%: LINKED ^(%CFG%^) & set /a NLINKED+=1 & exit /b 0 )
if exist "%OUT%\bin\%EXE%.exe" del "%OUT%\bin\%EXE%.exe"
findstr /C:"LNK2019" /C:"LNK2001" "%OUT%\cells\%EXE%.link.log" >nul
if not errorlevel 1 (
    echo [%STEP%] %EXE%: RED BY LINK ^(%CFG%^) -- unresolved:
    findstr /C:"LNK2019" /C:"LNK2001" "%OUT%\cells\%EXE%.link.log"
    set /a NRED+=1
) else (
    echo [%STEP%] %EXE%: UNEXPECTED LINK FAILURE -- see %OUT%\cells\%EXE%.link.log
    set /a NCOMPILE+=1
)
exit /b 0

rem ---- :cell SRC [DEF]: compiles one cell source once per configuration and definition, and appends its object
rem to OBJS ('#' in DEF stands for '=': PKVG_STEP#3 is /DPKVG_STEP=3).
:cell
if defined CELLFAIL exit /b 0
set "SRC=%~1"
set "D=%~2"
set "TAG=none"
set "DFLAG="
if not "%D%"=="" set "TAG=%D:#=_%"
if not "%D%"=="" set "D=%D:#==%"
if not "%D%"=="" set "DFLAG=/D%D%"
set "ODIR=%OUT%\cells\%CFG%_%TAG%"
if not exist "%ODIR%" mkdir "%ODIR%"
if exist "%ODIR%\%SRC%.obj" goto :cell_have
cl /c %CLF% !DEF_%CFG%! %DFLAG% /Fo"%ODIR%\%SRC%.obj" "%HEREDIR%%SRC%.cpp" > "%ODIR%\%SRC%.log" 2>&1
if errorlevel 1 (
    echo [%STEP%] %EXE%: RED BY COMPILE in %SRC%.cpp
    findstr /R /C:"error C[0-9]*" "%ODIR%\%SRC%.log"
    set /a NCOMPILE+=1
    set CELLFAIL=1
    exit /b 0
)
:cell_have
set OBJS=%OBJS% "%ODIR%\%SRC%.obj"
exit /b 0

rem ---- :gpulib CONFIG: every GPU source of build.bat's list, with the configuration's definitions ----------------
:gpulib
echo ===== engine: GPU sources, configuration %1 =====
if exist "%OUT%\gpu_%1\*.obj" del /Q "%OUT%\gpu_%1\*.obj"
for %%S in ("%GPUDIR%\superslm_gpu.cpp" "%GPUDIR%\gpu_1p0.cpp") do (
    cl /c %CLF% !DEF_%1! /Fo"%OUT%\gpu_%1\\" "%%~S" > "%OUT%\gpu_%1\%%~nS.log" 2>&1
    if errorlevel 1 ( type "%OUT%\gpu_%1\%%~nS.log" & echo ENGINE GPU BUILD FAILED: %%~S ^(%1^) & exit /b 3 )
)
call :mklib gpu_%1
if errorlevel 1 exit /b 3
exit /b 0

:mklib
(for %%o in ("%OUT%\%1\*.obj") do @echo "%%o") > "%OUT%\%1.rsp"
lib /nologo /OUT:"%OUT%\%1.lib" @"%OUT%\%1.rsp" > "%OUT%\%1.lib.log" 2>&1
if errorlevel 1 ( type "%OUT%\%1.lib.log" & echo LIB %1 FAILED & exit /b 3 )
exit /b 0

rem ---- 9.8's leg 2, against the v1.12.0 tree: its own GPU sources, its own CPU library (g1e's draft, unchanged) -----
:oldreader
echo ===== g1e_old_reader_9_8 against %OLDTREE% =====
if not exist "%OUT%\old" mkdir "%OUT%\old"
if not defined OLDCPULIB set OLDCPULIB=%OLDTREE%\build\Release\superslm.lib
if not exist "%OLDCPULIB%" ( echo NO v1.12.0 CPU LIBRARY AT %OLDCPULIB% & set /a NCOMPILE+=1 & exit /b 0 )
rem /MD: the default --old-cpu-lib is the old tree's CMake (Visual Studio generator) library, built /MD.
set OCLF=/nologo /std:c++20 /O2 /W4 /fp:precise /EHsc /MD /I"%OLDTREE%\include"
for %%S in ("%OLDTREE%\src\gpu\superslm_gpu.cpp" "%OLDTREE%\src\gpu\gpu_1p0.cpp") do (
    cl /c %OCLF% /I"%OLDTREE%\src\gpu" /Fo"%OUT%\old\\" "%%~S" > "%OUT%\old\%%~nS.log" 2>&1
    if errorlevel 1 ( type "%OUT%\old\%%~nS.log" & echo v1.12.0 GPU BUILD FAILED & set /a NCOMPILE+=1 & exit /b 0 )
)
cl /c %OCLF% /Fo"%OUT%\old\g1e_old_reader_9_8.obj" "%HEREDIR%g1e_old_reader_9_8.cpp" > "%OUT%\old\reader.log" 2>&1
if errorlevel 1 ( type "%OUT%\old\reader.log" & set /a NCOMPILE+=1 & exit /b 0 )
link /nologo /OUT:"%OUT%\old\g1e_old_reader_9_8.exe" "%OUT%\old\g1e_old_reader_9_8.obj" "%OUT%\old\superslm_gpu.obj" "%OUT%\old\gpu_1p0.obj" ^
    "%OLDCPULIB%" d3d12.lib dxgi.lib dxguid.lib shlwapi.lib >> "%OUT%\old\reader.log" 2>&1
    if errorlevel 1 ( type "%OUT%\old\reader.log" & set /a NCOMPILE+=1 & exit /b 0 )
for %%P in ("%OLDTREE%\out\shaders" "%OLDTREE%\build\Release\shaders") do (
    if exist "%%~P\*.cso" if not exist "%OUT%\old\shaders\*.cso" (
        mkdir "%OUT%\old\shaders" 2>nul
        xcopy /Y /Q "%%~P\*.cso" "%OUT%\old\shaders\" >nul
    )
)
echo    LINKED (v1.12.0): %OUT%\old\g1e_old_reader_9_8.exe   (pkvg_suite --old-bin=%OUT%\old)
exit /b 0
