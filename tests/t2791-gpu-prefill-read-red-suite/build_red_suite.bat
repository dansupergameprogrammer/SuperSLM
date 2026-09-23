@echo off
rem T-2791 (Curie): builds every cell of the SuperSLM 1.6.0 GPU embedding-read red suite against
rem the engine sources in this checkout and classifies each cell's result. Documented-local, like
rem every GPU red suite here (D-SLM3432: no GPU CI runner). The CPU-only source scan of plan
rem Sec3.4 row 7 lives in tests\ci\test_t2791_gpu_fault_pin_env_gate.py and runs under pytest.
rem
rem Expected at v1.5.0 (the red reading, Claude/Curie/t2791-gpu-read-red-2026-09-18.md):
rem   cell_status_ordinals            RED BY COMPILE on the two new enumerator names only
rem   every cell calling the verb     RED BY LINK: LNK2019 on sslm_gpu_seq_read_prefill_final_hidden
rem                                   and/or sslm_gpu_model_hidden_size, nothing else
rem   cell_prompt_guard_status        RED BY LINK: LNK2019 on sslm_gpu_seq_read_prefill_final_hidden and
rem                                   superslm_gpu::ArmPrefillGuardDeviceRemovedQueryInjection (plan Sec3.6)
rem   cell_schema_guard_status        RED BY LINK: LNK2019 on sslm_gpu_seq_read_prefill_final_hidden (Q5-1, T-2814)
rem   cell_prompt_guard_removal       LINKS; GREEN AT RUN (Q4-1e: a real removal is SSLM_DEVICE_LOST at 1.5.0
rem                                   too -- a liveness and cleanliness cell, not a classification cell)
rem   cell_env_pins_shipping_leg      LINKS; RED AT RUN (the shipping build still reads the pins)
rem   cell_wrapper_census_standing    LINKS; M16 RED AT RUN on v1.5.0's post-prefill schema unbind;
rem                                   GREEN from v1.6.0 under D-SLM7625, with next-decode census preserved
rem
rem Usage (from any directory):
rem   tests\t2791-gpu-prefill-read-red-suite\build_red_suite.bat ["--out=DIR"] ["--shaders=DIR"]
rem       ["--gpu1p0=FILE"] ["--superslmgpu=FILE"] ["--gpu-dir=DIR"] ["--include-first=DIR"]
rem       ["--extra-define=NAME"] ["--only=CELL"] ["--cpu-lib=FILE"]
rem       (quote each flag)
rem   --out            build output directory (default: <repo>\build\t2791)
rem   --shaders        compiled .cso directory to stage beside the executables
rem                    (default: <repo>\build\Release\shaders, from the CMake SUPERSLM_BUILD_GPU build)
rem   --gpu1p0         compile this file in place of src\gpu\gpu_1p0.cpp -- how a guard-vitality mutant
rem                    of plan Sec3.4 row 11 is built against the real implementation (see the record)
rem   --superslmgpu    compile this file in place of src\gpu\superslm_gpu.cpp -- plan Sec3.6 item 1 and
rem                    mutant (k) live there (plan Sec3.5 step 2)
rem   --gpu-dir        a whole copy of src\gpu: compile its gpu_1p0.cpp and superslm_gpu.cpp, and compile every
rem                    cell whose source includes "d3d12_harness.h" against that directory's header. This is how a
rem                    mutant of the harness header is built (plan Sec3.5 step 2, T-2834 F3): a quoted #include
rem                    "d3d12_harness.h" resolves from the including file's own directory first, so a mutated
rem                    header is read only from a copy of the whole directory, and a cell that includes it must
rem                    read the same copy the engine was built from. Overrides --gpu1p0 and --superslmgpu.
rem   --include-first  an include directory searched before <repo>\include
rem   --extra-define   one extra preprocessor definition for the engine's GPU translation units
rem   --only           build only this cell (any cell_*.cpp in this directory, listed below or not) and skip the
rem                    shipping-configuration library unless the cell needs it (run_fullcopy_mutants.ps1 uses it)
rem   --cpu-lib        reuse an already built CPU engine library instead of compiling the CPU sources
rem The GPU sources are compiled one translation unit per cl call, and each unit's compiler output is kept
rem in <out>\<config>\<unit>.log (as well as <out>\<config>\build.log): run_fullcopy_mutants.ps1 reads a
rem mutant's "SSLM MUTANT <id> APPLIED" #pragma message there, per translation unit.
rem Then run: tests\t2791-gpu-prefill-read-red-suite\run_red_suite.bat with the artifact flags.
setlocal enabledelayedexpansion
set HEREDIR=%~dp0
for %%I in ("%HEREDIR%..\..") do set ENG=%%~fI
set OUT=%ENG%\build\t2791
set SHADERS=%ENG%\build\Release\shaders
set GPU1P0=%ENG%\src\gpu\gpu_1p0.cpp
set SSGPU=%ENG%\src\gpu\superslm_gpu.cpp
set GPUDIR=
set ONLY=
set CPULIB=
set INCFIRST=
set EXTRADEF=
rem Each flag must be passed QUOTED ("--out=D:\x"): cmd splits an unquoted argument at '='.
:parse_args
if "%~1"=="" goto :args_done
set ARG=%~1
if "!ARG:~0,6!"=="--out=" set OUT=!ARG:~6!
if "!ARG:~0,10!"=="--shaders=" set SHADERS=!ARG:~10!
if "!ARG:~0,9!"=="--gpu1p0=" set GPU1P0=!ARG:~9!
if "!ARG:~0,14!"=="--superslmgpu=" set SSGPU=!ARG:~14!
if "!ARG:~0,16!"=="--include-first=" set INCFIRST=/I"!ARG:~16!"
if "!ARG:~0,15!"=="--extra-define=" set EXTRADEF=/D!ARG:~15!
if "!ARG:~0,10!"=="--gpu-dir=" set GPUDIR=!ARG:~10!
if "!ARG:~0,7!"=="--only=" set ONLY=!ARG:~7!
if "!ARG:~0,10!"=="--cpu-lib=" set CPULIB=!ARG:~10!
shift
goto :parse_args
:args_done
if defined GPUDIR (
    set GPU1P0=!GPUDIR!\gpu_1p0.cpp
    set SSGPU=!GPUDIR!\superslm_gpu.cpp
) else (
    set GPUDIR=%ENG%\src\gpu
)
echo T-2791 suite build: out=%OUT% gpu_1p0=%GPU1P0% superslm_gpu=%SSGPU% gpu-dir=%GPUDIR% include-first=%INCFIRST% extra=%EXTRADEF% only=%ONLY%
call "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 -no_logo
for %%D in (cpu gpu_fi gpu_plain cells bin) do if not exist "%OUT%\%%D" mkdir "%OUT%\%%D"
set CLF=/nologo /std:c++20 /O2 /W4 /fp:precise /EHsc %INCFIRST% /I"%ENG%\include" /I"%ENG%\tests" /I"%HEREDIR%."

if defined CPULIB (
    echo ===== engine: CPU library reused from %CPULIB% =====
    if /I not "%CPULIB%"=="%OUT%\cpu.lib" (
        copy /Y "%CPULIB%" "%OUT%\cpu.lib" >nul || ( echo CPU LIBRARY COPY FAILED & exit /b 3 )
    )
    goto :cpu_done
)
echo ===== engine: CPU sources =====
cl /c %CLF% /Fo"%OUT%\cpu\\" "%ENG%\src\artifact.cpp" "%ENG%\src\sha256.cpp" "%ENG%\src\tokenizer.cpp" ^
    "%ENG%\src\model.cpp" "%ENG%\src\intmath.cpp" "%ENG%\src\silu_lut.cpp" "%ENG%\src\matmul.cpp" ^
    "%ENG%\src\proof_manifest.cpp" "%ENG%\src\trace_hook.cpp" "%ENG%\src\forward\checked_chain_funnel.cpp" ^
    "%ENG%\src\forward\forward_sites.cpp" "%ENG%\src\decode_digest.cpp" > "%OUT%\cpu\build.log" 2>&1
if errorlevel 1 ( type "%OUT%\cpu\build.log" & echo ENGINE CPU BUILD FAILED & exit /b 3 )
call :mklib cpu || exit /b 3
:cpu_done

rem The fault-injection engine: the T-2169 chunk-recording seams plan Sec3.4 row 5 names
rem (tests\t2178-gpu-batched-prefill-red-suite\build_red_suite.bat's own precedent).
echo ===== engine: GPU sources, T-2169 fault seams compiled in =====
call :gpulib gpu_fi /DSUPERSLM_T2169_CHUNK_RECORDING_FAULT_INJECTION || exit /b 3

rem The shipping configuration: no test or bench definitions at all, exactly what the installed
rem superslm_gpu library compiles (CMakeLists.txt, target superslm_gpu: /W4 /fp:precise only).
set NEEDPLAIN=1
if defined ONLY (
    set NEEDPLAIN=0
    if "%ONLY%"=="cell_env_pins_shipping_leg" set NEEDPLAIN=1
    if "%ONLY%"=="cell_prompt_guard_removal" set NEEDPLAIN=1
)
if "%NEEDPLAIN%"=="1" (
    echo ===== engine: GPU sources, shipping configuration =====
    call :gpulib gpu_plain || exit /b 3
)

if exist "%SHADERS%" (
    if not exist "%OUT%\bin\shaders" mkdir "%OUT%\bin\shaders"
    xcopy /Y /Q "%SHADERS%\*.cso" "%OUT%\bin\shaders\" >nul
) else (
    echo WARNING: no shaders at %SHADERS% -- the GPU cells will fail at context creation
)

set UNEXPECTED=0
if defined ONLY (
    call :buildcell %ONLY%
    goto :cells_done
)
for %%f in (cell_status_ordinals cell_census_lifetime cell_hostile_capacity cell_prefill_faults_schema cell_final_norm_guard ^
            cell_prompt_guard_status cell_schema_guard_status cell_prompt_guard_removal cell_concurrency ^
            cell_determinism_composition cell_functional_commission ^
            cell_env_pins_shipping_leg cell_wrapper_census_standing) do (
    call :buildcell %%f
)
:cells_done
echo.
if "%UNEXPECTED%"=="1" ( echo SUITE BUILD: unexpected failure class & exit /b 2 )
echo SUITE BUILD: done. A cell that did not link is red by the reason printed above.
exit /b 0

rem One cell. It links the fault-injection library, except the two cells that must be the shipping build: the
rem environment-pin leg, and Q4-1e, which removes the process-wide harness device and needs no seam. A cell whose
rem source includes "d3d12_harness.h" is compiled against --gpu-dir's copy of it.
:buildcell
set CELL=%1
echo ===== %CELL% =====
if not exist "%HEREDIR%%CELL%.cpp" ( echo    NO SUCH CELL: %HEREDIR%%CELL%.cpp & set UNEXPECTED=1 & exit /b 0 )
set VARIANT=gpu_fi
if "%CELL%"=="cell_env_pins_shipping_leg" set VARIANT=gpu_plain
if "%CELL%"=="cell_prompt_guard_removal" set VARIANT=gpu_plain
set CELLINC=
findstr /R /C:"^ *# *include *\"d3d12_harness.h\"" "%HEREDIR%%CELL%.cpp" >nul && set CELLINC=/I"%GPUDIR%"
cl /c %CLF% !CELLINC! /DSUPERSLM_T2169_CHUNK_RECORDING_FAULT_INJECTION /Fo"%OUT%\cells\%CELL%.obj" "%HEREDIR%%CELL%.cpp" > "%OUT%\cells\%CELL%.log" 2>&1
if errorlevel 1 (
    echo    RED BY COMPILE:
    findstr /R /C:"error C[0-9]*" "%OUT%\cells\%CELL%.log"
    exit /b 0
)
link /nologo /OUT:"%OUT%\bin\%CELL%.exe" "%OUT%\cells\%CELL%.obj" "%OUT%\cpu.lib" "%OUT%\!VARIANT!.lib" ^
    d3d12.lib dxgi.lib dxguid.lib >> "%OUT%\cells\%CELL%.log" 2>&1
if not errorlevel 1 (
    echo    LINKED -- run it: %OUT%\bin\%CELL%.exe
    exit /b 0
)
findstr /C:"LNK2019" "%OUT%\cells\%CELL%.log" >nul
if not errorlevel 1 (
    echo    RED BY LINK -- unresolved:
    findstr /C:"LNK2019" "%OUT%\cells\%CELL%.log"
) else (
    echo    UNEXPECTED LINK FAILURE -- see %OUT%\cells\%CELL%.log
    set UNEXPECTED=1
)
exit /b 0

rem One GPU engine library configuration (%1) from %SSGPU% and %GPU1P0%, one translation unit per cl call. Each
rem unit's compiler output is kept in %OUT%\%1\<unit>.log and all of it in %OUT%\%1\build.log. %2 is an extra
rem definition for this configuration, or empty.
:gpulib
if exist "%OUT%\%1\*.obj" del /Q "%OUT%\%1\*.obj"
if exist "%OUT%\%1\build.log" del /Q "%OUT%\%1\build.log"
for %%S in ("%SSGPU%" "%GPU1P0%") do (
    cl /c %CLF% /I"%GPUDIR%" %EXTRADEF% %2 /Fo"%OUT%\%1\\" "%%~S" > "%OUT%\%1\%%~nS.log" 2>&1
    if errorlevel 1 ( type "%OUT%\%1\%%~nS.log" & echo ENGINE GPU BUILD FAILED: %%~S & exit /b 3 )
    type "%OUT%\%1\%%~nS.log" >> "%OUT%\%1\build.log"
)
call :mklib %1 || exit /b 3
exit /b 0

:mklib
(for %%o in ("%OUT%\%1\*.obj") do @echo "%%o") > "%OUT%\%1.rsp"
lib /nologo /OUT:"%OUT%\%1.lib" @"%OUT%\%1.rsp" > "%OUT%\%1.lib.log" 2>&1
if errorlevel 1 ( type "%OUT%\%1.lib.log" & echo LIB %1 FAILED & exit /b 3 )
exit /b 0
