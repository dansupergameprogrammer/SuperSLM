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
rem   cell_prompt_guard_removal       LINKS; GREEN AT RUN (Q4-1e: a real removal is SSLM_DEVICE_LOST at 1.5.0
rem                                   too -- a liveness and cleanliness cell, not a classification cell)
rem   cell_env_pins_shipping_leg      LINKS; RED AT RUN (the shipping build still reads the pins)
rem   cell_wrapper_census_standing    LINKS; GREEN AT RUN (a standing 1.5.0 baseline that must stay green)
rem
rem Usage (from any directory):
rem   tests\t2791-gpu-prefill-read-red-suite\build_red_suite.bat ["--out=DIR"] ["--shaders=DIR"]
rem       ["--gpu1p0=FILE"] ["--superslmgpu=FILE"] ["--include-first=DIR"] ["--extra-define=NAME"]
rem       (quote each flag)
rem   --out            build output directory (default: <repo>\build\t2791)
rem   --shaders        compiled .cso directory to stage beside the executables
rem                    (default: <repo>\build\Release\shaders, from the CMake SUPERSLM_BUILD_GPU build)
rem   --gpu1p0         compile this file in place of src\gpu\gpu_1p0.cpp -- how a guard-vitality mutant
rem                    of plan Sec3.4 row 11 is built against the real implementation (see the record)
rem   --superslmgpu    compile this file in place of src\gpu\superslm_gpu.cpp -- plan Sec3.6 item 1 and
rem                    mutant (k) live there (plan Sec3.5 step 2)
rem   --include-first  an include directory searched before <repo>\include
rem   --extra-define   one extra preprocessor definition for the engine's GPU translation units
rem Then run: tests\t2791-gpu-prefill-read-red-suite\run_red_suite.bat with the artifact flags.
setlocal enabledelayedexpansion
set HEREDIR=%~dp0
for %%I in ("%HEREDIR%..\..") do set ENG=%%~fI
set OUT=%ENG%\build\t2791
set SHADERS=%ENG%\build\Release\shaders
set GPU1P0=%ENG%\src\gpu\gpu_1p0.cpp
set SSGPU=%ENG%\src\gpu\superslm_gpu.cpp
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
shift
goto :parse_args
:args_done
echo T-2791 suite build: out=%OUT% gpu_1p0=%GPU1P0% superslm_gpu=%SSGPU% include-first=%INCFIRST% extra=%EXTRADEF%
call "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 -no_logo
for %%D in (cpu gpu_fi gpu_plain cells bin) do if not exist "%OUT%\%%D" mkdir "%OUT%\%%D"
set CLF=/nologo /std:c++20 /O2 /W4 /fp:precise /EHsc %INCFIRST% /I"%ENG%\include" /I"%ENG%\tests" /I"%HEREDIR%."

echo ===== engine: CPU sources =====
cl /c %CLF% /Fo"%OUT%\cpu\\" "%ENG%\src\artifact.cpp" "%ENG%\src\sha256.cpp" "%ENG%\src\tokenizer.cpp" ^
    "%ENG%\src\model.cpp" "%ENG%\src\intmath.cpp" "%ENG%\src\silu_lut.cpp" "%ENG%\src\matmul.cpp" ^
    "%ENG%\src\proof_manifest.cpp" "%ENG%\src\trace_hook.cpp" "%ENG%\src\forward\checked_chain_funnel.cpp" ^
    "%ENG%\src\forward\forward_sites.cpp" "%ENG%\src\decode_digest.cpp" > "%OUT%\cpu\build.log" 2>&1
if errorlevel 1 ( type "%OUT%\cpu\build.log" & echo ENGINE CPU BUILD FAILED & exit /b 3 )
call :mklib cpu || exit /b 3

rem The fault-injection engine: the T-2169 chunk-recording seams plan Sec3.4 row 5 names
rem (tests\t2178-gpu-batched-prefill-red-suite\build_red_suite.bat's own precedent).
echo ===== engine: GPU sources, T-2169 fault seams compiled in =====
cl /c %CLF% /I"%ENG%\src\gpu" %EXTRADEF% /DSUPERSLM_T2169_CHUNK_RECORDING_FAULT_INJECTION /Fo"%OUT%\gpu_fi\\" ^
    "%SSGPU%" "%GPU1P0%" > "%OUT%\gpu_fi\build.log" 2>&1
if errorlevel 1 ( type "%OUT%\gpu_fi\build.log" & echo ENGINE GPU BUILD FAILED & exit /b 3 )
call :mklib gpu_fi || exit /b 3

rem The shipping configuration: no test or bench definitions at all, exactly what the installed
rem superslm_gpu library compiles (CMakeLists.txt, target superslm_gpu: /W4 /fp:precise only).
echo ===== engine: GPU sources, shipping configuration =====
cl /c %CLF% /I"%ENG%\src\gpu" %EXTRADEF% /Fo"%OUT%\gpu_plain\\" "%SSGPU%" "%GPU1P0%" > "%OUT%\gpu_plain\build.log" 2>&1
if errorlevel 1 ( type "%OUT%\gpu_plain\build.log" & echo ENGINE GPU BUILD FAILED & exit /b 3 )
call :mklib gpu_plain || exit /b 3

if exist "%SHADERS%" (
    if not exist "%OUT%\bin\shaders" mkdir "%OUT%\bin\shaders"
    xcopy /Y /Q "%SHADERS%\*.cso" "%OUT%\bin\shaders\" >nul
) else (
    echo WARNING: no shaders at %SHADERS% -- the GPU cells will fail at context creation
)

set UNEXPECTED=0
for %%f in (cell_status_ordinals cell_census_lifetime cell_hostile_capacity cell_prefill_faults_schema cell_final_norm_guard ^
            cell_prompt_guard_status cell_prompt_guard_removal cell_concurrency cell_determinism_composition cell_functional_commission ^
            cell_env_pins_shipping_leg cell_wrapper_census_standing) do (
    set VARIANT=gpu_fi
    set CELLINC=
    if "%%f"=="cell_env_pins_shipping_leg" set VARIANT=gpu_plain
    rem Q4-1e removes the process-wide harness device from d3d12_harness.h and needs no seam: shipping build.
    if "%%f"=="cell_prompt_guard_removal" set VARIANT=gpu_plain
    if "%%f"=="cell_prompt_guard_removal" set CELLINC=/I"%ENG%\src\gpu"
    echo ===== %%f =====
    cl /c %CLF% !CELLINC! /DSUPERSLM_T2169_CHUNK_RECORDING_FAULT_INJECTION /Fo"%OUT%\cells\%%f.obj" "%HEREDIR%%%f.cpp" > "%OUT%\cells\%%f.log" 2>&1
    if errorlevel 1 (
        echo    RED BY COMPILE:
        findstr /R /C:"error C[0-9]*" "%OUT%\cells\%%f.log"
    ) else (
        link /nologo /OUT:"%OUT%\bin\%%f.exe" "%OUT%\cells\%%f.obj" "%OUT%\cpu.lib" "%OUT%\!VARIANT!.lib" ^
            d3d12.lib dxgi.lib dxguid.lib >> "%OUT%\cells\%%f.log" 2>&1
        if errorlevel 1 (
            findstr /C:"LNK2019" "%OUT%\cells\%%f.log" >nul
            if not errorlevel 1 (
                echo    RED BY LINK -- unresolved:
                findstr /C:"LNK2019" "%OUT%\cells\%%f.log"
            ) else (
                echo    UNEXPECTED LINK FAILURE -- see %OUT%\cells\%%f.log
                set UNEXPECTED=1
            )
        ) else (
            echo    LINKED -- run it: %OUT%\bin\%%f.exe
        )
    )
)
echo.
if "%UNEXPECTED%"=="1" ( echo SUITE BUILD: unexpected failure class & exit /b 2 )
echo SUITE BUILD: done. A cell that did not link is red by the reason printed above.
exit /b 0

:mklib
(for %%o in ("%OUT%\%1\*.obj") do @echo "%%o") > "%OUT%\%1.rsp"
lib /nologo /OUT:"%OUT%\%1.lib" @"%OUT%\%1.rsp" > "%OUT%\%1.lib.log" 2>&1
if errorlevel 1 ( type "%OUT%\%1.lib.log" & echo LIB %1 FAILED & exit /b 3 )
exit /b 0
