@echo off
rem Schema fast-decode plan rev 1.4.4, slices G1-F, G1-M and G1-J: the GPU cells' box build (MSVC). It builds the
rem same executables as build_sfd_gpu_mingw.sh (the cloud build); the two tables must stay in step.
rem
rem Usage: tests\schema-fast-decode-gpu\build_sfd_gpu.bat ["--out=DIR"] ["--shaders=DIR"] [TARGET ...]
rem   TARGET   g1f | g1m | g1j (default: all three), or one executable's name
rem   --out    default <repo>\build\sfdg (executables in <out>\bin, shaders copied to <out>\bin\shaders)
rem   --shaders default <repo>\out\shaders (build.bat's), else <repo>\build\Release\shaders (CMake's)
rem
rem Engine: build.bat's source list restricted to src\, compiled once with the test seams
rem (/DSUPERSLM_GPU_TEST_SEAMS /DSUPERSLM_GPU_ALLOC_FAULT_INJECTION /DSUPERSLM_ENABLE_BAD_ALLOC_INJECTION /DSUPERSLM_GPU_G5_FINISH_ROW_FAULT_INJECTION).
rem Each executable is reported LINKED, RED BY LINK or RED BY COMPILE. Exit 0 unless the engine failed to build.
rem The box driver (tools\schema_fast_decode_box\run_box_step.ps1, steps GB-F, GB-M, GB-J, GB1) runs them with
rem SSLM_GPU_ENABLE_DEBUG_LAYER=2; the cells never set it (G0.1's children excepted).
setlocal enabledelayedexpansion
set HEREDIR=%~dp0
for %%I in ("%HEREDIR%..\..") do set ENG=%%~fI
set OUT=%ENG%\build\sfdg
set SHADERS=
set TARGETS=
:parse_args
if "%~1"=="" goto :args_done
set "ARG=%~1"
set "TAKEN="
if "!ARG:~0,6!"=="--out=" ( set "OUT=!ARG:~6!" & set "TAKEN=1" )
if "!ARG:~0,10!"=="--shaders=" ( set "SHADERS=!ARG:~10!" & set "TAKEN=1" )
if not defined TAKEN set "TARGETS=!TARGETS! %~1"
shift
goto :parse_args
:args_done
if "%TARGETS%"=="" set TARGETS= g1f g1m g1j
if "%SHADERS%"=="" set SHADERS=%ENG%\out\shaders
if not exist "%SHADERS%\*.cso" set SHADERS=%ENG%\build\Release\shaders
call "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 -no_logo
if exist "%OUT%\cells" rmdir /S /Q "%OUT%\cells"
for %%D in (engine cells bin) do if not exist "%OUT%\%%D" mkdir "%OUT%\%%D"
set T2791=%ENG%\tests\t2791-gpu-prefill-read-red-suite
set CLF=/nologo /std:c++20 /O2 /W4 /fp:precise /EHsc /I"%ENG%\include" /I"%ENG%\src" /I"%ENG%\src\gpu" /I"%ENG%\tests" ^
 /I"%ENG%\tests\paged-kv" /I"%ENG%\tests\paged-kv-gpu" /I"%T2791%" /I"%HEREDIR%." ^
 /DSUPERSLM_GPU_TEST_SEAMS /DSUPERSLM_GPU_ALLOC_FAULT_INJECTION /DSUPERSLM_ENABLE_BAD_ALLOC_INJECTION /DSUPERSLM_GPU_G5_FINISH_ROW_FAULT_INJECTION
set NLINKED=0
set NRED=0
set NCOMPILE=0

echo ===== engine: build.bat's sources, test-seam configuration =====
cl /c %CLF% /Fo"%OUT%\engine\\" "%ENG%\src\artifact.cpp" "%ENG%\src\sha256.cpp" "%ENG%\src\tokenizer.cpp" ^
    "%ENG%\src\model.cpp" "%ENG%\src\intmath.cpp" "%ENG%\src\silu_lut.cpp" "%ENG%\src\matmul.cpp" ^
    "%ENG%\src\proof_manifest.cpp" "%ENG%\src\trace_hook.cpp" "%ENG%\src\forward\checked_chain_funnel.cpp" ^
    "%ENG%\src\forward\forward_sites.cpp" "%ENG%\src\forward\parallel_split.cpp" "%ENG%\src\gpu\gpu_inflight_registry.cpp" ^
    "%ENG%\src\decode_digest.cpp" "%ENG%\src\kv_pages.cpp" "%ENG%\src\spx1_format.cpp" ^
    "%ENG%\src\sslm_abi.cpp" "%ENG%\src\damped_greedy_antilm.cpp" "%ENG%\src\damped_greedy_topk.cpp" ^
    "%ENG%\src\damped_greedy_phaseD.cpp" "%ENG%\src\damped_greedy_phaseD_loop.cpp" ^
    "%ENG%\src\gpu\superslm_gpu.cpp" "%ENG%\src\gpu\gpu_1p0.cpp" > "%OUT%\engine\build.log" 2>&1
if errorlevel 1 ( type "%OUT%\engine\build.log" & echo ENGINE BUILD FAILED & exit /b 3 )
lib /nologo /OUT:"%OUT%\engine.lib" "%OUT%\engine\*.obj" > "%OUT%\engine\lib.log" 2>&1
if errorlevel 1 ( type "%OUT%\engine\lib.log" & exit /b 3 )

if exist "%SHADERS%\*.cso" (
    if not exist "%OUT%\bin\shaders" mkdir "%OUT%\bin\shaders"
    xcopy /Y /Q "%SHADERS%\*.cso" "%OUT%\bin\shaders\" >nul
) else (
    echo WARNING: no shaders at %SHADERS% -- run build.bat first; the GPU cells will fail at context creation
)

rem ---- the executables (keep in step with build_sfd_gpu_mingw.sh's TABLE) ---------------------------------------
call :exe sfdg_smoke G1F
call :exe sfdg_cell_g0_1 G1F
call :exe sfdg_cell_g2_1 G1F
call :exe sfdg_cell_g12_4 G1F
call :exe sfdg_cell_g12_5 G1F
call :exe sfdg_cell_g5_1 G1M
call :exe sfdg_cell_g6 G1M
call :exe sfdg_cell_g8_1 G1M
call :exe sfdg_cell_g7_1 G1J
call :exe sfdg_cell_g9_1 G1J
call :exe sfdg_cell_g11_1 G1J
call :exe sfdg_cell_g12_verb G1J
call :exe sfdg_cell_g12_6 G1J
call :exe sfdg_cell_g12_7 G1J
call :exe sfdg_cell_g12_10 G1J
call :exe schema_fast_decode_gpu_p2_ref G1J "%ENG%\tools\"
echo SFDG BUILD: %NLINKED% linked, %NRED% red by link, %NCOMPILE% red by compile
exit /b 0

:exe
set EXE=%~1
set SLICE=%~2
set SRCDIR=%~3
if "%SRCDIR%"=="" set SRCDIR=%HEREDIR%
set WANT=0
for %%T in (%TARGETS%) do (
    if /I "%%T"=="%EXE%" set WANT=1
    if /I "%%T"=="%SLICE%" set WANT=1
    if /I "%%T"=="g1f" if "%SLICE%"=="G1F" set WANT=1
    if /I "%%T"=="g1m" if "%SLICE%"=="G1M" set WANT=1
    if /I "%%T"=="g1j" if "%SLICE%"=="G1J" set WANT=1
)
if "%WANT%"=="0" exit /b 0
if not exist "%SRCDIR%%EXE%.cpp" ( echo [%SLICE%] %EXE%: NOT WRITTEN & exit /b 0 )
cl /c %CLF% /Fo"%OUT%\cells\%EXE%.obj" "%SRCDIR%%EXE%.cpp" > "%OUT%\cells\%EXE%.log" 2>&1
if errorlevel 1 ( type "%OUT%\cells\%EXE%.log" & echo [%SLICE%] %EXE%: RED BY COMPILE & set /a NCOMPILE+=1 & exit /b 0 )
link /nologo /OUT:"%OUT%\bin\%EXE%.exe" "%OUT%\cells\%EXE%.obj" "%OUT%\engine.lib" ^
    d3d12.lib dxgi.lib dxguid.lib shlwapi.lib psapi.lib > "%OUT%\cells\%EXE%.link.log" 2>&1
if not errorlevel 1 ( echo [%SLICE%] %EXE%: LINKED & set /a NLINKED+=1 & exit /b 0 )
if exist "%OUT%\bin\%EXE%.exe" del "%OUT%\bin\%EXE%.exe"
echo [%SLICE%] %EXE%: RED BY LINK -- unresolved:
findstr /C:"LNK2019" /C:"LNK2001" "%OUT%\cells\%EXE%.link.log"
set /a NRED+=1
exit /b 0
