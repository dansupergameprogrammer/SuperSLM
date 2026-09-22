@echo off
rem T-2948 round 2. Build one added no-GPU cell at a time: f1, path, or host.
setlocal
set "MODE=%~1"
if not "%MODE%"=="f1" if not "%MODE%"=="path" if not "%MODE%"=="host" (
    echo Usage: build_r2_red.bat f1^|path^|host
    exit /b 2
)
set "HERE=%~dp0"
for %%I in ("%HERE%..\..") do set "ENG=%%~fI"
call "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 -no_logo
cd /d "%HERE%"
if not exist obj\r2 mkdir obj\r2
if "%MODE%"=="f1" set "CELL=cell_shader_stale_unicode.cpp"
if "%MODE%"=="path" set "CELL=cell_shader_dir.cpp"
if "%MODE%"=="host" set "CELL=%ENG%\tests\test_main.cpp"
if "%MODE%"=="f1" set "EXE=cell_shader_stale_unicode.exe"
if "%MODE%"=="path" set "EXE=cell_shader_dir.exe"
if "%MODE%"=="host" set "EXE=superslm_tests.exe"
cl /nologo /std:c++20 /O2 /W4 /fp:precise /EHsc /DSUPERSLM_ENABLE_BAD_ALLOC_INJECTION /DSUPERSLM_O11_ALLOC_INJECTION /I"%ENG%\include" /I"%ENG%\tests" /I. ^
    "%ENG%\src\artifact.cpp" "%ENG%\src\sha256.cpp" "%ENG%\src\tokenizer.cpp" "%ENG%\src\model.cpp" ^
    "%ENG%\src\intmath.cpp" "%ENG%\src\silu_lut.cpp" "%ENG%\src\matmul.cpp" "%ENG%\src\proof_manifest.cpp" ^
    "%ENG%\src\trace_hook.cpp" "%ENG%\src\forward\checked_chain_funnel.cpp" ^
    "%ENG%\src\forward\forward_sites.cpp" "%ENG%\src\decode_digest.cpp" ^
    "%ENG%\src\gpu\superslm_gpu.cpp" "%ENG%\src\gpu\gpu_1p0.cpp" "%CELL%" ^
    /Fo:"obj\r2\\" /Fe:"obj\r2\%EXE%" /link d3d12.lib dxgi.lib dxguid.lib >"obj\r2\%MODE%.build.log" 2>&1
if errorlevel 1 (type "obj\r2\%MODE%.build.log" & exit /b 2)
findstr /C:"error C" /C:"fatal error" /C:"LNK1120" "obj\r2\%MODE%.build.log" >nul
if not errorlevel 1 (type "obj\r2\%MODE%.build.log" & exit /b 2)
echo T-2948 round 2 %MODE% linked clean: obj\r2\%EXE%
exit /b 0
