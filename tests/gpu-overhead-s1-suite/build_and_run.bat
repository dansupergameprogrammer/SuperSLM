@echo off
rem GPU per-token overhead plan rev 3.1, slice S1: builds and runs the real-artifact S1 cells
rem (cell_s1_real_model.cpp), against the same engine source list build.bat's superslm_tests line
rem compiles. Box-only: it needs the D3D12 device and the artifacts.
rem
rem   build_and_run.bat --model0p5b=PATH [--modelqk=PATH] [--v1p11-blob=PATH] [--steps=N]
rem       builds cell_s1_real_model.exe and runs it with the arguments given. Exit 0 = green,
rem       1 = a cell failed, 2 = the build failed.
rem   build_and_run.bat writer --model0p5b=PATH --out=PATH
rem       builds write_v1p11_blob.exe and runs it. Run this arm from a v1.11.0 tree (copy this
rem       directory into it first): its blob is cell 9.1's v1.11.0 reference.
rem
rem The engine build's compiled shaders (out\shaders) are copied beside the executables, since
rem ShaderPath resolves <exe_dir>\shaders\*.cso.
setlocal enabledelayedexpansion
set HEREDIR=%~dp0
set ENG=%HEREDIR%..\..
call "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 -no_logo
cd /d "%HEREDIR%"
if not exist obj mkdir obj
if exist ..\..\out\shaders xcopy /Y /I /Q ..\..\out\shaders obj\shaders >nul

set TARGET=cell_s1_real_model
if "%~1"=="writer" (
    set TARGET=write_v1p11_blob
    shift
)
set ARGS=
:collect
if "%~1"=="" goto :build
set ARGS=!ARGS! "%~1"
shift
goto :collect

:build
cl /nologo /std:c++20 /O2 /W4 /fp:precise /EHsc /I%ENG%\include /I%ENG%\tests /I. ^
    %ENG%\src\artifact.cpp %ENG%\src\sha256.cpp %ENG%\src\tokenizer.cpp %ENG%\src\model.cpp ^
    %ENG%\src\intmath.cpp %ENG%\src\silu_lut.cpp %ENG%\src\matmul.cpp %ENG%\src\proof_manifest.cpp ^
    %ENG%\src\trace_hook.cpp %ENG%\src\forward\checked_chain_funnel.cpp ^
    %ENG%\src\forward\forward_sites.cpp %ENG%\src\forward\parallel_split.cpp %ENG%\src\decode_digest.cpp ^
    %ENG%\src\gpu\superslm_gpu.cpp %ENG%\src\gpu\gpu_1p0.cpp ^
    %TARGET%.cpp /Fo:"obj\\" /Fe:"obj\%TARGET%.exe" ^
    /link d3d12.lib dxgi.lib dxguid.lib > "obj\%TARGET%.build.log" 2>&1
if errorlevel 1 (
    type "obj\%TARGET%.build.log"
    echo %TARGET%: BUILD FAILED
    exit /b 2
)
"obj\%TARGET%.exe" !ARGS!
if errorlevel 1 (
    echo %TARGET%: FAILED
    exit /b 1
)
echo %TARGET%: GREEN
exit /b 0
