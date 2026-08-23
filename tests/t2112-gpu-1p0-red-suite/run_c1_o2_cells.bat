@echo off
rem T-2243 (Brunel): builds and runs dim9_persistence_red.cpp (carries the C1 cell) and
rem dim11_guard_red.cpp (carries the O2/M2 cells) against a real 1.5B artifact, on this
rem machine's real D3D12 hardware. Ad hoc build/run script for this session's own execution
rem evidence -- not part of the canonical build.bat pipeline (which runs these suites without
rem real-artifact argv, per its own existing convention).
setlocal enabledelayedexpansion
set HEREDIR=%~dp0
set ENG=%HEREDIR%..\..
call "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 -no_logo
cd /d "%HEREDIR%"
if not exist obj mkdir obj
if exist ..\..\out\shaders xcopy /Y /I /Q ..\..\out\shaders obj\shaders >nul

set MODEL=D:\SuperSLM\.worktrees\run\out\t1942_capture\sslm\qwen2.5-1.5b-instruct-armc-fixed.sslm

echo ===== building dim9_persistence_red.cpp (C1 cell) =====
cl /nologo /std:c++20 /O2 /W4 /fp:precise /EHsc /DSUPERSLM_ENABLE_BAD_ALLOC_INJECTION /DSUPERSLM_O11_ALLOC_INJECTION /I%ENG%\include /I%ENG%\tests /I. ^
    %ENG%\src\artifact.cpp %ENG%\src\sha256.cpp %ENG%\src\tokenizer.cpp %ENG%\src\model.cpp ^
    %ENG%\src\intmath.cpp %ENG%\src\silu_lut.cpp %ENG%\src\matmul.cpp %ENG%\src\proof_manifest.cpp ^
    %ENG%\src\trace_hook.cpp %ENG%\src\forward\checked_chain_funnel.cpp ^
    %ENG%\src\forward\forward_sites.cpp %ENG%\src\decode_digest.cpp %ENG%\src\gpu\superslm_gpu.cpp ^
    %ENG%\src\gpu\gpu_1p0.cpp ^
    dim9_persistence_red.cpp /Fo:"obj\\" /Fe:"obj\dim9_c1.exe" ^
    /link d3d12.lib dxgi.lib dxguid.lib
if errorlevel 1 (
    echo BUILD FAILED: dim9_persistence_red.cpp
    exit /b 1
)
echo ===== running dim9_persistence_red.exe (real artifact, C1 cell included) =====
obj\dim9_c1.exe --model1p5b=%MODEL%
echo dim9 exit: %errorlevel%

echo ===== building dim11_guard_red.cpp (O2/M2 cells) =====
cl /nologo /std:c++20 /O2 /W4 /fp:precise /EHsc /DSUPERSLM_ENABLE_BAD_ALLOC_INJECTION /DSUPERSLM_O11_ALLOC_INJECTION /I%ENG%\include /I%ENG%\tests /I. ^
    %ENG%\src\artifact.cpp %ENG%\src\sha256.cpp %ENG%\src\tokenizer.cpp %ENG%\src\model.cpp ^
    %ENG%\src\intmath.cpp %ENG%\src\silu_lut.cpp %ENG%\src\matmul.cpp %ENG%\src\proof_manifest.cpp ^
    %ENG%\src\trace_hook.cpp %ENG%\src\forward\checked_chain_funnel.cpp ^
    %ENG%\src\forward\forward_sites.cpp %ENG%\src\decode_digest.cpp %ENG%\src\gpu\superslm_gpu.cpp ^
    %ENG%\src\gpu\gpu_1p0.cpp ^
    dim11_guard_red.cpp /Fo:"obj\\" /Fe:"obj\dim11_o2m2.exe" ^
    /link d3d12.lib dxgi.lib dxguid.lib
if errorlevel 1 (
    echo BUILD FAILED: dim11_guard_red.cpp
    exit /b 1
)
echo ===== running dim11_guard_red.exe (real artifact, O2/M2 cells included) =====
obj\dim11_o2m2.exe --model1p5b=%MODEL%
echo dim11 exit: %errorlevel%
exit /b 0
