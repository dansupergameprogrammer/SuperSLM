@echo off
setlocal
set HEREDIR=%~dp0
set ENG=%HEREDIR%..\..
call "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 -no_logo
cd /d "%HEREDIR%"
if not exist obj mkdir obj
cl /nologo /std:c++20 /O2 /W4 /fp:precise /EHsc /I%ENG%\include ^
    %ENG%\src\artifact.cpp %ENG%\src\sha256.cpp %ENG%\src\tokenizer.cpp %ENG%\src\model.cpp ^
    %ENG%\src\intmath.cpp %ENG%\src\silu_lut.cpp %ENG%\src\matmul.cpp %ENG%\src\proof_manifest.cpp ^
    %ENG%\src\trace_hook.cpp %ENG%\src\forward\checked_chain_funnel.cpp ^
    %ENG%\src\forward\forward_sites.cpp %ENG%\src\decode_digest.cpp %ENG%\src\gpu\superslm_gpu.cpp ^
    dispatch_geometry_policy_red.cpp /Fo:"obj\\" /Fe:"obj\dispatch_geometry_policy_red.exe" ^
    /link d3d12.lib dxgi.lib dxguid.lib
if errorlevel 1 (
    echo BUILD FAILED
    exit /b 1
)
obj\dispatch_geometry_policy_red.exe
if errorlevel 1 exit /b 1

rem T-2240/O3 (SuperSLM 1.2.1): the ADAPTER_U region-rounding cell -- same host-only,
rem no-device shape as the policy pin above; links the real superslm_gpu.cpp so the cell's
rem subject (AdapterURegionBytes, gpu_port.h) and its one consumer (the work_total site)
 rem are the production definitions.
cl /nologo /std:c++20 /O2 /W4 /fp:precise /EHsc /I%ENG%\include ^
    %ENG%\src\artifact.cpp %ENG%\src\sha256.cpp %ENG%\src\tokenizer.cpp %ENG%\src\model.cpp ^
    %ENG%\src\intmath.cpp %ENG%\src\silu_lut.cpp %ENG%\src\matmul.cpp %ENG%\src\proof_manifest.cpp ^
    %ENG%\src\trace_hook.cpp %ENG%\src\forward\checked_chain_funnel.cpp ^
    %ENG%\src\forward\forward_sites.cpp %ENG%\src\decode_digest.cpp %ENG%\src\gpu\superslm_gpu.cpp ^
    adapter_u_region_red.cpp /Fo:"obj\\" /Fe:"obj\adapter_u_region_red.exe" ^
    /link d3d12.lib dxgi.lib dxguid.lib
if errorlevel 1 (
    echo BUILD FAILED ^(adapter_u_region_red^)
    exit /b 1
)
obj\adapter_u_region_red.exe
exit /b %errorlevel%
