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
exit /b %errorlevel%
