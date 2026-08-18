@echo off
rem DISPOSABLE build helper for t2132_dim8m2_diag (T-2132, Brunel, G5-6 mechanism-cell-2
rem investigation -- Claude/Curie/t2130-g5-red-suite-composition-joins-2026-08-17.md).
rem Not wired into build.bat -- run directly: tools\t2132_dim8m2_diag_build.bat
setlocal
set VSDEVCMD="C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat"
if not exist %VSDEVCMD% (
	echo VsDevCmd.bat not found at %VSDEVCMD%
	exit /b 1
)
call %VSDEVCMD% -arch=x64 -no_logo
pushd %~dp0\..
if not exist tools\obj mkdir tools\obj
cl /nologo /std:c++20 /O2 /W4 /fp:precise /EHsc /Iinclude ^
	src\artifact.cpp src\sha256.cpp src\tokenizer.cpp src\model.cpp src\intmath.cpp src\silu_lut.cpp src\matmul.cpp src\proof_manifest.cpp src\trace_hook.cpp ^
	src\forward\checked_chain_funnel.cpp src\forward\forward_sites.cpp src\decode_digest.cpp ^
	src\gpu\superslm_gpu.cpp ^
	src\sslm_abi.cpp ^
	tools\t2132_dim8m2_diag.cpp ^
	/Fo:tools\obj\ /Fe:tools\t2132_dim8m2_diag.exe ^
	/link d3d12.lib dxgi.lib dxguid.lib
set BUILD_ERR=%errorlevel%
popd
exit /b %BUILD_ERR%
