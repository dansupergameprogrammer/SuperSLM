@echo off
rem DISPOSABLE build helper for t2132_diag_layer_bisect (T-2132, Brunel diagnosis session).
rem Not wired into build.bat -- run directly: tools\t2132_diag_build.bat
setlocal
set VSDEVCMD="C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat"
if not exist %VSDEVCMD% (
	echo VsDevCmd.bat not found at %VSDEVCMD%
	exit /b 1
)
call %VSDEVCMD% -arch=x64 -no_logo
pushd %~dp0\..
if not exist out\t2132diag mkdir out\t2132diag
cl /nologo /std:c++20 /O2 /W4 /fp:precise /EHsc /Iinclude /Itools ^
	src\artifact.cpp src\sha256.cpp src\tokenizer.cpp src\model.cpp src\intmath.cpp src\silu_lut.cpp src\matmul.cpp src\proof_manifest.cpp src\trace_hook.cpp ^
	src\forward\checked_chain_funnel.cpp src\forward\forward_sites.cpp src\decode_digest.cpp ^
	src\sslm_abi.cpp ^
	src\gpu\superslm_gpu.cpp src\gpu\gpu_1p0.cpp ^
	tools\t2132_diag_layer_bisect.cpp tools\t2132_diag_layer_bisect_cpu.cpp tools\t2132_diag_layer_bisect_gpu.cpp ^
	/Fo:out\t2132diag\ /Fe:out\t2132_diag_layer_bisect.exe ^
	/link d3d12.lib dxgi.lib dxguid.lib
set BUILD_ERR=%errorlevel%
popd
exit /b %BUILD_ERR%
