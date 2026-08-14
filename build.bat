@echo off
rem Quick MSVC build + test. For other compilers / the full matrix use CMake.
setlocal
set VSDEVCMD="C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat"
if not exist %VSDEVCMD% (
	echo VsDevCmd.bat not found at %VSDEVCMD%
	exit /b 1
)
call %VSDEVCMD% -arch=x64 -no_logo
pushd %~dp0
if not exist out mkdir out
if not exist out\shaders mkdir out\shaders

rem T-1986 GPU-serial port (Sec5.7): every dxc invocation this design's build
rem issues adds -WX, at the pinned compile target (cs_6_2 -HV 2018 -O3).
set DXC="C:\Program Files (x86)\Windows Kits\10\bin\10.0.19041.0\x64\dxc.exe"
if not exist %DXC% (
	echo dxc.exe not found at %DXC%
	popd & exit /b 1
)
for %%f in (src\gpu\shaders\*.hlsl) do (
	%DXC% -T cs_6_2 -E main -Fo out\shaders\%%~nf.cso %%f -O3 -HV 2018 -WX
	if errorlevel 1 (
		popd & exit /b 1
	)
)

cl /nologo /std:c++20 /O2 /W4 /fp:precise /EHsc /Iinclude /Itests /DSUPERSLM_ENABLE_BAD_ALLOC_INJECTION ^
	src\artifact.cpp src\sha256.cpp src\tokenizer.cpp src\model.cpp src\intmath.cpp src\silu_lut.cpp src\matmul.cpp src\proof_manifest.cpp src\trace_hook.cpp ^
	src\forward\checked_chain_funnel.cpp src\forward\forward_sites.cpp src\decode_digest.cpp ^
	src\gpu\superslm_gpu.cpp ^
	tests\test_main.cpp /Fo:out\ /Fe:out\superslm_tests.exe ^
	/link d3d12.lib dxgi.lib dxguid.lib
if errorlevel 1 (
	popd & exit /b 1
)
out\superslm_tests.exe
set ec=%errorlevel%
popd
exit /b %ec%
