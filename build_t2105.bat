@echo off
rem T-2105 (Laplace, disposable): shaders + the throughput instrument only.
rem Deliberately does NOT build the test suite -- this is an experiment loop,
rem not a ship gate. Correctness here is the throughput harness's own per-step
rem CPU/GPU bit-equality plus the C5 harness, both of which this builds.
setlocal
set VSDEVCMD="C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat"
call %VSDEVCMD% -arch=x64 -no_logo
pushd %~dp0
if not exist out mkdir out
if not exist out\shaders mkdir out\shaders
if not exist out\thr mkdir out\thr

set DXC="C:\Program Files (x86)\Windows Kits\10\bin\10.0.19041.0\x64\dxc.exe"
setlocal enabledelayedexpansion
set SHADERFAIL=0
for %%f in (src\gpu\shaders\*.hlsl) do (
	%DXC% -T cs_6_2 -E main -Fo out\shaders\%%~nf.cso %%f -O3 -HV 2018 -WX
	if errorlevel 1 (
		echo SHADER FAILED: %%f
		set SHADERFAIL=1
	)
)
if !SHADERFAIL! NEQ 0 ( popd & exit /b 1 )

cl /nologo /std:c++20 /O2 /W4 /fp:precise /EHsc /Iinclude /Itests /Itools /DSUPERSLM_ENABLE_BAD_ALLOC_INJECTION ^
	src\artifact.cpp src\sha256.cpp src\tokenizer.cpp src\model.cpp src\intmath.cpp src\silu_lut.cpp src\matmul.cpp src\proof_manifest.cpp src\trace_hook.cpp ^
	src\forward\checked_chain_funnel.cpp src\forward\forward_sites.cpp src\decode_digest.cpp ^
	src\gpu\superslm_gpu.cpp ^
	tools\t2100_gpu_throughput.cpp /Fo:out\thr\ /Fe:out\t2100_gpu_throughput.exe ^
	/link d3d12.lib dxgi.lib dxguid.lib
if errorlevel 1 ( popd & exit /b 1 )

if not exist out\c5 mkdir out\c5
cl /nologo /std:c++20 /O2 /W4 /fp:precise /EHsc /Iinclude /Itests /Itools /DSUPERSLM_ENABLE_BAD_ALLOC_INJECTION ^
	src\artifact.cpp src\sha256.cpp src\tokenizer.cpp src\model.cpp src\intmath.cpp src\silu_lut.cpp src\matmul.cpp src\proof_manifest.cpp src\trace_hook.cpp ^
	src\forward\checked_chain_funnel.cpp src\forward\forward_sites.cpp src\decode_digest.cpp ^
	src\gpu\superslm_gpu.cpp ^
	tools\t2039_c5_harness.cpp /Fo:out\c5\ /Fe:out\t2039_c5_harness.exe ^
	/link d3d12.lib dxgi.lib dxguid.lib
if errorlevel 1 ( popd & exit /b 1 )

echo T2105_BUILD_OK
popd
exit /b 0
