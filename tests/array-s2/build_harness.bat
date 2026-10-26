@echo off
rem Builds the S2 array harness beside build.bat's own outputs (out\shaders must already hold the
rem compiled shaders, which build.bat writes): out\array_s2.exe, the production library, and
rem out\array_s2_seam.exe, the same sources with the schema fast-decode test seams compiled in,
rem used only to run the full-vocabulary finish for the masked-finish comparison.
setlocal
rem VSDEVCMD may be set by the caller (quoted path), for an install outside VS 2022 Community.
if not defined VSDEVCMD set VSDEVCMD="C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat"
call %VSDEVCMD% -arch=x64 -no_logo
pushd %~dp0..\..
set SRCS=src\artifact.cpp src\sha256.cpp src\tokenizer.cpp src\model.cpp src\intmath.cpp src\silu_lut.cpp src\matmul.cpp src\proof_manifest.cpp src\trace_hook.cpp ^
	src\forward\checked_chain_funnel.cpp src\forward\forward_sites.cpp src\forward\parallel_split.cpp src\gpu\gpu_inflight_registry.cpp src\decode_digest.cpp ^
	src\kv_pages.cpp src\spx1_format.cpp src\sslm_abi.cpp ^
	src\damped_greedy_antilm.cpp src\damped_greedy_topk.cpp src\damped_greedy_phaseD.cpp ^
	src\gpu\superslm_gpu.cpp src\gpu\gpu_1p0.cpp ^
	tests\array-s2\array_s2_main.cpp tests\array-s2\array_s2_cpu.cpp tests\array-s2\array_s2_gpu.cpp
if not exist out\array_s2 mkdir out\array_s2
if not exist out\array_s2_seam mkdir out\array_s2_seam
cl /nologo /std:c++20 /O2 /W4 /fp:precise /EHsc /Iinclude /Itests\array-s2 %SRCS% ^
	/Fo:out\array_s2\ /Fe:out\array_s2.exe /link d3d12.lib dxgi.lib dxguid.lib
if errorlevel 1 goto :fail
cl /nologo /std:c++20 /O2 /W4 /fp:precise /EHsc /Iinclude /Itests\array-s2 /Itests /DSUPERSLM_ENABLE_SCHEMA_FAST_DECODE_TEST_SEAMS %SRCS% ^
	/Fo:out\array_s2_seam\ /Fe:out\array_s2_seam.exe /link d3d12.lib dxgi.lib dxguid.lib
if errorlevel 1 goto :fail
popd
exit /b 0
:fail
popd
exit /b 1
