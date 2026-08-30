@echo off
setlocal
set VSDEVCMD="C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat"
call %VSDEVCMD% -arch=x64 -no_logo
pushd %~dp0
if not exist out\t2423 mkdir out\t2423

echo === building sslm_verify.exe ===
cl /nologo /std:c++20 /O2 /W4 /fp:precise /EHsc /Iinclude ^
	src\artifact.cpp src\sha256.cpp src\tokenizer.cpp src\model.cpp src\intmath.cpp src\silu_lut.cpp src\matmul.cpp src\proof_manifest.cpp src\trace_hook.cpp ^
	src\forward\checked_chain_funnel.cpp src\forward\forward_sites.cpp src\decode_digest.cpp ^
	src\sslm_abi.cpp ^
	src\damped_greedy_antilm.cpp src\damped_greedy_topk.cpp src\damped_greedy_phaseD.cpp src\damped_greedy_phaseD_loop.cpp ^
	tools\sslm_verify.cpp /Fo:out\t2423\ /Fe:out\t2423\sslm_verify.exe
if errorlevel 1 (
	echo sslm_verify BUILD FAILED
	popd & exit /b 1
)

echo === building t2423_model_map_harness.exe ===
cl /nologo /std:c++20 /O2 /W4 /fp:precise /EHsc /Iinclude ^
	src\artifact.cpp src\sha256.cpp src\tokenizer.cpp src\model.cpp src\intmath.cpp src\silu_lut.cpp src\matmul.cpp src\proof_manifest.cpp src\trace_hook.cpp ^
	src\forward\checked_chain_funnel.cpp src\forward\forward_sites.cpp src\decode_digest.cpp ^
	src\sslm_abi.cpp ^
	src\damped_greedy_antilm.cpp src\damped_greedy_topk.cpp src\damped_greedy_phaseD.cpp src\damped_greedy_phaseD_loop.cpp ^
	tools\t2423_model_map_harness.cpp /Fo:out\t2423\ /Fe:out\t2423\t2423_model_map_harness.exe
if errorlevel 1 (
	echo t2423_model_map_harness BUILD FAILED
	popd & exit /b 1
)

echo === BUILD OK ===
popd
exit /b 0
