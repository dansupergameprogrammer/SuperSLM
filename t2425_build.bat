@echo off
setlocal
set VSDEVCMD="C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat"
call %VSDEVCMD% -arch=x64 -no_logo
pushd %~dp0
if not exist out\t2425 mkdir out\t2425

echo === building sslm_verify.exe ===
cl /nologo /std:c++20 /O2 /W4 /fp:precise /EHsc /Iinclude ^
	src\artifact.cpp src\sha256.cpp src\tokenizer.cpp src\model.cpp src\intmath.cpp src\silu_lut.cpp src\matmul.cpp src\proof_manifest.cpp src\trace_hook.cpp ^
	src\forward\checked_chain_funnel.cpp src\forward\forward_sites.cpp src\decode_digest.cpp ^
	src\sslm_abi.cpp ^
	src\damped_greedy_antilm.cpp src\damped_greedy_topk.cpp src\damped_greedy_phaseD.cpp src\damped_greedy_phaseD_loop.cpp ^
	tools\sslm_verify.cpp /Fo:out\t2425\ /Fe:out\t2425\sslm_verify.exe
if errorlevel 1 (
	echo sslm_verify BUILD FAILED
	popd & exit /b 1
)

echo === building t2425_trackbd_harness.exe ===
cl /nologo /std:c++20 /O2 /W4 /fp:precise /EHsc /Iinclude ^
	src\artifact.cpp src\sha256.cpp src\tokenizer.cpp src\model.cpp src\intmath.cpp src\silu_lut.cpp src\matmul.cpp src\proof_manifest.cpp src\trace_hook.cpp ^
	src\forward\checked_chain_funnel.cpp src\forward\forward_sites.cpp src\decode_digest.cpp ^
	src\sslm_abi.cpp ^
	src\damped_greedy_antilm.cpp src\damped_greedy_topk.cpp src\damped_greedy_phaseD.cpp src\damped_greedy_phaseD_loop.cpp ^
	tools\t2425_trackbd_harness.cpp /Fo:out\t2425\ /Fe:out\t2425\t2425_trackbd_harness.exe
if errorlevel 1 (
	echo t2425_trackbd_harness BUILD FAILED
	popd & exit /b 1
)

echo === building t2425_marshal_diag.exe ===
cl /nologo /std:c++20 /O2 /W4 /fp:precise /EHsc /Iinclude ^
	src\artifact.cpp src\sha256.cpp src\tokenizer.cpp src\model.cpp src\intmath.cpp src\silu_lut.cpp src\matmul.cpp src\proof_manifest.cpp src\trace_hook.cpp ^
	src\forward\checked_chain_funnel.cpp src\forward\forward_sites.cpp src\decode_digest.cpp ^
	tools\t2425_marshal_diag.cpp /Fo:out\t2425\ /Fe:out\t2425\t2425_marshal_diag.exe
if errorlevel 1 (
	echo t2425_marshal_diag BUILD FAILED
	popd & exit /b 1
)

echo === BUILD OK ===
popd
exit /b 0
