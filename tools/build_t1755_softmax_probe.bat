@echo off
rem Build the t1755_softmax_probe tool with MSVC (adhoc tool build; links the clean superslm core).
rem Mirrors tools\build_layer_trace.bat exactly, substituting tools\t1755_softmax_probe.cpp.
setlocal
set VSDEVCMD="C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat"
if not exist %VSDEVCMD% (echo VsDevCmd.bat not found & exit /b 1)
call %VSDEVCMD% -arch=x64 -no_logo
pushd %~dp0\..
if not exist out mkdir out
cl /nologo /std:c++20 /O2 /W4 /fp:precise /EHsc /Iinclude ^
	src\artifact.cpp src\sha256.cpp src\tokenizer.cpp src\model.cpp src\intmath.cpp src\silu_lut.cpp src\matmul.cpp src\proof_manifest.cpp src\trace_hook.cpp ^
	src\forward\checked_chain_funnel.cpp src\forward\forward_sites.cpp src\decode_digest.cpp ^
	tools\t1755_softmax_probe.cpp /Fo:out\ /Fe:out\t1755_softmax_probe.exe
set ec=%errorlevel%
popd
exit /b %ec%
