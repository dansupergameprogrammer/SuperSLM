@echo off
rem Build the T-1691 primitive probe (test-support tool, not production code --
rem see tests\t1691_primitive_probe.cpp's own header). Adhoc tool build, MSVC;
rem mirrors tools\build_layer_trace.bat's source list and invocation shape,
rem substituting tests\t1691_primitive_probe.cpp and dropping the tokenizer/
rem artifact-loader sources this probe never calls.
setlocal
set VSDEVCMD="C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat"
if not exist %VSDEVCMD% (echo VsDevCmd.bat not found & exit /b 1)
call %VSDEVCMD% -arch=x64 -no_logo
pushd %~dp0\..
if not exist out mkdir out
cl /nologo /std:c++20 /O2 /W4 /fp:precise /EHsc /Iinclude ^
	src\artifact.cpp src\sha256.cpp src\tokenizer.cpp src\model.cpp src\intmath.cpp src\silu_lut.cpp src\matmul.cpp src\proof_manifest.cpp src\trace_hook.cpp src\decode_digest.cpp ^
	src\forward\checked_chain_funnel.cpp src\forward\forward_sites.cpp ^
	tests\t1691_primitive_probe.cpp /Fo:out\ /Fe:out\t1691_primitive_probe.exe
set ec=%errorlevel%
popd
exit /b %ec%
