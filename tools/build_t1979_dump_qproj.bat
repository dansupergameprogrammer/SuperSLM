@echo off
rem Build the t1979_dump_qproj CPU-side dump tool with MSVC (adhoc tool build;
rem links the clean superslm core). Mirrors tools\build_generate.bat and
rem tools\build_layer_trace.bat exactly, substituting tools\t1979_dump_qproj.cpp.
rem
rem T-1987 (closing T-1983 review S-3): this script did not exist anywhere in
rem HEAD, so nothing reproduced the T-1979 spike's build from a fresh checkout.
setlocal
set VSDEVCMD="C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat"
if not exist %VSDEVCMD% (echo VsDevCmd.bat not found & exit /b 1)
call %VSDEVCMD% -arch=x64 -no_logo
pushd %~dp0\..
if not exist out mkdir out
cl /nologo /std:c++20 /O2 /W4 /fp:precise /EHsc /Iinclude ^
	src\artifact.cpp src\sha256.cpp src\tokenizer.cpp src\model.cpp src\intmath.cpp src\silu_lut.cpp src\matmul.cpp src\proof_manifest.cpp src\trace_hook.cpp ^
	src\forward\checked_chain_funnel.cpp src\forward\forward_sites.cpp src\decode_digest.cpp ^
	tools\t1979_dump_qproj.cpp /Fo:out\ /Fe:out\t1979_dump_qproj.exe
set ec=%errorlevel%
popd
exit /b %ec%
