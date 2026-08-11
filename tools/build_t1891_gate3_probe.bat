@echo off
rem T-1891 gate G3 probe build (disposable spike tool; adhoc build, mirrors
rem build_generate.bat's own recipe). Iinclude+Itests needed for the fixture headers.
setlocal
set VSDEVCMD="C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat"
if not exist %VSDEVCMD% (echo VsDevCmd.bat not found & exit /b 1)
call %VSDEVCMD% -arch=x64 -no_logo
pushd %~dp0\..
if not exist out mkdir out
cl /nologo /std:c++20 /O2 /W4 /fp:precise /EHsc /Iinclude /Itests ^
	src\artifact.cpp src\sha256.cpp src\tokenizer.cpp src\model.cpp src\intmath.cpp src\silu_lut.cpp src\matmul.cpp src\proof_manifest.cpp src\trace_hook.cpp ^
	src\forward\checked_chain_funnel.cpp src\forward\forward_sites.cpp src\decode_digest.cpp ^
	tools\t1891_optiong_gate3_probe.cpp /Fo:out\ /Fe:out\t1891_gate3_probe.exe
set ec=%errorlevel%
popd
exit /b %ec%
