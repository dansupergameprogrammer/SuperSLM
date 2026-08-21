@echo off
setlocal
set VSDEVCMD="C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat"
if not exist %VSDEVCMD% (echo VsDevCmd.bat not found & exit /b 1)
call %VSDEVCMD% -arch=x64 -no_logo
pushd %~dp0\..
if not exist out mkdir out
cl /nologo /std:c++20 /O2 /W4 /fp:precise /EHsc /Iinclude ^
	src\intmath.cpp src\matmul.cpp src\trace_hook.cpp src\forward\checked_chain_funnel.cpp ^
	src\damped_greedy_antilm.cpp src\damped_greedy_topk.cpp ^
	tools\t2199_b0_probe.cpp /Fo:out\ /Fe:out\t2199_b0_probe.exe
set ec=%errorlevel%
popd
exit /b %ec%
