@echo off
rem Build the S2.1 exhaustive certifier (SuperSLM_Plan.md §6.8 Popper boundary).
rem A one-time-per-device-tier proof run, not part of the CI suite.
setlocal
set VSDEVCMD="C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat"
if not exist %VSDEVCMD% (
	echo VsDevCmd.bat not found at %VSDEVCMD%
	exit /b 1
)
call %VSDEVCMD% -arch=x64 -no_logo
pushd %~dp0
if not exist out mkdir out
cl /nologo /std:c++20 /O2 /W4 /fp:precise /EHsc /Iinclude ^
	src\intmath.cpp tests\cert_intmath.cpp /Fo:out\ /Fe:out\cert_intmath.exe
if errorlevel 1 (popd & exit /b 1)
popd
exit /b 0
