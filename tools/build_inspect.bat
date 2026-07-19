@echo off
rem Build the sslm_inspect debug/cross-check tool with MSVC.
setlocal
set VSDEVCMD="C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat"
if not exist %VSDEVCMD% (echo VsDevCmd.bat not found & exit /b 1)
call %VSDEVCMD% -arch=x64 -no_logo
pushd %~dp0\..
if not exist out mkdir out
cl /nologo /std:c++20 /O2 /W4 /fp:precise /EHsc /Iinclude ^
	src\artifact.cpp src\sha256.cpp tools\sslm_inspect.cpp ^
	/Fo:out\ /Fe:out\sslm_inspect.exe
if errorlevel 1 (popd & exit /b 1)
cl /nologo /std:c++20 /O2 /W4 /fp:precise /EHsc /Iinclude ^
	src\artifact.cpp src\sha256.cpp src\tokenizer.cpp tools\tok_verify.cpp ^
	/Fo:out\ /Fe:out\tok_verify.exe
set ec=%errorlevel%
popd
exit /b %ec%
