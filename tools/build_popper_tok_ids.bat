@echo off
setlocal
set VSDEVCMD="C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat"
call %VSDEVCMD% -arch=x64 -no_logo
pushd %~dp0\..
if not exist out mkdir out
cl /nologo /std:c++20 /O2 /W4 /fp:precise /EHsc /Iinclude ^
	src\artifact.cpp src\sha256.cpp src\tokenizer.cpp ^
	tools\popper_tok_ids.cpp /Fo:out\ /Fe:out\popper_tok_ids.exe
set ec=%errorlevel%
popd
exit /b %ec%
