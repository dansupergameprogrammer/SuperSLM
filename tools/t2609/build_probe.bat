@echo off
setlocal
set "SSLM_T2609_PRE_VS_PATH=%PATH%"
set "PATH="
set "Path="
set "PATH=%SSLM_T2609_PRE_VS_PATH%"
set "SSLM_T2609_PRE_VS_PATH="
call "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 -no_logo
if errorlevel 1 exit /b 1
if not exist "%~dp0\..\..\out\t2609" mkdir "%~dp0\..\..\out\t2609"
cl /nologo /std:c++20 /EHsc /MD /W4 /fp:precise /I"%~dp0\..\..\include" /I"%~dp0\.." ^
  "%~dp0cpp_trace.cpp" "%~dp0\..\..\out\t2604-build\superslm.lib" ^
  /Fe:"%~dp0\..\..\out\t2609\sslm_t2609_trace.exe"
exit /b %errorlevel%
