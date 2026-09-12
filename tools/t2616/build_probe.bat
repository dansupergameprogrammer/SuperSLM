@echo off
setlocal
set "SSLM_T2616_PRE_VS_PATH=%PATH%"
set "PATH="
set "Path="
set "PATH=%SSLM_T2616_PRE_VS_PATH%"
set "SSLM_T2616_PRE_VS_PATH="
call "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 -no_logo
if errorlevel 1 exit /b 1
if not exist "%~dp0\..\..\out\t2616" mkdir "%~dp0\..\..\out\t2616"
cl /nologo /std:c++20 /EHsc /MD /W4 /fp:precise /I"%~dp0\..\..\include" /I"%~dp0\.." ^
  "%~dp0cpp_layer_probe.cpp" "%~dp0\..\..\out\t2604-build\superslm.lib" ^
  /Fe:"%~dp0\..\..\out\t2616\sslm_t2616_probe.exe"
exit /b %errorlevel%
