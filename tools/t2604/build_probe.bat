@echo off
setlocal
set "SSLM_T2604_PRE_VS_PATH=%PATH%"
set "PATH="
set "Path="
set "PATH=%SSLM_T2604_PRE_VS_PATH%"
set "SSLM_T2604_PRE_VS_PATH="
call "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 -no_logo
if errorlevel 1 exit /b 1
cmake -S "%~dp0\..\.." -B "%~dp0\..\..\out\t2604-build" -G Ninja -DCMAKE_BUILD_TYPE=Release -DSUPERSLM_BUILD_TESTS=OFF -DSUPERSLM_BUILD_GPU=OFF
if errorlevel 1 exit /b 1
cmake --build "%~dp0\..\..\out\t2604-build" --target sslm_t2604_trace
if errorlevel 1 exit /b 1
if /I "%~1"=="test" (
	cmake --build "%~dp0\..\..\out\t2604-build" --target superslm_tests
	if errorlevel 1 exit /b 1
	ctest --test-dir "%~dp0\..\..\out\t2604-build" --output-on-failure
)
exit /b %errorlevel%
