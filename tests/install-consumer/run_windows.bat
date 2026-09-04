@echo off
setlocal EnableExtensions EnableDelayedExpansion

rem MSBuild rejects a raw environment containing both PATH and Path. Normalize
rem before VsDevCmd extends the remaining entry.
set "SSLM_PRE_VS_PATH=%PATH%"
set "PATH="
set "Path="
set "PATH=%SSLM_PRE_VS_PATH%"
set "SSLM_PRE_VS_PATH="

set "VSDEVCMD=C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat"
if not exist "%VSDEVCMD%" set "VSDEVCMD=C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat"
if not exist "%VSDEVCMD%" (
	echo ERROR: Visual Studio 2022 VsDevCmd.bat not found.
	exit /b 1
)
call "%VSDEVCMD%" -arch=x64 -no_logo
if errorlevel 1 exit /b 1

pushd "%~dp0\..\.."
set "PACKAGE_BUILD=%CD%\out\install-package-build"
set "PACKAGE_PREFIX=%CD%\out\install-package-prefix"
set "CONSUMER_BUILD=%CD%\out\install-consumer-build"

cmake -S . -B "%PACKAGE_BUILD%" -G "Visual Studio 17 2022" -A x64 -DBUILD_TESTING=OFF -DSUPERSLM_BUILD_GPU=ON -DCMAKE_INSTALL_PREFIX="%PACKAGE_PREFIX%"
if errorlevel 1 goto :fail
cmake --build "%PACKAGE_BUILD%" --config Release --target superslm_gpu
if errorlevel 1 goto :fail
cmake --install "%PACKAGE_BUILD%" --config Release
if errorlevel 1 goto :fail

rem This consumer intentionally declares no C++ standard. Its two translation
rem units include the installed CPU/GPU headers in opposite orders.
cmake -S tests\install-consumer -B "%CONSUMER_BUILD%" -G "Visual Studio 17 2022" -A x64 -DCMAKE_PREFIX_PATH="%PACKAGE_PREFIX%"
if errorlevel 1 goto :fail
cmake --build "%CONSUMER_BUILD%" --config Release
if errorlevel 1 goto :fail
"%CONSUMER_BUILD%\Release\headers_cpu_first.exe"
if errorlevel 1 goto :fail
"%CONSUMER_BUILD%\Release\headers_gpu_first.exe"
if errorlevel 1 goto :fail

set /a INSTALLED_SHADERS=0
for %%f in ("%PACKAGE_PREFIX%\share\superslm\gpu-shaders\*.cso") do set /a INSTALLED_SHADERS+=1
set /a DEPLOYED_SHADERS=0
for %%f in ("%CONSUMER_BUILD%\Release\shaders\*.cso") do set /a DEPLOYED_SHADERS+=1
if !INSTALLED_SHADERS! LEQ 0 (
	echo ERROR: installed package contains no GPU shaders.
	goto :fail
)
if not !INSTALLED_SHADERS!==!DEPLOYED_SHADERS! (
	echo ERROR: shader helper deployed !DEPLOYED_SHADERS! of !INSTALLED_SHADERS! installed shaders.
	goto :fail
)

echo INSTALL CONSUMER: PASS -- both header orders, transitive C++20, and !DEPLOYED_SHADERS! deployed shaders.
popd
exit /b 0

:fail
echo INSTALL CONSUMER: FAILED.
popd
exit /b 1
