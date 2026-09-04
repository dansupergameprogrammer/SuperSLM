@echo off
setlocal
pushd %~dp0\..\..

set "CPU_ONLY_BUILD=%CD%\out\cmake-cpu-only-default"
cmake -E remove_directory "%CPU_ONLY_BUILD%"
if errorlevel 1 goto :fail

cmake -S . -B "%CPU_ONLY_BUILD%" -G "Visual Studio 17 2022" -A x64 ^
  -DBUILD_TESTING=OFF -DSUPERSLM_BUILD_GPU=OFF ^
  -DSUPERSLM_DXC_EXECUTABLE="%CPU_ONLY_BUILD%\must-not-exist\dxc.exe"
if errorlevel 1 goto :fail

rem Deliberately build the default target. P1 was a broken superslm_tests link
rem which an explicit --target superslm invocation concealed.
cmake --build "%CPU_ONLY_BUILD%" --config Release
if errorlevel 1 goto :fail

if exist "%CPU_ONLY_BUILD%\superslm_tests.vcxproj" (
  echo CPU-only regression FAILED: superslm_tests target exists with BUILD_TESTING=OFF.
  goto :fail
)
if exist "%CPU_ONLY_BUILD%\superslm_test_injection.vcxproj" (
  echo CPU-only regression FAILED: test-injection target exists with BUILD_TESTING=OFF.
  goto :fail
)
if exist "%CPU_ONLY_BUILD%\Release\superslm_tests.exe" (
  echo CPU-only regression FAILED: superslm_tests.exe was produced.
  goto :fail
)

echo CPU-only default CMake build: PASS -- no DXC, GPU, or test target required.
popd
exit /b 0

:fail
set "CPU_ONLY_EC=%errorlevel%"
if "%CPU_ONLY_EC%"=="0" set "CPU_ONLY_EC=1"
popd
exit /b %CPU_ONLY_EC%
