@echo off
setlocal enabledelayedexpansion
rem T-2919 (TE-372 S3): builds te368_schema_run.exe from this directory's own vendored
rem te366_schema_run.cpp, against a named engine install, instead of the prior
rem probe's hardcoded D:\_artifacts\superslm\_te259\engine-install-v150 (never committed, unbuildable from a
rem fresh clone). See ../PROVENANCE.md.
rem
rem T-2921 (TE-373 M1): an omitted <engine-install-dir> means THIS checkout's CURRENT source,
rem not a previously installed binary. The no-argument path configures, builds, and installs on
rem every invocation so a source edit cannot be silently measured against a stale engine. Linking
rem a DIFFERENT engine (v1.5.0, or any other build) remains an EXPLICIT override: name its
rem install directory as the first argument.
rem
rem Usage: build_te368_harness.bat [engine-install-dir] [output-dir]
rem   [engine-install-dir] the directory containing include\superslm\sslm_abi.h and
rem     lib\superslm.lib to link.
rem       Omitted: this checkout's own engine (CPU ABI only -- the harness never calls the GPU
rem       surface -- SUPERSLM_BUILD_GPU=OFF, Release) is rebuilt and installed at
rem       <repo-root>\build\install-checkout-under-test on every call, then linked.
rem       Named explicitly: that install is used as-is (already built by the caller) -- e.g. to
rem       compare a different engine (v1.5.0, or any other build) on purpose.
rem   [output-dir] defaults to %TEMP%\te368_harness.

set "SCRIPT_DIR=%~dp0"
for %%I in ("%SCRIPT_DIR%..\..\..\..") do set "REPO_ROOT=%%~fI"

rem Collapse a duplicate PATH/Path pair before VsDevCmd adds the toolchain (build.bat's own
rem convention, same underlying MSBuild rejection).
set "SSLM_PRE_VS_PATH=%PATH%"
set "PATH="
set "Path="
set "PATH=%SSLM_PRE_VS_PATH%"
set "SSLM_PRE_VS_PATH="
call "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 -no_logo

if "%~1"=="" (
    set "ENGINE_INSTALL=%REPO_ROOT%\build\install-checkout-under-test"
    echo [build_te368_harness] no engine-install-dir given -- rebuilding this checkout's current source into "!ENGINE_INSTALL!"
    cmake -S "%REPO_ROOT%" -B "%REPO_ROOT%\build\checkout-under-test" -DSUPERSLM_BUILD_GPU=OFF -DCMAKE_INSTALL_PREFIX="!ENGINE_INSTALL!"
    if errorlevel 1 exit /b 1
    cmake --build "%REPO_ROOT%\build\checkout-under-test" --config Release --target superslm
    if errorlevel 1 exit /b 1
    cmake --install "%REPO_ROOT%\build\checkout-under-test" --config Release
    if errorlevel 1 exit /b 1
) else (
    set "ENGINE_INSTALL=%~1"
)
set "OUT_DIR=%~2"
if "%OUT_DIR%"=="" set "OUT_DIR=%TEMP%\te368_harness"

if not exist "%ENGINE_INSTALL%\include\superslm\sslm_abi.h" (
    echo missing %ENGINE_INSTALL%\include\superslm\sslm_abi.h
    exit /b 1
)
if not exist "%ENGINE_INSTALL%\lib\superslm.lib" (
    echo missing %ENGINE_INSTALL%\lib\superslm.lib
    exit /b 1
)

if not exist "%OUT_DIR%" mkdir "%OUT_DIR%"
cl /nologo /std:c++20 /O2 /EHsc /MD /I"%ENGINE_INSTALL%\include" "%~dp0te366_schema_run.cpp" ^
    /Fe:"%OUT_DIR%\te368_schema_run.exe" /Fo:"%OUT_DIR%\\" /link "%ENGINE_INSTALL%\lib\superslm.lib"
set STATUS=%errorlevel%
if not "%STATUS%"=="0" (
    echo BUILD FAILED
    exit /b %STATUS%
)
echo built %OUT_DIR%\te368_schema_run.exe
exit /b 0
