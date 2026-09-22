@echo off
setlocal enabledelayedexpansion
rem T-2919 (TE-372 S3): builds te368_schema_run.exe from this directory's own vendored
rem te366_schema_run.cpp, against a named engine install, instead of the prior
rem probe's hardcoded D:\_te259\engine-install-v150 (never committed, unbuildable from a
rem fresh clone). See ../PROVENANCE.md.
rem
rem T-2920 (TE-372 M6): every run of this script so far named an explicit v1.5.0 install (the
rem operator's own available build), so no 1.6.0 engine had run the string field on a real model
rem before TE-372's own review built one to check. <engine-install-dir> is now OPTIONAL: omitted,
rem this script builds (if not already built) and links THIS CHECKOUT'S OWN engine -- the tip
rem actually under test -- rather than leaving the choice to whatever install directory happens
rem to be lying around. Linking a DIFFERENT engine (v1.5.0, or any other build) is still
rem available, as an EXPLICIT override: name its install directory as the first argument.
rem
rem Usage: build_te368_harness.bat [engine-install-dir] [output-dir]
rem   [engine-install-dir] the directory containing include\superslm\sslm_abi.h and
rem     lib\superslm.lib to link.
rem       Omitted: this checkout's own engine (CPU ABI only -- the harness never calls the GPU
rem       surface -- SUPERSLM_BUILD_GPU=OFF, Release) is built if not already present at
rem       <repo-root>\build\install-checkout-under-test and installed there, then linked. A later
rem       call with no argument reuses that install without rebuilding.
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
    if not exist "!ENGINE_INSTALL!\include\superslm\sslm_abi.h" (
        echo [build_te368_harness] no engine-install-dir given -- building this checkout's own engine into "!ENGINE_INSTALL!"
        cmake -S "%REPO_ROOT%" -B "%REPO_ROOT%\build\checkout-under-test" -DSUPERSLM_BUILD_GPU=OFF -DCMAKE_INSTALL_PREFIX="!ENGINE_INSTALL!"
        if errorlevel 1 exit /b 1
        cmake --build "%REPO_ROOT%\build\checkout-under-test" --config Release --target superslm
        if errorlevel 1 exit /b 1
        cmake --install "%REPO_ROOT%\build\checkout-under-test" --config Release
        if errorlevel 1 exit /b 1
    ) else (
        echo [build_te368_harness] no engine-install-dir given -- reusing this checkout's own engine already built at "!ENGINE_INSTALL!"
    )
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
