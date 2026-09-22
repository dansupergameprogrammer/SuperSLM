@echo off
setlocal enabledelayedexpansion
rem T-2919 (TE-372 S3): builds te368_schema_run.exe from this directory's own vendored
rem te366_schema_run.cpp, against an explicitly-named engine install, instead of the prior
rem probe's hardcoded D:\_te259\engine-install-v150 (never committed, unbuildable from a
rem fresh clone). See ../PROVENANCE.md.
rem
rem Usage: build_te368_harness.bat <engine-install-dir> [output-dir]
rem   <engine-install-dir> must contain include\superslm\sslm_abi.h and lib\superslm.lib --
rem   e.g. a v1.5.0 install (TE-372 M6's own disclosed choice, unchanged by this script) or a
rem   fresh build of this branch's own engine. Which one to link is the caller's decision; this
rem   script only makes the choice explicit and the harness source reproducible.
rem   [output-dir] defaults to %TEMP%\te368_harness.

if "%~1"=="" (
    echo usage: build_te368_harness.bat ^<engine-install-dir^> [output-dir]
    echo   engine-install-dir must contain include\superslm\sslm_abi.h and lib\superslm.lib
    exit /b 1
)
set ENGINE_INSTALL=%~1
set OUT_DIR=%~2
if "%OUT_DIR%"=="" set OUT_DIR=%TEMP%\te368_harness

if not exist "%ENGINE_INSTALL%\include\superslm\sslm_abi.h" (
    echo missing %ENGINE_INSTALL%\include\superslm\sslm_abi.h
    exit /b 1
)
if not exist "%ENGINE_INSTALL%\lib\superslm.lib" (
    echo missing %ENGINE_INSTALL%\lib\superslm.lib
    exit /b 1
)

call "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 -no_logo
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
