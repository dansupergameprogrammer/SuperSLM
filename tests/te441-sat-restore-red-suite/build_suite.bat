@echo off
rem TE-441: builds the cells against one engine tree and its build_engine.bat output.
rem Usage: build_suite.bat <engine-source-tree> <engine-build-dir> <new-out-dir>
rem   <engine-source-tree>  the `git archive` extraction build_engine.bat built (its include\ is what the
rem                         cells compile against)
rem   <engine-build-dir>    build_engine.bat's output (superslm_test_injection.lib, superslm_gpu.lib,
rem                         gpu-shaders-staged\)
rem   <new-out-dir>         must not exist: every binary read as evidence is built here, fresh
rem Both executables link superslm_test_injection.lib -- the same core sources as superslm.lib,
rem compiled with the test-only live-state accessor -- in place of superslm.lib, never beside it.
setlocal
if "%~3"=="" (echo usage: build_suite.bat ^<engine-source-tree^> ^<engine-build-dir^> ^<new-out-dir^> & exit /b 2)
if exist "%~3" (echo refusing: %~3 already exists & exit /b 3)
set HERE=%~dp0
set TREE=%~1
set BLD=%~2
set OUT=%~3
mkdir "%OUT%" || exit /b 3
mkdir "%OUT%\shaders" || exit /b 3
copy /Y "%BLD%\gpu-shaders-staged\*.cso" "%OUT%\shaders\" >nul || exit /b 3
call "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 -no_logo
if errorlevel 1 exit /b 4
set CXX=cl /nologo /std:c++20 /W4 /EHsc /MD /O2 /fp:precise /D_CRT_SECURE_NO_WARNINGS
set BAD=0
%CXX% /I"%TREE%\include" /Fo"%OUT%\te441_cpu_cells.obj" /Fe"%OUT%\te441_cpu_cells.exe" "%HERE%te441_cpu_cells.cpp" "%BLD%\superslm_test_injection.lib" >"%OUT%\te441_cpu_cells.build.txt" 2>&1 || (type "%OUT%\te441_cpu_cells.build.txt" & set BAD=1)
%CXX% /I"%TREE%\include" /Fo"%OUT%\te441_gpu_cells.obj" /Fe"%OUT%\te441_gpu_cells.exe" "%HERE%te441_gpu_cells.cpp" "%BLD%\superslm_gpu.lib" "%BLD%\superslm_test_injection.lib" d3d12.lib dxgi.lib dxguid.lib >"%OUT%\te441_gpu_cells.build.txt" 2>&1 || (type "%OUT%\te441_gpu_cells.build.txt" & set BAD=1)
if "%BAD%"=="0" echo BUILT %OUT%
exit /b %BAD%
