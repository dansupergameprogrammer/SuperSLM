@echo off
rem TE-425: builds the red suite against one engine tree and its seam build.
rem Usage: build_red_suite.bat <engine-source-tree> <engine-seam-build-dir> <new-out-dir>
rem   <engine-source-tree>     the `git archive` extraction the seam build was made from (its include\ and
rem                            src\gpu\ are what the cells compile against)
rem   <engine-seam-build-dir>  build_engine.bat's output (superslm.lib, superslm_gpu.lib, gpu-shaders-staged\)
rem   <new-out-dir>            must not exist: every binary read as evidence is built here, fresh
rem The suite's own sources are this directory's; tests\t2956-token-finish-red-suite\cell_alloc_faults.cpp
rem (plan Sec3.5 R1) is built into the same directory.
setlocal
if "%~3"=="" (echo usage: build_red_suite.bat ^<engine-source-tree^> ^<engine-seam-build-dir^> ^<new-out-dir^> & exit /b 2)
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
set CXX=cl /nologo /std:c++20 /W4 /EHsc /MD /O2 /fp:precise
set LIBS="%BLD%\superslm.lib" "%BLD%\superslm_gpu.lib" d3d12.lib dxgi.lib dxguid.lib psapi.lib
set BAD=0
%CXX% /c /I"%TREE%\include" /Fo"%OUT%\te425_cells.obj" "%HERE%te425_cells.cpp" >"%OUT%\te425_cells.compile.txt" 2>&1 || (type "%OUT%\te425_cells.compile.txt" & set BAD=1)
%CXX% /c /I"%TREE%\include" /I"%TREE%\src\gpu" /Fo"%OUT%\te425_getdevice.obj" "%HERE%te425_getdevice.cpp" >"%OUT%\te425_getdevice.compile.txt" 2>&1 || (type "%OUT%\te425_getdevice.compile.txt" & set BAD=1)
if "%BAD%"=="0" (
  link /nologo /OUT:"%OUT%\te425_cells.exe" "%OUT%\te425_cells.obj" "%OUT%\te425_getdevice.obj" %LIBS% >"%OUT%\te425_cells.link.txt" 2>&1 || (type "%OUT%\te425_cells.link.txt" & set BAD=1)
)
rem R13 is red at v1.7.1 by EXIT STATUS, not by compile: it must build at both versions.
%CXX% /I"%TREE%\include" /Fo"%OUT%\te425_r13_ordinals.obj" /Fe"%OUT%\te425_r13_ordinals.exe" "%HERE%te425_r13_ordinals.cpp" "%BLD%\superslm.lib" >"%OUT%\te425_r13.build.txt" 2>&1 || (type "%OUT%\te425_r13.build.txt" & set BAD=1)
rem TE-433: the submission-tail catch-all pin, its own binary so that it can replace operator new and
rem hook ExecuteCommandLists without changing te425_cells.cpp.
%CXX% /I"%TREE%\include" /Fo"%OUT%\te433_tail_pin.obj" /Fe"%OUT%\te433_tail_pin.exe" "%HERE%te433_tail_pin.cpp" %LIBS% >"%OUT%\te433_tail_pin.build.txt" 2>&1 || (type "%OUT%\te433_tail_pin.build.txt" & set BAD=1)
set T2956=%HERE%..\t2956-token-finish-red-suite
%CXX% /DT2956_CANDIDATE /I"%TREE%\include" /Fo"%OUT%\cell_alloc_faults.obj" /Fe"%OUT%\cell_alloc_faults.exe" "%T2956%\cell_alloc_faults.cpp" %LIBS% >"%OUT%\cell_alloc_faults.build.txt" 2>&1 || (type "%OUT%\cell_alloc_faults.build.txt" & set BAD=1)
if "%BAD%"=="0" echo BUILT %OUT%
exit /b %BAD%
