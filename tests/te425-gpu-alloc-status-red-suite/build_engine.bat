@echo off
rem TE-425: builds the engine under test with the GPU test seams, from an exact source tree, into a
rem build directory that must not already exist (so no earlier build output is read as evidence).
rem Usage: build_engine.bat <source-tree> <new-build-dir>
rem   <source-tree>: a `git archive <sha>` extraction of SuperSLM (v1.7.1 = 2a32042 for the red reading,
rem                  the builder's tip for the green reading).
rem The seam build defines SUPERSLM_GPU_ALLOC_FAULT_INJECTION only (CMakeLists.txt,
rem SUPERSLM_GPU_TEST_SEAMS), the same configuration tests/t2956-token-finish-red-suite links against.
setlocal
if "%~2"=="" (echo usage: build_engine.bat ^<source-tree^> ^<new-build-dir^> & exit /b 2)
if exist "%~2" (echo refusing: %~2 already exists & exit /b 3)
call "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 -no_logo
if errorlevel 1 exit /b 4
cmake -S "%~1" -B "%~2" -G Ninja -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF -DSUPERSLM_BUILD_GPU=ON -DSUPERSLM_GPU_TEST_SEAMS=ON
if errorlevel 1 exit /b 5
cmake --build "%~2" --target superslm superslm_gpu superslm_gpu_shaders -j 4
exit /b %errorlevel%
