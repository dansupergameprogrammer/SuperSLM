@echo off
rem TE-441: builds the engine under test from an exact source tree into a build directory that must
rem not already exist, so no earlier build output is read as evidence.
rem Usage: build_engine.bat <source-tree> <new-build-dir>
rem   <source-tree>  a `git archive <sha>` extraction of SuperSLM (84bed02 = v1.8.1 for the red reading,
rem                  the builder's tip for the green reading)
rem Targets: superslm_test_injection (the CPU library with the test-only SslmSeqLiveStateForTest
rem accessor), superslm_gpu and its shaders. No GPU test seams are compiled in.
setlocal
if "%~2"=="" (echo usage: build_engine.bat ^<source-tree^> ^<new-build-dir^> & exit /b 2)
if exist "%~2" (echo refusing: %~2 already exists & exit /b 3)
call "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 -no_logo
if errorlevel 1 exit /b 4
cmake -S "%~1" -B "%~2" -G Ninja -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON -DSUPERSLM_BUILD_GPU=ON
if errorlevel 1 exit /b 5
cmake --build "%~2" --target superslm_test_injection superslm_gpu superslm_gpu_shaders -j 4
exit /b %errorlevel%
