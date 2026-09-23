@echo off
setlocal
if "%~1"=="" exit /b 2
set SRC=%~1
set CMAKE=C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe
call "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 -no_logo
if errorlevel 1 exit /b 3
if not exist "%SRC%\build\t2956-mutant\CMakeCache.txt" (
  "%CMAKE%" -S "%SRC%" -B "%SRC%\build\t2956-mutant" -G Ninja -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF -DSUPERSLM_BUILD_GPU=ON -DSUPERSLM_GPU_TEST_SEAMS=ON
  if errorlevel 1 exit /b 4
)
"%CMAKE%" --build "%SRC%\build\t2956-mutant" --target superslm superslm_gpu
exit /b %ERRORLEVEL%
