@echo off
setlocal
if "%~3"=="" exit /b 2
set SOURCE=%~1
set OUTPUT=%~2
set MODE=%~3
set HERE=%~dp0
for %%I in ("%HERE%..\..") do set REPO=%%~fI
set CMAKE=C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe
call "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 -no_logo
if errorlevel 1 exit /b 3
"%CMAKE%" -S "%SOURCE%" -B "%OUTPUT%" -G Ninja -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF -DSUPERSLM_BUILD_GPU=ON -DSUPERSLM_GPU_TEST_SEAMS=ON "-DCMAKE_CXX_FLAGS=/DSUPERSLM_CPU_G5_FINISH_ROW_FAULT_INJECTION /DSUPERSLM_GPU_G5_FINISH_ROW_FAULT_INJECTION"
if errorlevel 1 exit /b 4
"%CMAKE%" --build "%OUTPUT%" --target superslm superslm_gpu
if errorlevel 1 exit /b 5
if not exist "%OUTPUT%\shaders" mkdir "%OUTPUT%\shaders"
copy /Y "%OUTPUT%\gpu-shaders-staged\*.cso" "%OUTPUT%\shaders\" >nul
if errorlevel 1 exit /b 6
if /I "%MODE%"=="baseline" (
  set T2956_DEFINE=/DT2956_BASELINE
) else (
  set T2956_DEFINE=/DT2956_CANDIDATE
)
cl /nologo /std:c++20 /W4 /EHsc /MD /fp:precise %T2956_DEFINE% /DT2956_ALL_MASKED ^
  /I"%REPO%\include" /Fe"%OUTPUT%\cell_real_decode.exe" ^
  "%HERE%cell_real_decode.cpp" "%OUTPUT%\superslm.lib" "%OUTPUT%\superslm_gpu.lib" ^
  d3d12.lib dxgi.lib dxguid.lib >"%OUTPUT%\cell_real_decode.compile.log" 2>&1
if errorlevel 1 (type "%OUTPUT%\cell_real_decode.compile.log" & exit /b 7)
echo LINKED all-masked %MODE%
exit /b 0
