@echo off
setlocal
set HERE=%~dp0
for %%I in ("%HERE%..\..") do set REPO=%%~fI
if not defined T2956_BASE_CPU_LIB set T2956_BASE_CPU_LIB=D:\_slm170-probe\build-base\superslm.lib
if not defined T2956_BASE_GPU_LIB set T2956_BASE_GPU_LIB=D:\_slm170-probe\build-base\superslm_gpu.lib
if not defined T2956_BASE_SHADERS set T2956_BASE_SHADERS=D:\_slm170-probe\build-base\gpu-shaders-staged
if not exist "%HERE%out" mkdir "%HERE%out"
if not exist "%HERE%out\shaders" mkdir "%HERE%out\shaders"
copy /Y "%T2956_BASE_SHADERS%\*.cso" "%HERE%out\shaders\" >nul
if errorlevel 1 exit /b 2
call "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 -no_logo
if errorlevel 1 exit /b 2
cl /nologo /std:c++20 /W4 /EHsc /MD /fp:precise /DT2956_BASELINE ^
  /I"%REPO%\include" /Fe"%HERE%out\cell_real_decode_base.exe" ^
  "%HERE%cell_real_decode.cpp" "%T2956_BASE_CPU_LIB%" "%T2956_BASE_GPU_LIB%" ^
  d3d12.lib dxgi.lib dxguid.lib >"%HERE%out\build_baseline.log" 2>&1
set CODE=%errorlevel%
type "%HERE%out\build_baseline.log"
if not "%CODE%"=="0" exit /b %CODE%
cl /nologo /std:c++20 /W4 /EHsc /MD /fp:precise /DT2956_BASELINE ^
  /I"%REPO%\include" /Fe"%HERE%out\cell_private_bytes_base.exe" ^
  "%HERE%cell_private_bytes.cpp" "%T2956_BASE_CPU_LIB%" "%T2956_BASE_GPU_LIB%" ^
  d3d12.lib dxgi.lib dxguid.lib psapi.lib >"%HERE%out\build_private_baseline.log" 2>&1
set CODE=%errorlevel%
type "%HERE%out\build_private_baseline.log"
exit /b %CODE%
