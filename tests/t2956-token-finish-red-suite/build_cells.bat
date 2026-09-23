@echo off
setlocal
set HERE=%~dp0
for %%I in ("%HERE%..\..") do set REPO=%%~fI
if not exist "%HERE%out" mkdir "%HERE%out"
if not defined T2956_CPU_LIB set T2956_CPU_LIB=D:\_slm170-probe\build-base\superslm.lib
if not defined T2956_GPU_LIB set T2956_GPU_LIB=D:\_slm170-probe\build-base\superslm_gpu.lib
if defined T2956_SHADERS (
  if not exist "%HERE%out\shaders" mkdir "%HERE%out\shaders"
  copy /Y "%T2956_SHADERS%\*.cso" "%HERE%out\shaders\" >nul
  if errorlevel 1 exit /b 2
)
call "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 -no_logo
if errorlevel 1 exit /b 2
set BAD=0
for %%C in (cell_status_ordinals cell_flags cell_setters cell_real_decode cell_rows cell_device_logits cell_alloc_faults cell_residency cell_hook_lifecycle) do (
  echo ===== %%C =====
  cl /nologo /std:c++20 /W4 /EHsc /MD /fp:precise /c /DT2956_CANDIDATE ^
    /I"%REPO%\include" /Fo"%HERE%out\%%C.obj" ^
    "%HERE%%%C.cpp" >"%HERE%out\%%C.compile.log" 2>&1
  if errorlevel 1 (
    type "%HERE%out\%%C.compile.log"
    set BAD=1
  ) else (
    link /nologo /OUT:"%HERE%out\%%C.exe" "%HERE%out\%%C.obj" ^
      "%T2956_CPU_LIB%" "%T2956_GPU_LIB%" d3d12.lib dxgi.lib dxguid.lib ^
      >"%HERE%out\%%C.link.log" 2>&1
    if errorlevel 1 (
      type "%HERE%out\%%C.link.log"
      set BAD=1
    ) else (
      echo LINKED
    )
  )
)
exit /b %BAD%
