@echo off
setlocal
if "%~4"=="" exit /b 2
set CELL=%~1
set CPU=%~2
set GPU=%~3
set OUT=%~4
set SHADERS=%~5
set HERE=%~dp0
for %%I in ("%HERE%..\..") do set REPO=%%~fI
if not exist "%OUT%" mkdir "%OUT%"
if defined SHADERS (
  if not exist "%OUT%\shaders" mkdir "%OUT%\shaders"
  copy /Y "%SHADERS%\*.cso" "%OUT%\shaders\" >nul
  if errorlevel 1 exit /b 3
)
call "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 -no_logo
if errorlevel 1 exit /b 4
cl /nologo /std:c++20 /W4 /EHsc /MD /fp:precise /c /DT2956_CANDIDATE ^
  /I"%REPO%\include" /Fo"%OUT%\%CELL%.obj" "%HERE%%CELL%.cpp" >"%OUT%\%CELL%.compile.log" 2>&1
if errorlevel 1 (type "%OUT%\%CELL%.compile.log" & exit /b 5)
link /nologo /OUT:"%OUT%\%CELL%.exe" "%OUT%\%CELL%.obj" ^
  "%CPU%" "%GPU%" d3d12.lib dxgi.lib dxguid.lib >"%OUT%\%CELL%.link.log" 2>&1
if errorlevel 1 (type "%OUT%\%CELL%.link.log" & exit /b 6)
echo LINKED %CELL%
exit /b 0
