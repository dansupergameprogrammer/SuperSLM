@echo off
rem T-2948 / design section 4.4. Invoke from any directory with the A-EX tokenizer artifact.
rem Usage: tests\t2948-build-inspect-red.bat "D:\...\qwen2.5-0.5b-instruct-cap4096-aex-tok.sslm"
setlocal
if "%~1"=="" (echo Artifact argument required & exit /b 2)
if not exist "%~1" (echo Artifact missing: %~1 & exit /b 2)
set "ROOT=%~dp0.."
set "ARTIFACT=%~f1"
set "T2948_RED=0"
set "CMAKE_EXE=cmake"
where cmake >nul 2>&1
if errorlevel 1 set "CMAKE_EXE=C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe"
if not exist "%CMAKE_EXE%" if not "%CMAKE_EXE%"=="cmake" (echo cmake unavailable & exit /b 2)
call "%ROOT%\tools\build_inspect.bat"
if errorlevel 1 (echo RED: build_inspect.bat link failed & set "T2948_RED=1")
"%CMAKE_EXE%" -S "%ROOT%" -B "%ROOT%\build\t2948-tools"
if errorlevel 1 exit /b 2
"%CMAKE_EXE%" --build "%ROOT%\build\t2948-tools" --config Release --target sslm_inspect tok_verify
if errorlevel 1 (echo RED: CMake tool targets missing or failed & set "T2948_RED=1")
if "%T2948_RED%"=="1" exit /b 1
"%ROOT%\out\sslm_inspect.exe" "%ARTIFACT%" >"%ROOT%\build\t2948-tools\inspect.log" 2>&1
if errorlevel 1 (type "%ROOT%\build\t2948-tools\inspect.log" & exit /b 1)
findstr /C:"MODELVIEW OK" "%ROOT%\build\t2948-tools\inspect.log" >nul
if errorlevel 1 (type "%ROOT%\build\t2948-tools\inspect.log" & exit /b 1)
"%ROOT%\out\tok_verify.exe" "%ARTIFACT%" "%ROOT%\tests\fixtures\qwen_tok_golden.gld" >"%ROOT%\build\t2948-tools\tok_verify.log" 2>&1
if errorlevel 1 (type "%ROOT%\build\t2948-tools\tok_verify.log" & exit /b 1)
findstr /C:"0 mismatches" "%ROOT%\build\t2948-tools\tok_verify.log" >nul
if errorlevel 1 (type "%ROOT%\build\t2948-tools\tok_verify.log" & exit /b 1)
type "%ROOT%\build\t2948-tools\inspect.log"
type "%ROOT%\build\t2948-tools\tok_verify.log"
echo T-2948 inspect and tokenizer tool check green
exit /b 0
