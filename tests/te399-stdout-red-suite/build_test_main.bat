@echo off
rem TE-399 (Curie) -- builds superslm_tests (tests/test_main.cpp, ~30k lines, includes the T-2116
rem cells this ticket edits) against this branch's tip, target-only (not "tools", not the other
rem GPU red suites) to keep this documented-local confirmation run inside the tool's time limit.
setlocal enabledelayedexpansion
set HEREDIR=%~dp0
for %%I in ("%HEREDIR%..\..") do set ENG=%%~fI
set SCRATCH=D:\_te399
set SRC=%SCRATCH%\src2
set OUT=%SCRATCH%\build2

call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1

if exist "%SRC%" rmdir /s /q "%SRC%"
mkdir "%SRC%"
pushd "%ENG%"
git archive HEAD -o "%SCRATCH%\te399-archive2.tar" || (popd & exit /b 1)
popd
pushd "%SRC%"
tar -xf "%SCRATCH%\te399-archive2.tar" || (popd & exit /b 1)
popd

cmake -G Ninja -S "%SRC%" -B "%OUT%" -DCMAKE_BUILD_TYPE=Release -DSUPERSLM_BUILD_GPU=ON -DBUILD_TESTING=ON || exit /b 1
cmake --build "%OUT%" --target superslm_tests --parallel 4 || exit /b 1
echo BUILD_DONE
