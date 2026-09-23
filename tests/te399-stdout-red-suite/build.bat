@echo off
rem TE-399 (Curie) -- builds cell_gpu_pipeline_stdout_empty against the real engine sources at
rem this branch's tip. Documented-local, like every GPU red suite here (D-SLM3432: no GPU CI
rem runner): this script archives the current worktree into a scratch copy and appends the one
rem executable target to THAT copy's CMakeLists.txt (never the committed one -- this suite's
rem writable scope is test files only), the same technique
rem Claude/Loki/te393-u3-strike-probe/build.bat used to build te393_stdout_probe against v1.7.0.
rem BUILD_TESTING=OFF so only this target and its `superslm_gpu` dependency compile, not the full
rem test_main.cpp suite -- fast enough to run inline under the tool's time limit.
rem
rem Usage (from this directory): build.bat
setlocal enabledelayedexpansion
set HEREDIR=%~dp0
for %%I in ("%HEREDIR%..\..") do set ENG=%%~fI
set SCRATCH=D:\_te399
set SRC=%SCRATCH%\src
set OUT=%SCRATCH%\build

call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul || exit /b 1

if exist "%SRC%" rmdir /s /q "%SRC%"
mkdir "%SRC%"
pushd "%ENG%"
git archive HEAD -o "%SCRATCH%\te399-archive.tar" || (popd & exit /b 1)
popd
pushd "%SRC%"
tar -xf "%SCRATCH%\te399-archive.tar" || (popd & exit /b 1)
popd
if not exist "%SRC%\CMakeLists.txt" (
	echo ARCHIVE FAILED: %SRC%\CMakeLists.txt missing
	exit /b 1
)

(
	echo.
	echo # TE-399 red cell ^(scratch build copy only^)
	echo add_executable^(te399_stdout_pipeline tests/te399-stdout-red-suite/cell_gpu_pipeline_stdout_empty.cpp^)
	echo target_link_libraries^(te399_stdout_pipeline PRIVATE superslm_gpu^)
) >> "%SRC%\CMakeLists.txt"

cmake -G Ninja -S "%SRC%" -B "%OUT%" -DCMAKE_BUILD_TYPE=Release -DSUPERSLM_BUILD_GPU=ON -DBUILD_TESTING=OFF || exit /b 1
cmake --build "%OUT%" --target te399_stdout_pipeline --parallel 4 || exit /b 1
echo BUILD_DONE
