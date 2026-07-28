@echo off
rem Quick MSVC build + test. For other compilers / the full matrix use CMake.
setlocal
set VSDEVCMD="C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat"
if not exist %VSDEVCMD% (
	echo VsDevCmd.bat not found at %VSDEVCMD%
	exit /b 1
)
call %VSDEVCMD% -arch=x64 -no_logo
pushd %~dp0
if not exist out mkdir out
rem tests\test_main.cpp's T-1267 mechanism-pin cell shadow-recompiles this
rem file's own current text a second time via a hardcoded relative include,
rem `#include "..\src\forward_sites.cpp"`, fixed at the file's PRIOR location
rem (tests\ is read-only to this build, so that path is never edited). The
rem copy below regenerates a byte-identical, untracked mirror there for the
rem compile, always taken from the real, single, tracked file at
rem src\forward\forward_sites.cpp; it is deleted below so the working tree is
rem clean afterward (never staged, never committed).
copy /Y src\forward\forward_sites.cpp src\forward_sites.cpp >nul
cl /nologo /std:c++20 /O2 /W4 /fp:precise /EHsc /Iinclude /Itests /DSUPERSLM_ENABLE_BAD_ALLOC_INJECTION ^
	src\artifact.cpp src\sha256.cpp src\tokenizer.cpp src\model.cpp src\intmath.cpp src\silu_lut.cpp src\matmul.cpp src\proof_manifest.cpp src\trace_hook.cpp ^
	src\forward\checked_chain_funnel.cpp src\forward\forward_sites.cpp ^
	tests\test_main.cpp /Fo:out\ /Fe:out\superslm_tests.exe
if errorlevel 1 (
	del src\forward_sites.cpp
	popd & exit /b 1
)
del src\forward_sites.cpp
out\superslm_tests.exe
set ec=%errorlevel%
popd
exit /b %ec%
