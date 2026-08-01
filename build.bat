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
if not exist out\tests mkdir out\tests
rem tests\areas\*.cpp files are deliberately named after the src\ production TU
rem each tests -- tests\areas\proof_manifest.cpp vs src\proof_manifest.cpp, and
rem so on for every later Stage -- so /Fo must put src\ and tests\ objects in
rem separate directories. A single shared /Fo:out\ silently let the later
rem compile overwrite the earlier same-named .obj on disk (LNK4042 "object
rem specified more than once"), dropping tests\areas\proof_manifest.cpp's own
rem SSLM_TEST registrations from the link without any build failure --
rem T-1574 Stage 1, found by executing this exact script against the split.
cl /nologo /std:c++20 /O2 /W4 /fp:precise /EHsc /Iinclude /Itests /DSUPERSLM_ENABLE_BAD_ALLOC_INJECTION ^
	src\artifact.cpp src\sha256.cpp src\tokenizer.cpp src\model.cpp src\intmath.cpp src\silu_lut.cpp src\matmul.cpp src\proof_manifest.cpp src\trace_hook.cpp ^
	src\forward\checked_chain_funnel.cpp src\forward\forward_sites.cpp src\decode_digest.cpp ^
	/Fo:out\ /c
if errorlevel 1 (
	popd & exit /b 1
)
cl /nologo /std:c++20 /O2 /W4 /fp:precise /EHsc /Iinclude /Itests /DSUPERSLM_ENABLE_BAD_ALLOC_INJECTION ^
	tests\test_main.cpp tests\support\test_harness.cpp tests\support\test_registry.cpp ^
	tests\support\crash_probe.cpp tests\support\shared_fixtures.cpp ^
	tests\areas\proof_manifest.cpp ^
	/Fo:out\tests\ /c
if errorlevel 1 (
	popd & exit /b 1
)
cl /nologo out\artifact.obj out\sha256.obj out\tokenizer.obj out\model.obj out\intmath.obj out\silu_lut.obj ^
	out\matmul.obj out\proof_manifest.obj out\trace_hook.obj out\checked_chain_funnel.obj out\forward_sites.obj ^
	out\decode_digest.obj out\tests\test_main.obj out\tests\test_harness.obj out\tests\test_registry.obj ^
	out\tests\crash_probe.obj out\tests\shared_fixtures.obj out\tests\proof_manifest.obj ^
	/Fe:out\superslm_tests.exe
if errorlevel 1 (
	popd & exit /b 1
)
out\superslm_tests.exe
set ec=%errorlevel%
popd
exit /b %ec%
