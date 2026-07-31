@echo off
rem T-1409 Popper probe build. Same toolchain and flags as ..\build.bat and as
rem the solver's build_experiment.bat, so the probe is measured in the solver's
rem own cell.
setlocal
set VSDEVCMD="C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat"
if not exist %VSDEVCMD% (
	echo VsDevCmd.bat not found at %VSDEVCMD%
	exit /b 1
)
call %VSDEVCMD% -arch=x64 -no_logo
pushd %~dp0\..
if not exist out mkdir out
cl /nologo /std:c++20 /O2 /W4 /fp:precise /EHsc /Iinclude /Itests %POPPER_DEFINES% ^
	src\artifact.cpp src\sha256.cpp src\tokenizer.cpp src\model.cpp src\intmath.cpp src\silu_lut.cpp src\matmul.cpp src\proof_manifest.cpp src\trace_hook.cpp ^
	src\forward\checked_chain_funnel.cpp src\forward\forward_sites.cpp src\decode_digest.cpp ^
	experiments\popper_t1409_probe.cpp /Fo:out\popper\ /Fe:out\popper_probe.exe
set ec=%errorlevel%
popd
exit /b %ec%
