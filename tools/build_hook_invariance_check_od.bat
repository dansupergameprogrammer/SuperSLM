@echo off
rem Build the T-1300 instrumentation-invariance check with MSVC, /Od (adhoc
rem tool build; links the clean superslm core). Mirrors
rem tools\build_hook_invariance_check.bat exactly except for the optimization
rem flag and the output exe name -- codegen (inlining, register allocation,
rem vectorization) is the plausible mechanism by which installing a hook could
rem perturb integer arithmetic, and this project has been burned by an
rem optimization-level difference before (/Od vs /O2, DecisionLog D-SLM487/
rem D-SLM492/D-SLM495), so the T-1300 measurement runs both.
setlocal
set VSDEVCMD="C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat"
if not exist %VSDEVCMD% (echo VsDevCmd.bat not found & exit /b 1)
call %VSDEVCMD% -arch=x64 -no_logo
pushd %~dp0\..
if not exist out mkdir out
cl /nologo /std:c++20 /Od /W4 /fp:precise /EHsc /Iinclude ^
	src\artifact.cpp src\sha256.cpp src\tokenizer.cpp src\model.cpp src\intmath.cpp src\silu_lut.cpp src\matmul.cpp src\proof_manifest.cpp src\trace_hook.cpp ^
	src\forward\checked_chain_funnel.cpp src\forward\forward_sites.cpp src\decode_digest.cpp ^
	tools\sslm_hook_invariance_check.cpp /Fo:out\od_ /Fe:out\sslm_hook_invariance_check_od.exe
set ec=%errorlevel%
popd
exit /b %ec%
