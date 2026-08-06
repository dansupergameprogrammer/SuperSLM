@echo off
rem Build the t1768_qln2_derivation_probe tool with MSVC (adhoc tool build; links the clean
rem superslm core). Extends T-1763's own t1763_layer0_diagnosis_probe.cpp (branch
rem claude/t1763-layer0-diagnosis@10cd1e2, read not owned) with the q_ln2 derivation-chain
rem diagnostics T-1768 needs (normed_scale, q_scale, per-kv-head softmax carried scale, the
rem combined scale, and DynamicScaleReciprocal's own r_m -- the exact inputs
rem IExpScaleConstants consumes at its real production call site).
setlocal
set VSDEVCMD="C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat"
if not exist %VSDEVCMD% (echo VsDevCmd.bat not found & exit /b 1)
call %VSDEVCMD% -arch=x64 -no_logo
pushd %~dp0\..
if not exist out mkdir out
cl /nologo /std:c++20 /O2 /W4 /fp:precise /EHsc /Iinclude ^
	src\artifact.cpp src\sha256.cpp src\tokenizer.cpp src\model.cpp src\intmath.cpp src\silu_lut.cpp src\matmul.cpp src\proof_manifest.cpp src\trace_hook.cpp ^
	src\forward\checked_chain_funnel.cpp src\forward\forward_sites.cpp src\decode_digest.cpp ^
	tools\t1768_qln2_derivation_probe.cpp /Fo:out\ /Fe:out\t1768_qln2_derivation_probe.exe
set ec=%errorlevel%
popd
exit /b %ec%
