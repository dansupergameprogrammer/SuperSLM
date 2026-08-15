@echo off
rem Build the T-2102 runtime-LoRA switch demonstration driver (adhoc tool build; links the
rem clean superslm core -- same convention as build_generate.bat/build_adapter_dump.bat).
setlocal
set VSDEVCMD="C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat"
if not exist %VSDEVCMD% (echo VsDevCmd.bat not found & exit /b 1)
call %VSDEVCMD% -arch=x64 -no_logo
pushd %~dp0\..
if not exist out mkdir out
cl /nologo /std:c++20 /O2 /W4 /fp:precise /EHsc /Iinclude ^
	src\artifact.cpp src\sha256.cpp src\tokenizer.cpp src\model.cpp src\intmath.cpp src\silu_lut.cpp src\matmul.cpp src\proof_manifest.cpp src\trace_hook.cpp ^
	src\forward\checked_chain_funnel.cpp src\forward\forward_sites.cpp src\decode_digest.cpp ^
	tools\sslm_lora_switch_demo.cpp /Fo:out\ /Fe:out\sslm_lora_switch_demo.exe
set ec=%errorlevel%
popd
exit /b %ec%
