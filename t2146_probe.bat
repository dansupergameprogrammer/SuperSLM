@echo off
rem T-2146 (Popper) -- disposable commissioning probe build. NOT product tooling.
rem
rem Builds t2145_bench in an AVX2 variant carrying one injected defect, so an
rem independent pass can score the `identity` instrument's must-accept and
rem must-reject constructions. Usage:
rem   t2146_probe.bat <tag> <extra-defines...>
rem e.g. t2146_probe.bat w65536 /DSUPERSLM_T2146_WINDOW=65536
setlocal
set VSDEVCMD="C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat"
if not exist %VSDEVCMD% (echo VsDevCmd.bat not found & exit /b 1)
call %VSDEVCMD% -arch=x64 -no_logo

rem Defines arrive via the PROBE_DEFS environment variable rather than as arguments,
rem because batch argument parsing splits on '=' and would mangle /DNAME=VALUE.
set TAG=%1
if "%TAG%"=="" (echo usage: set PROBE_DEFS=... ^& t2146_probe.bat ^<tag^> & exit /b 2)
set DEFS=%PROBE_DEFS%

pushd %~dp0
if not exist out\t2146\%TAG% mkdir out\t2146\%TAG%

set CORE=src\artifact.cpp src\sha256.cpp src\tokenizer.cpp src\model.cpp src\intmath.cpp src\silu_lut.cpp src\matmul.cpp src\proof_manifest.cpp src\trace_hook.cpp src\forward\checked_chain_funnel.cpp src\forward\forward_sites.cpp src\decode_digest.cpp

echo [t2146] building t2145_bench (%TAG%) defs:%DEFS%
cl /nologo /std:c++20 /O2 /W4 /fp:precise /EHsc /Iinclude /DSUPERSLM_MATMUL_T2145_AVX2 %DEFS% ^
	%CORE% tools\t2145_bench.cpp ^
	/Fo:out\t2146\%TAG%\ /Fe:out\t2146\t2145_bench_%TAG%.exe
if errorlevel 1 (popd & exit /b 1)

popd
exit /b 0
