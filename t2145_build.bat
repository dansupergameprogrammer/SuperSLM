@echo off
rem T-2145 (Laplace) -- disposable experiment build. Builds sslm_generate and the
rem axis digest in a named kernel variant. NOT product tooling; scratch only.
rem
rem Usage: t2145_build.bat <variant>
rem   scalar : SUPERSLM_FORCE_SCALAR_MATMUL  (the normative reference path)
rem   sse2   : production default (128-bit, 8 lanes/iter)
rem   avx2   : SUPERSLM_MATMUL_T2145_AVX2    (256-bit, 16 lanes/iter)
rem
rem The AVX2 variant uses intrinsics WITHOUT /arch:AVX2, so every other
rem translation unit's codegen is byte-for-byte what the other variants get.
rem That keeps the A/B a one-variable experiment: only the kernel changes.
setlocal
set VSDEVCMD="C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat"
if not exist %VSDEVCMD% (echo VsDevCmd.bat not found & exit /b 1)
call %VSDEVCMD% -arch=x64 -no_logo

set VARIANT=%1
if "%VARIANT%"=="" (echo usage: t2145_build.bat ^<scalar^|sse2^|avx2^> & exit /b 2)

set DEFS=
if "%VARIANT%"=="scalar" set DEFS=/DSUPERSLM_FORCE_SCALAR_MATMUL
if "%VARIANT%"=="avx2"   set DEFS=/DSUPERSLM_MATMUL_T2145_AVX2

pushd %~dp0
if not exist out\t2145\%VARIANT% mkdir out\t2145\%VARIANT%

set CORE=src\artifact.cpp src\sha256.cpp src\tokenizer.cpp src\model.cpp src\intmath.cpp src\silu_lut.cpp src\matmul.cpp src\proof_manifest.cpp src\trace_hook.cpp src\forward\checked_chain_funnel.cpp src\forward\forward_sites.cpp src\decode_digest.cpp

echo [t2145] building sslm_generate (%VARIANT%)
cl /nologo /std:c++20 /O2 /W4 /fp:precise /EHsc /Iinclude %DEFS% ^
	%CORE% tools\sslm_generate.cpp ^
	/Fo:out\t2145\%VARIANT%\ /Fe:out\t2145\sslm_generate_%VARIANT%.exe
if errorlevel 1 (popd & exit /b 1)

echo [t2145] building sslm_axis_digest (%VARIANT%)
cl /nologo /std:c++20 /O2 /W4 /fp:precise /EHsc /Iinclude %DEFS% ^
	%CORE% tools\ci\sslm_axis_digest.cpp ^
	/Fo:out\t2145\%VARIANT%\ /Fe:out\t2145\sslm_axis_digest_%VARIANT%.exe
if errorlevel 1 (popd & exit /b 1)

echo [t2145] building t2145_bench (%VARIANT%)
cl /nologo /std:c++20 /O2 /W4 /fp:precise /EHsc /Iinclude %DEFS% ^
	%CORE% tools\t2145_bench.cpp ^
	/Fo:out\t2145\%VARIANT%\ /Fe:out\t2145\t2145_bench_%VARIANT%.exe
if errorlevel 1 (popd & exit /b 1)

popd
exit /b 0
