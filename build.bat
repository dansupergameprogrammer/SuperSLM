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
if not exist out\shaders mkdir out\shaders

rem T-1986 GPU-serial port (Sec5.7): every dxc invocation this design's build
rem issues adds -WX, at the pinned compile target (cs_6_2 -HV 2018 -O3).
set DXC="C:\Program Files (x86)\Windows Kits\10\bin\10.0.19041.0\x64\dxc.exe"
if not exist %DXC% (
	echo dxc.exe not found at %DXC%
	popd & exit /b 1
)
for %%f in (src\gpu\shaders\*.hlsl) do (
	%DXC% -T cs_6_2 -E main -Fo out\shaders\%%~nf.cso %%f -O3 -HV 2018 -WX
	if errorlevel 1 (
		popd & exit /b 1
	)
)

cl /nologo /std:c++20 /O2 /W4 /fp:precise /EHsc /Iinclude /Itests /DSUPERSLM_ENABLE_BAD_ALLOC_INJECTION /DSUPERSLM_O11_ALLOC_INJECTION ^
	src\artifact.cpp src\sha256.cpp src\tokenizer.cpp src\model.cpp src\intmath.cpp src\silu_lut.cpp src\matmul.cpp src\proof_manifest.cpp src\trace_hook.cpp ^
	src\forward\checked_chain_funnel.cpp src\forward\forward_sites.cpp src\decode_digest.cpp ^
	src\gpu\superslm_gpu.cpp ^
	tests\test_main.cpp /Fo:out\ /Fe:out\superslm_tests.exe ^
	/link d3d12.lib dxgi.lib dxguid.lib
if errorlevel 1 (
	popd & exit /b 1
)

rem T-2045 (S5, Claude/Poirot/82cfca7-gpu-serial-port-build-review.md): the C5
rem harness (tools/t2039_c5_harness.cpp) had no committed build recipe -- the
rem one load-bearing result of the whole GPU-serial-port arc was not
rem reproducible from HEAD (the design's own S-3 lesson, Sec2). Built here,
rem alongside the test binary, from the identical source list plus
rem tools/sslm_marshal.h's own -Itools include path -- NOT auto-run (it needs
rem a real .sslm artifact on disk this build does not assume exists, matching
rem tools/t1657_load_harness.cpp's own precedent of a built-but-manually-
rem invoked tool). Usage after a successful build: out\t2039_c5_harness.exe
rem ^<model.sslm^> [token_id].
if not exist out\c5 mkdir out\c5
cl /nologo /std:c++20 /O2 /W4 /fp:precise /EHsc /Iinclude /Itests /Itools /DSUPERSLM_ENABLE_BAD_ALLOC_INJECTION ^
	src\artifact.cpp src\sha256.cpp src\tokenizer.cpp src\model.cpp src\intmath.cpp src\silu_lut.cpp src\matmul.cpp src\proof_manifest.cpp src\trace_hook.cpp ^
	src\forward\checked_chain_funnel.cpp src\forward\forward_sites.cpp src\decode_digest.cpp ^
	src\gpu\superslm_gpu.cpp ^
	tools\t2039_c5_harness.cpp /Fo:out\c5\ /Fe:out\t2039_c5_harness.exe ^
	/link d3d12.lib dxgi.lib dxguid.lib
if errorlevel 1 (
	popd & exit /b 1
)

out\superslm_tests.exe
set ec=%errorlevel%

rem T-2091 (O30's own local-half closure, Claude/Poirot/2aceac3-gpu-serial-port-ship-candidate-
rem review.md; build log §27): this script ran no Python at all until now, so the O11 gate-flag pin
rem and the rest of tests/ci/check_gpu_guard_status_parity.py's own structural population never
rem fired on the LOCAL build path -- only in GitHub Actions, which does not even compile
rem src/gpu/superslm_gpu.cpp into the target that runs it (EXECUTION_SCOPE_WAIVERS's own named,
rem dated residual in that same module). Guarded, non-fatal if python is absent: this script's own
rem contract is a C++-only build, and "don't chase CI" (Claude/CLAUDE.md) is about not gating the
rem local build on tooling that may not be installed, never about skipping a check that IS
rem installed and IS the real gate this arc's own ship decisions run against.
where python >nul 2>nul
if not errorlevel 1 (
	python tests\ci\check_gpu_guard_status_parity.py
	if errorlevel 1 (
		echo check_gpu_guard_status_parity.py FAILED -- see output above
		set ec=1
	)
) else (
	echo python not found on PATH -- skipping tests\ci\check_gpu_guard_status_parity.py ^(non-fatal^)
)

popd
exit /b %ec%
