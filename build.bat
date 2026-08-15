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

rem T-2113 (B1, Claude/Vitruvius/t2107-gpu-core-1p0-design-2026-08-14.md Sec10 B1):
rem the context-lifecycle bench proof (tools/t2113_b1_context_smoke.cpp) -- built and
rem RUN here (unlike the C5 harness above, this needs no external .sslm artifact),
rem so its own pass/fail folds into this script's exit code. Same full source list as
rem the main test binary (src\gpu\superslm_gpu.cpp is still the pre-1.0 substrate this
rem tool's own PlanDispatchBudgetGpu non-regression check calls into) plus the new
rem src\gpu\gpu_1p0.cpp translation unit B1 adds.
if not exist out\b1 mkdir out\b1
cl /nologo /std:c++20 /O2 /W4 /fp:precise /EHsc /Iinclude /Itests /DSUPERSLM_ENABLE_BAD_ALLOC_INJECTION /DSUPERSLM_O11_ALLOC_INJECTION ^
	src\artifact.cpp src\sha256.cpp src\tokenizer.cpp src\model.cpp src\intmath.cpp src\silu_lut.cpp src\matmul.cpp src\proof_manifest.cpp src\trace_hook.cpp ^
	src\forward\checked_chain_funnel.cpp src\forward\forward_sites.cpp src\decode_digest.cpp ^
	src\gpu\superslm_gpu.cpp src\gpu\gpu_1p0.cpp ^
	tools\t2113_b1_context_smoke.cpp /Fo:out\b1\ /Fe:out\t2113_b1_context_smoke.exe ^
	/link d3d12.lib dxgi.lib dxguid.lib
if errorlevel 1 (
	popd & exit /b 1
)
out\t2113_b1_context_smoke.exe
set b1_ec=%errorlevel%

rem T-2113 (B2, design Sec10 B2): the model-handle-map/unmap bench proof
rem (tools\t2113_b2_model_smoke.cpp) -- needs two real .sslm artifacts, so it is built here
rem but NOT auto-run by default (matching the C5 harness's own precedent above); the build
rem seat's own session invokes it manually against the real 1.5B/0.5B artifacts on disk. Same
rem full source list as B1's own smoke build.
if not exist out\b2 mkdir out\b2
cl /nologo /std:c++20 /O2 /W4 /fp:precise /EHsc /Iinclude /Itests /DSUPERSLM_ENABLE_BAD_ALLOC_INJECTION /DSUPERSLM_O11_ALLOC_INJECTION ^
	src\artifact.cpp src\sha256.cpp src\tokenizer.cpp src\model.cpp src\intmath.cpp src\silu_lut.cpp src\matmul.cpp src\proof_manifest.cpp src\trace_hook.cpp ^
	src\forward\checked_chain_funnel.cpp src\forward\forward_sites.cpp src\decode_digest.cpp ^
	src\gpu\superslm_gpu.cpp src\gpu\gpu_1p0.cpp ^
	tools\t2113_b2_model_smoke.cpp /Fo:out\b2\ /Fe:out\t2113_b2_model_smoke.exe ^
	/link d3d12.lib dxgi.lib dxguid.lib
if errorlevel 1 (
	popd & exit /b 1
)

rem T-2113 (B3, design Sec10 B3): the sequence-handle bench proof
rem (tools\t2113_b3_sequence_smoke.cpp) -- needs one real .sslm artifact, so it is built here
rem but NOT auto-run by default (matching B2's own precedent above); the build seat's own
rem session invokes it manually against the real 1.5B artifact on disk. Same full source list
rem as B1/B2's own smoke builds.
if not exist out\b3 mkdir out\b3
cl /nologo /std:c++20 /O2 /W4 /fp:precise /EHsc /Iinclude /Itests /DSUPERSLM_ENABLE_BAD_ALLOC_INJECTION /DSUPERSLM_O11_ALLOC_INJECTION ^
	src\artifact.cpp src\sha256.cpp src\tokenizer.cpp src\model.cpp src\intmath.cpp src\silu_lut.cpp src\matmul.cpp src\proof_manifest.cpp src\trace_hook.cpp ^
	src\forward\checked_chain_funnel.cpp src\forward\forward_sites.cpp src\decode_digest.cpp ^
	src\gpu\superslm_gpu.cpp src\gpu\gpu_1p0.cpp ^
	tools\t2113_b3_sequence_smoke.cpp /Fo:out\b3\ /Fe:out\t2113_b3_sequence_smoke.exe ^
	/link d3d12.lib dxgi.lib dxguid.lib
if errorlevel 1 (
	popd & exit /b 1
)

rem T-2113 (B5, design Sec10 B5): the async-boundary bench proof
rem (tools\t2113_b5_async_smoke.cpp) -- needs one real .sslm artifact, so it is built here
rem but NOT auto-run by default (matching B2/B3's own precedent above); the build seat's own
rem session invokes it manually against the real 1.5B artifact on disk (once with a clean
rem environment, and twice more with each of SSLM_B5_ASYNC_DROP_UAV_REBIND/
rem SSLM_B5_ASYNC_SWAP_SRV_REBIND=1 set, for the plant-and-revert violation-pin protocol --
rem see that tool's own header comment). Same full source list as B1/B2/B3's own smoke builds.
if not exist out\b5 mkdir out\b5
cl /nologo /std:c++20 /O2 /W4 /fp:precise /EHsc /Iinclude /Itests /DSUPERSLM_ENABLE_BAD_ALLOC_INJECTION /DSUPERSLM_O11_ALLOC_INJECTION ^
	src\artifact.cpp src\sha256.cpp src\tokenizer.cpp src\model.cpp src\intmath.cpp src\silu_lut.cpp src\matmul.cpp src\proof_manifest.cpp src\trace_hook.cpp ^
	src\forward\checked_chain_funnel.cpp src\forward\forward_sites.cpp src\decode_digest.cpp ^
	src\gpu\superslm_gpu.cpp src\gpu\gpu_1p0.cpp ^
	tools\t2113_b5_async_smoke.cpp /Fo:out\b5\ /Fe:out\t2113_b5_async_smoke.exe ^
	/link d3d12.lib dxgi.lib dxguid.lib
if errorlevel 1 (
	popd & exit /b 1
)

rem T-2113 (B6 checkpoint, design Sec10 B6): the adapter-residency/guard bench proof
rem (tools\t2113_b6_adapter_smoke.cpp) -- needs three real artifacts (1.5B model, 0.5B model,
rem a real converted adapter), so it is built here but NOT auto-run (matching B2/B3/B5's own
rem precedent above). Proves residency/base-hash validation/the AdapterModelMismatch guard --
rem NOT a numerical divergence from a bound adapter, since no GEMM-site dispatch reads the
rem adapter's own resident buffers yet (Claude/Brunel/t2113-1p0-core-build-2026-08-15.md Sec9).
if not exist out\b6 mkdir out\b6
cl /nologo /std:c++20 /O2 /W4 /fp:precise /EHsc /Iinclude /Itests /DSUPERSLM_ENABLE_BAD_ALLOC_INJECTION /DSUPERSLM_O11_ALLOC_INJECTION ^
	src\artifact.cpp src\sha256.cpp src\tokenizer.cpp src\model.cpp src\intmath.cpp src\silu_lut.cpp src\matmul.cpp src\proof_manifest.cpp src\trace_hook.cpp ^
	src\forward\checked_chain_funnel.cpp src\forward\forward_sites.cpp src\decode_digest.cpp ^
	src\gpu\superslm_gpu.cpp src\gpu\gpu_1p0.cpp ^
	tools\t2113_b6_adapter_smoke.cpp /Fo:out\b6\ /Fe:out\t2113_b6_adapter_smoke.exe ^
	/link d3d12.lib dxgi.lib dxguid.lib
if errorlevel 1 (
	popd & exit /b 1
)

out\superslm_tests.exe
set ec=%errorlevel%
if not %b1_ec%==0 set ec=%b1_ec%

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
	rem T-2101 (the reviewer's own named residual, code review 6d9e04e-t2101-gpu-throughput-review.md,
	rem second confirmation pass): the shader half of the original S3 class -- each split GEMM site's
	rem own [numthreads(N,1,1)] and stride formula, cross-checked against ComputeGpuGemmSiteGroupPlan's
	rem own threads_per_group for that site, so a host/shader thread-width divergence fails the build
	rem instead of producing a silent wrong answer at real dimensions.
	python tests\ci\check_gemm_site_thread_width_parity.py
	if errorlevel 1 (
		echo check_gemm_site_thread_width_parity.py FAILED -- see output above
		set ec=1
	)
) else (
	echo python not found on PATH -- skipping tests\ci\check_gpu_guard_status_parity.py and check_gemm_site_thread_width_parity.py ^(non-fatal^)
)

popd
exit /b %ec%
