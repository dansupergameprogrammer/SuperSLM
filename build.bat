@echo off
rem Quick MSVC build + test. For other compilers / the full matrix use CMake.
setlocal
rem Some process hosts supply both PATH and Path entries. MSBuild imports the raw
rem environment into a case-insensitive table and rejects that duplicate before CL runs.
rem Collapse the pair before VsDevCmd adds the compiler toolchain directories.
set "SSLM_PRE_VS_PATH=%PATH%"
set "PATH="
set "Path="
set "PATH=%SSLM_PRE_VS_PATH%"
set "SSLM_PRE_VS_PATH="
set VSDEVCMD="C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat"
if not exist %VSDEVCMD% (
	echo VsDevCmd.bat not found at %VSDEVCMD%
	exit /b 1
)
call %VSDEVCMD% -arch=x64 -no_logo
pushd %~dp0
if not exist out mkdir out
if not exist out\shaders mkdir out\shaders

rem SuperSLM 1.4.0 external re-review P1: prove the advertised Windows CPU-only
rem configuration survives a clean DEFAULT build with tests and GPU both off.
call tests\cmake-cpu-only\run_windows.bat
if errorlevel 1 (
	popd & exit /b 1
)

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

cl /nologo /std:c++20 /O2 /W4 /fp:precise /EHsc /Iinclude /Itests /DSUPERSLM_ENABLE_BAD_ALLOC_INJECTION /DSUPERSLM_O11_ALLOC_INJECTION /DSUPERSLM_ENABLE_GPU_API_FAILURE_INJECTION ^
	src\artifact.cpp src\sha256.cpp src\tokenizer.cpp src\model.cpp src\intmath.cpp src\silu_lut.cpp src\matmul.cpp src\proof_manifest.cpp src\trace_hook.cpp ^
	src\forward\checked_chain_funnel.cpp src\forward\forward_sites.cpp src\decode_digest.cpp ^
	src\gpu\superslm_gpu.cpp src\gpu\gpu_1p0.cpp ^
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

rem T-2441 (Poirot 327ee29-t2438-ask5-tracka-review.md, Significant 5, D-SLM5439): the T-2432
rem Track A acceptance harness (tools/t2432_geometry_harness.cpp) had no committed build
rem recipe -- the one load-bearing result of the whole Ask 5 Track A arc was not reproducible
rem from HEAD, the identical S-3 lesson tools/t2039_c5_harness.cpp's own header (and its own
rem recipe immediately above) already records. Built here, alongside it, from the identical
rem source list -- NOT auto-run (it needs a real .sslm artifact on disk this build does not
rem assume exists, matching tools/t2039_c5_harness.cpp's own precedent). Usage after a
rem successful build: first compile the 33 GPU shaders into a `shaders\` directory beside the
rem harness executable exactly as this file's own dxc loop above does (ShaderPath resolves
rem .cso relative to the EXECUTABLE's directory, never the working directory -- skipping this
rem step produces a false GpuAllocationFailed, this ticket's own build log Sec7 names the exact
rem failure mode); then out\t2432_geometry_harness.exe ^<model.sslm^> [token_id]. A non-square
rem fixture is generated with tools\_t2432_nonsquare_fixture.py <checkpoint_dir> {nonsquare|
rem square}, then tools\calibrate_checkpoint.py --checkpoint <checkpoint_dir> --out <calibrated_
rem dir>, then tools\convert_model.py --artifact <calibrated_dir> --out <model.sslm>. T-2445
rem (Claude/Poirot/ddbc57a-t2443-ask5-tracka-confirmation.md, Minor 6, D-SLM5436 superseded):
rem this comment used to end the sequence with --skip-verify, a weaker form of design Sec6
rem Track A's own acceptance criterion that the fixture load "through the standard
rem convert_model.py pipeline" -- --skip-verify prints its own warning that the artifact's
rem must-load-Ok contract is unproven for that conversion. Executed with sslm_verify built
rem (cmake --build build --config Release --target sslm_verify) and the flag dropped: the
rem non-square fixture converts cleanly, independent loader accepted the artifact, proof
rem manifest written. The flag is not needed; dropped from the documented sequence.
if not exist out\geoharness mkdir out\geoharness
cl /nologo /std:c++20 /O2 /W4 /fp:precise /EHsc /Iinclude /Itests /Itools ^
	src\artifact.cpp src\sha256.cpp src\tokenizer.cpp src\model.cpp src\intmath.cpp src\silu_lut.cpp src\matmul.cpp src\proof_manifest.cpp src\trace_hook.cpp ^
	src\forward\checked_chain_funnel.cpp src\forward\forward_sites.cpp src\decode_digest.cpp ^
	src\gpu\superslm_gpu.cpp ^
	tools\t2432_geometry_harness.cpp /Fo:out\geoharness\ /Fe:out\t2432_geometry_harness.exe ^
	/link d3d12.lib dxgi.lib dxguid.lib
if errorlevel 1 (
	popd & exit /b 1
)

rem T-2169 (Brunel, Rung 2b, Claude/Brunel/t2180-t2169-gpu-batched-prefill-build-2026-08-18.md):
rem the chunk-submission primitive's own self-check harness (tools/t2169_rung2b_selfcheck.cpp) --
rem design Sec8's exit condition (b)/(b2), the chunk_tokens=1 mechanism cell and the 4-token
rem discriminating cell (D-SLM3608/D-SLM3615), both against the real Qwen2.5-1.5B artifact. Same
rem committed-build-recipe discipline t2039_c5_harness.cpp's own S5 lesson established above --
rem NOT auto-run (needs a real .sslm artifact on disk this build does not assume exists). Usage
rem after a successful build: out\t2169_rung2b_selfcheck.exe ^<model.sslm^>.
if not exist out\t2169selfcheck mkdir out\t2169selfcheck
cl /nologo /std:c++20 /O2 /W4 /fp:precise /EHsc /Iinclude /Itests /Itools /DSUPERSLM_ENABLE_BAD_ALLOC_INJECTION ^
	src\artifact.cpp src\sha256.cpp src\tokenizer.cpp src\model.cpp src\intmath.cpp src\silu_lut.cpp src\matmul.cpp src\proof_manifest.cpp src\trace_hook.cpp ^
	src\forward\checked_chain_funnel.cpp src\forward\forward_sites.cpp src\decode_digest.cpp ^
	src\gpu\superslm_gpu.cpp ^
	tools\t2169_rung2b_selfcheck.cpp /Fo:out\t2169selfcheck\ /Fe:out\t2169_rung2b_selfcheck.exe ^
	/link d3d12.lib dxgi.lib dxguid.lib
if errorlevel 1 (
	popd & exit /b 1
)

rem T-2169 (Brunel, Rung 2, D-SLM3596): the TDR-safe bound measurement harness
rem (tools/t2169_tdr_measure.cpp) -- queries this machine's own real TDR delay and measures
rem gpu_busy_ms across a swept chunk_tokens range against the real Qwen2.5-1.5B artifact. Same
rem committed-build-recipe discipline as the two tools above; links advapi32.lib for the registry
rem query. NOT auto-run. Usage after a successful build: out\t2169_tdr_measure.exe ^<model.sslm^>.
rem STATUS (Claude/Brunel/t2180-t2169-gpu-batched-prefill-build-2026-08-18.md): this tool's own
rem measurement run found chunk_tokens>=8 reproducibly crashes the process
rem (STATUS_STACK_OVERFLOW) -- a genuine, hardware-executed defect inside the NVIDIA driver
rem (nvwgf2umx.dll), not a TDR event (no Display/nvlddmkm TDR-recovery event logged at the crash
rem time), root-caused under cdb (D-SLM3649). T-2184 remedy M2 (Brunel fix round 1, D-SLM3662):
rem corrected -- superslm_gpu::kT2169TdrSafeMaxChunkTokens is DEFINED (src/gpu/superslm_gpu.cpp,
rem `= 4`, half the confirmed-crashing 8) as of Rung 2's close; this paragraph described the gap
rem only until then.
if not exist out\t2169tdr mkdir out\t2169tdr
cl /nologo /std:c++20 /O2 /W4 /fp:precise /EHsc /Iinclude /Itests /Itools /DSUPERSLM_ENABLE_BAD_ALLOC_INJECTION ^
	src\artifact.cpp src\sha256.cpp src\tokenizer.cpp src\model.cpp src\intmath.cpp src\silu_lut.cpp src\matmul.cpp src\proof_manifest.cpp src\trace_hook.cpp ^
	src\forward\checked_chain_funnel.cpp src\forward\forward_sites.cpp src\decode_digest.cpp ^
	src\gpu\superslm_gpu.cpp ^
	tools\t2169_tdr_measure.cpp /Fo:out\t2169tdr\ /Fe:out\t2169_tdr_measure.exe ^
	/link d3d12.lib dxgi.lib dxguid.lib advapi32.lib
if errorlevel 1 (
	popd & exit /b 1
)

rem T-2180 (Brunel, Rung 6, Claude/Brunel/t2180-t2169-gpu-batched-prefill-build-2026-08-18.md):
rem the public-bridge bit-identity sweep (tools\t2180_rung6_public_bridge_sweep.cpp) -- design
rem Sec6's own size/boundary-split proof cells generalized to the exact size list this build's own
rem log reports against ({1,2,4,8,16,256}, plus splits straddling the TDR-safe sub-chunk bound of
rem 4), driven through SslmGpuSeqPrefillPromptForG5Bridge, the same PUBLIC entry point Rungs 3/4
rem wired. Same committed-build-recipe discipline as the two tools above. NOT auto-run (needs a
rem real .sslm artifact). Usage after a successful build: out\t2180_rung6_sweep.exe ^<model.sslm^>.
if not exist out\t2180 mkdir out\t2180
cl /nologo /std:c++20 /O2 /W4 /fp:precise /EHsc /Iinclude /Itests /Itools /DSUPERSLM_ENABLE_BAD_ALLOC_INJECTION ^
	src\artifact.cpp src\sha256.cpp src\tokenizer.cpp src\model.cpp src\intmath.cpp src\silu_lut.cpp src\matmul.cpp src\proof_manifest.cpp src\trace_hook.cpp ^
	src\forward\checked_chain_funnel.cpp src\forward\forward_sites.cpp src\decode_digest.cpp ^
	src\gpu\superslm_gpu.cpp src\gpu\gpu_1p0.cpp ^
	tools\t2180_rung6_public_bridge_sweep.cpp /Fo:out\t2180\ /Fe:out\t2180_rung6_sweep.exe ^
	/link d3d12.lib dxgi.lib dxguid.lib
if errorlevel 1 (
	popd & exit /b 1
)

rem T-2180 (Brunel, Rung 6): the product cell -- tok/s on a long forced span, chunked (one
rem public-bridge call) vs shipped-per-token (N separate single-token public-bridge calls),
rem through the SAME public entry point (tools\t2180_rung6_tokps.cpp). 1.1's headline GPU number
rem (design Sec10). NOT auto-run. Usage after a successful build:
rem out\t2180_rung6_tokps.exe ^<model.sslm^> [span_len].
rem T-2185 remedy N6/observation (Brunel fix round 2, D-SLM3677): /DSUPERSLM_ENABLE_GPU_BENCH_
rem PRE_BATCHING added -- this is the ONLY compile line in this script that defines it, gating
rem SslmGpuSeqPrefillPromptPreBatchingBenchOnly (gpu_1p0.cpp/gpu_1p0_bench_bridge.h) into THIS
rem binary only. Every other line below that compiles src\gpu\gpu_1p0.cpp does not define this
rem macro, so that symbol is absent from those binaries entirely.
if not exist out\t2180 mkdir out\t2180
cl /nologo /std:c++20 /O2 /W4 /fp:precise /EHsc /Iinclude /Itests /Itools /DSUPERSLM_ENABLE_BAD_ALLOC_INJECTION /DSUPERSLM_ENABLE_GPU_BENCH_PRE_BATCHING ^
	src\artifact.cpp src\sha256.cpp src\tokenizer.cpp src\model.cpp src\intmath.cpp src\silu_lut.cpp src\matmul.cpp src\proof_manifest.cpp src\trace_hook.cpp ^
	src\forward\checked_chain_funnel.cpp src\forward\forward_sites.cpp src\decode_digest.cpp ^
	src\gpu\superslm_gpu.cpp src\gpu\gpu_1p0.cpp ^
	tools\t2180_rung6_tokps.cpp /Fo:out\t2180\ /Fe:out\t2180_rung6_tokps.exe ^
	/link d3d12.lib dxgi.lib dxguid.lib
if errorlevel 1 (
	popd & exit /b 1
)

rem T-2116 (cross-vendor certification package): a minimal, dependency-free adapter
rem enumeration tool (tools\t2116_list_adapters.cpp) -- no .sslm artifact needed, so it is
rem built here and NOT auto-run; run_crossvendor.ps1 invokes it directly. See that file's
rem own header comment for why the battery tools themselves cannot serve this purpose.
if not exist out\t2116 mkdir out\t2116
cl /nologo /std:c++20 /O2 /W4 /EHsc ^
	tools\t2116_list_adapters.cpp /Fo:out\t2116\ /Fe:out\t2116_list_adapters.exe ^
	/link d3d12.lib dxgi.lib dxguid.lib
if errorlevel 1 (
	popd & exit /b 1
)

rem T-2100 (dispatch-path throughput benchmark, tools\t2100_gpu_throughput.cpp) -- same
rem throwaway-harness precedent as tools\t2039_c5_harness.cpp above (needs a real .sslm
rem artifact on disk this build does not assume exists, so it is built here but NOT
rem auto-run). Added by T-2116 (cross-vendor certification package): this tool had no
rem committed build recipe before this change, even though the T-2113/T-2114 build logs
rem cite its own measured tok/s figures -- the recipe mirrors C5's exactly (same source
rem list plus tools\sslm_marshal.h's -Itools include path). Usage after a successful
rem build: out\t2100_gpu_throughput.exe ^<model.sslm^> [steps] [token_id].
if not exist out\t2100 mkdir out\t2100
cl /nologo /std:c++20 /O2 /W4 /fp:precise /EHsc /Iinclude /Itests /Itools /DSUPERSLM_ENABLE_BAD_ALLOC_INJECTION ^
	src\artifact.cpp src\sha256.cpp src\tokenizer.cpp src\model.cpp src\intmath.cpp src\silu_lut.cpp src\matmul.cpp src\proof_manifest.cpp src\trace_hook.cpp ^
	src\forward\checked_chain_funnel.cpp src\forward\forward_sites.cpp src\decode_digest.cpp ^
	src\gpu\superslm_gpu.cpp ^
	tools\t2100_gpu_throughput.cpp /Fo:out\t2100\ /Fe:out\t2100_gpu_throughput.exe ^
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

rem T-2113 (B3.5, design Sec5.3a/Sec10 B3.5, mini-fold 2026-08-15 routing D-SLM3367): the
rem embed-token bench proof (tools\t2113_b35_embed_smoke.cpp) -- needs one real .sslm artifact,
rem so it is built here but NOT auto-run by default (matching B2/B3's own precedent above);
rem the build seat's own session invokes it manually against the real 1.5B artifact on disk.
rem Same full source list as B1/B2/B3's own smoke builds.
if not exist out\b35 mkdir out\b35
cl /nologo /std:c++20 /O2 /W4 /fp:precise /EHsc /Iinclude /Itests /DSUPERSLM_ENABLE_BAD_ALLOC_INJECTION /DSUPERSLM_O11_ALLOC_INJECTION ^
	src\artifact.cpp src\sha256.cpp src\tokenizer.cpp src\model.cpp src\intmath.cpp src\silu_lut.cpp src\matmul.cpp src\proof_manifest.cpp src\trace_hook.cpp ^
	src\forward\checked_chain_funnel.cpp src\forward\forward_sites.cpp src\decode_digest.cpp ^
	src\gpu\superslm_gpu.cpp src\gpu\gpu_1p0.cpp ^
	tools\t2113_b35_embed_smoke.cpp /Fo:out\b35\ /Fe:out\t2113_b35_embed_smoke.exe ^
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

rem T-2113 (B6b, design Sec10 B6): the GEMM-site adapter delta-application divergence proof
rem (tools\t2113_b6b_adapter_delta_smoke.cpp) -- needs the real 1.5B model and the real
rem shopkeeper adapter, so it is built here but NOT auto-run (matching B2/B3/B5/B6's own
rem precedent above).
if not exist out\b6b mkdir out\b6b
cl /nologo /std:c++20 /O2 /W4 /fp:precise /EHsc /Iinclude /Itests /DSUPERSLM_ENABLE_BAD_ALLOC_INJECTION /DSUPERSLM_O11_ALLOC_INJECTION ^
	src\artifact.cpp src\sha256.cpp src\tokenizer.cpp src\model.cpp src\intmath.cpp src\silu_lut.cpp src\matmul.cpp src\proof_manifest.cpp src\trace_hook.cpp ^
	src\forward\checked_chain_funnel.cpp src\forward\forward_sites.cpp src\decode_digest.cpp ^
	src\gpu\superslm_gpu.cpp src\gpu\gpu_1p0.cpp ^
	tools\t2113_b6b_adapter_delta_smoke.cpp /Fo:out\b6b\ /Fe:out\t2113_b6b_adapter_delta_smoke.exe ^
	/link d3d12.lib dxgi.lib dxguid.lib
if errorlevel 1 (
	popd & exit /b 1
)

rem T-2113 (B7, design Sec4.3/Sec7/Sec10 B7): the batch-decode bench proof
rem (tools\t2113_b7_batch_smoke.cpp) -- needs the real 1.5B model and the real shopkeeper
rem adapter, so it is built here but NOT auto-run (matching B2/B3/B5/B6/B6b's own precedent
rem above).
if not exist out\b7 mkdir out\b7
cl /nologo /std:c++20 /O2 /W4 /fp:precise /EHsc /Iinclude /Itests /DSUPERSLM_ENABLE_BAD_ALLOC_INJECTION /DSUPERSLM_O11_ALLOC_INJECTION ^
	src\artifact.cpp src\sha256.cpp src\tokenizer.cpp src\model.cpp src\intmath.cpp src\silu_lut.cpp src\matmul.cpp src\proof_manifest.cpp src\trace_hook.cpp ^
	src\forward\checked_chain_funnel.cpp src\forward\forward_sites.cpp src\decode_digest.cpp ^
	src\gpu\superslm_gpu.cpp src\gpu\gpu_1p0.cpp ^
	tools\t2113_b7_batch_smoke.cpp /Fo:out\b7\ /Fe:out\t2113_b7_batch_smoke.exe ^
	/link d3d12.lib dxgi.lib dxguid.lib
if errorlevel 1 (
	popd & exit /b 1
)

rem T-2113 (B8, design Sec5.4/Sec10 B8): the thread-safety bench proof
rem (tools\t2113_b8_thread_smoke.cpp) -- needs the real 1.5B model, so it is built here but NOT
rem auto-run (matching B2/B3/B5/B6/B6b/B7's own precedent above).
if not exist out\b8 mkdir out\b8
cl /nologo /std:c++20 /O2 /W4 /fp:precise /EHsc /Iinclude /Itests /DSUPERSLM_ENABLE_BAD_ALLOC_INJECTION /DSUPERSLM_O11_ALLOC_INJECTION ^
	src\artifact.cpp src\sha256.cpp src\tokenizer.cpp src\model.cpp src\intmath.cpp src\silu_lut.cpp src\matmul.cpp src\proof_manifest.cpp src\trace_hook.cpp ^
	src\forward\checked_chain_funnel.cpp src\forward\forward_sites.cpp src\decode_digest.cpp ^
	src\gpu\superslm_gpu.cpp src\gpu\gpu_1p0.cpp ^
	tools\t2113_b8_thread_smoke.cpp /Fo:out\b8\ /Fe:out\t2113_b8_thread_smoke.exe ^
	/link d3d12.lib dxgi.lib dxguid.lib
if errorlevel 1 (
	popd & exit /b 1
)

rem T-2124 (D-SLM3446, adapter UAF fix -- Claude/Poirot/435f730-t2124-adapter-uaf-review.md's own
rem adopted recommendation): a standalone executable witness for the adapter use-after-free fix,
rem needing the real 1.5B model and the real shopkeeper adapter, so it is built here but NOT
rem auto-run (matching B2/B3/B5/B6/B6b/B7/B8's own precedent above). Usage after a successful
rem build: out\t2124_adapter_uaf_repro.exe ^<model1p5b.sslm^> ^<adapter.sslm^> [--concurrent].
if not exist out\t2124 mkdir out\t2124
cl /nologo /std:c++20 /O2 /W4 /fp:precise /EHsc /Iinclude /Itests /DSUPERSLM_ENABLE_BAD_ALLOC_INJECTION /DSUPERSLM_O11_ALLOC_INJECTION ^
	src\artifact.cpp src\sha256.cpp src\tokenizer.cpp src\model.cpp src\intmath.cpp src\silu_lut.cpp src\matmul.cpp src\proof_manifest.cpp src\trace_hook.cpp ^
	src\forward\checked_chain_funnel.cpp src\forward\forward_sites.cpp src\decode_digest.cpp ^
	src\gpu\superslm_gpu.cpp src\gpu\gpu_1p0.cpp ^
	tools\t2124_adapter_uaf_repro.cpp /Fo:out\t2124\ /Fe:out\t2124_adapter_uaf_repro.exe ^
	/link d3d12.lib dxgi.lib dxguid.lib
if errorlevel 1 (
	popd & exit /b 1
)

rem T-2139 (Brunel, C1/C2): the Layer-1 CPU-side sslm_* ABI's own sizing/construction/model-
rem lifecycle object, Gates A/C as standing must-accept + must-reject CI fixtures (design
rem Claude/Vitruvius/t2133-layer1-c-abi-design-2026-08-16.md Sec9), and C2's own Gate B smoke.
if not exist out\t2139 mkdir out\t2139
cl /nologo /std:c++20 /O2 /W4 /fp:precise /EHsc /Iinclude /c src\sslm_abi.cpp /Fo:out\t2139\sslm_abi.obj
if errorlevel 1 (
	popd & exit /b 1
)

rem Gate A must-accept: compiles, links, runs to exit 0.
cl /nologo /std:c++20 /O2 /W4 /EHsc /Iinclude /Itools /Itests tools\t2139_gate_a_header_parity_check.cpp /Fo:out\t2139\ /Fe:out\t2139_gate_a_header_parity_check.exe
if errorlevel 1 (
	echo Gate A must-accept construction FAILED TO COMPILE -- this is a real regression, not expected
	popd & exit /b 1
)
out\t2139_gate_a_header_parity_check.exe
if errorlevel 1 (
	popd & exit /b 1
)

rem Gate A must-reject: MUST fail to compile. errorlevel 0 here is the regression.
cl /nologo /std:c++20 /O2 /W4 /EHsc /Iinclude /Itools /Itests tools\t2139_gate_a_header_parity_check_negative.cpp /Fo:out\t2139\ /Fe:out\t2139_gate_a_header_parity_check_negative.exe >out\t2139\gate_a_negative.log 2>&1
if not errorlevel 1 (
	echo Gate A must-reject construction COMPILED CLEAN -- Gate A has regressed, see out\t2139\gate_a_negative.log
	popd & exit /b 1
)
rem T-2139 sixth confirmation review (Claude/Poirot/5fbd04d-t2139-sixth-confirmation-review.md S1):
rem same marker-text treatment as the X-macro/sentinel negatives below -- a compile failure alone
rem is not proof the INTENDED assertion fired. This is Gate A's ONLY proof Gate A can fail at all
rem (its own header comment: the corruption is hand-maintained, not script-generated, so a stale
rem hand copy after a real declaration change fails the compile for a reason that is not the
rem assertion, and exit-code-only reports that as correct).
findstr /C:"sslm_model_map: library signature diverges from tests/t2130-g5-red-suite/sslm_g5.h" out\t2139\gate_a_negative.log >nul
if errorlevel 1 (
	echo Gate A must-reject construction failed to compile, but NOT for its own assertion -- see out\t2139\gate_a_negative.log
	popd & exit /b 1
)
echo Gate A must-reject construction correctly failed to compile, for its own reason ^(marker text confirmed, see out\t2139\gate_a_negative.log^)

rem Gate C must-accept (S8 fix round, Claude/Poirot/2c18dab-t2139-abi-build-review.md): now
rem includes the REAL include/superslm/sslm_abi.h at global scope for the library side (only the
rem suite side is still a hand transcription -- see the file's own header comment for why one
rem side, not both, must avoid the [dcl.link] collision).
rem
rem PLAIN MUST-PASS, matching Gate A (the known-block disclosure arm this step previously
rem carried is RETIRED): the block it disclosed closed when sslm_g5.h took the T-2133 Sec6
rem full-registry mirror (curie/t2130-g5-red-suite@52dc6cd) and the checker's second-half
rem assertion took the ruling's shape (registry-top identity; see the checker's own comments).
rem A compile failure here is a real regression now -- treating it as known/disclosed would
rem report the exact wrong direction.
cl /nologo /std:c++20 /O2 /W4 /EHsc /Iinclude tools\t2139_gate_c_type_identity_check.cpp /Fo:out\t2139\ /Fe:out\t2139_gate_c_type_identity_check.exe >out\t2139\gate_c_must_accept.log 2>&1
if errorlevel 1 (
	echo Gate C must-accept construction FAILED TO COMPILE -- this is a real regression, not expected -- see out\t2139\gate_c_must_accept.log
	type out\t2139\gate_c_must_accept.log
	popd & exit /b 1
)
out\t2139_gate_c_type_identity_check.exe
if errorlevel 1 (
	popd & exit /b 1
)
echo Gate C must-accept construction: PASS

rem Gate C must-reject: MUST fail to compile. errorlevel 0 here is the regression.
cl /nologo /std:c++20 /O2 /W4 /EHsc /Iinclude tools\t2139_gate_c_type_identity_check_negative.cpp /Fo:out\t2139\ /Fe:out\t2139_gate_c_type_identity_check_negative.exe >out\t2139\gate_c_negative.log 2>&1
if not errorlevel 1 (
	echo Gate C must-reject construction COMPILED CLEAN -- Gate C has regressed, see out\t2139\gate_c_negative.log
	popd & exit /b 1
)
rem T-2139 sixth confirmation review (Claude/Poirot/5fbd04d-t2139-sixth-confirmation-review.md S1):
rem same marker-text treatment as the X-macro/sentinel negatives below -- a compile failure alone
rem is not proof the INTENDED assertion fired.
findstr /C:"SSLM_ARTIFACT_REJECTED diverges" out\t2139\gate_c_negative.log >nul
if errorlevel 1 (
	echo Gate C must-reject construction failed to compile, but NOT for its own assertion -- see out\t2139\gate_c_negative.log
	popd & exit /b 1
)
echo Gate C must-reject construction correctly failed to compile, for its own reason ^(marker text confirmed, see out\t2139\gate_c_negative.log^)

rem Gate C's THIRD TU (S1, Claude/Poirot/3bcbe43-t2139-fourth-confirmation-review.md, owed remedy
rem 5): the structural half of S1's fix -- includes the REAL tests/t2130-g5-red-suite/sslm_g5.h
rem directly (not a transcription) alongside the library's own SSLM_STATUS_ENUM_LIST expansion, so
rem a divergence in the REAL suite header itself (not just a stale hand-transcription of it) fails
rem here. /Itests resolves the real suite header's own #include path (matching Gate A's own
rem convention above). MUST-ACCEPT: a compile failure here is a real regression.
cl /nologo /std:c++20 /O2 /W4 /EHsc /Iinclude /Itests tools\t2139_gate_c_real_suite_side_check.cpp /Fo:out\t2139\ /Fe:out\t2139_gate_c_real_suite_side_check.exe >out\t2139\gate_c_real_suite_side.log 2>&1
if errorlevel 1 (
	echo Gate C real-suite-side construction FAILED TO COMPILE -- this is a real regression, not expected -- see out\t2139\gate_c_real_suite_side.log
	type out\t2139\gate_c_real_suite_side.log
	popd & exit /b 1
)
out\t2139_gate_c_real_suite_side_check.exe
if errorlevel 1 (
	popd & exit /b 1
)
echo Gate C real-suite-side construction: PASS

rem T-2139 sixth confirmation review O1 (Claude/Poirot/5fbd04d-t2139-sixth-confirmation-review.md):
rem the third TU just proven above is must-accept-only -- nothing else in this battery proves it
rem CAN fail. This scripted mutate-compile-revert probe closes that gap as a repeatable step rather
rem than something re-derived by hand each round: it mutates a SCRATCH copy of the header only (a
rem one-sided append, under out\ which is gitignored and never touches the tracked file), confirms
rem the real third TU then fails to compile for its own registry-top sentinel reason, and reverts
rem the scratch tree, asserting git status is clean before and after. Guarded by `where python`
rem below (non-fatal skip if absent, same convention as the other python checkers this file runs).
where python >nul 2>nul
if not errorlevel 1 (
	python tools\ci\gate_c_third_tu_can_fail_probe.py
	if errorlevel 1 (
		echo gate_c_third_tu_can_fail_probe.py FAILED -- see output above
		popd & exit /b 1
	)
) else (
	echo python not found on PATH -- skipping tools\ci\gate_c_third_tu_can_fail_probe.py ^(non-fatal^)
)

rem Gate C must-reject, X-MACRO GENERATION mechanism specifically (S4): MUST fail to compile.
cl /nologo /std:c++20 /O2 /W4 /EHsc /Iinclude tools\t2139_gate_c_xmacro_check_negative.cpp /Fo:out\t2139\ /Fe:out\t2139_gate_c_xmacro_check_negative.exe >out\t2139\gate_c_xmacro_negative.log 2>&1
if not errorlevel 1 (
	echo Gate C X-macro must-reject construction COMPILED CLEAN -- Gate C has regressed, see out\t2139\gate_c_xmacro_negative.log
	popd & exit /b 1
)
rem T-2139 fifth confirmation review (Claude/Poirot/ce5aff2-t2139-fifth-confirmation-review.md S3):
rem a compile failure alone is not proof the INTENDED assertion fired -- findstr the log this step
rem already writes for the construction's own marker text, so a failure for the wrong reason (a
rem stray syntax error, a missing header) does not read as "correctly failed."
rem T-2142 (M1, Claude/Poirot/aea6116-t2139-seventh-confirmation-review.md): this marker was
rem TRUNCATED here until this fix -- "SSLM_ARTIFACT_REJECTED diverges" is a genuine PREFIX of the
rem construction's own full static_assert message, missing the macro-generated suffix
rem " (deliberate corruption, must-reject construction)". A truncated fragment of a real message
rem IS a substring of that message, so it passed findstr vacuously the whole time -- reproduced
rem against this exact log with the old text before fixing it. Full text now, matching
rem CMakeLists.txt's own -DMARKER_TEXT= fix for the same target.
findstr /C:"SSLM_ARTIFACT_REJECTED diverges (deliberate corruption, must-reject construction)" out\t2139\gate_c_xmacro_negative.log >nul
if errorlevel 1 (
	echo Gate C X-macro must-reject construction failed to compile, but NOT for its own assertion -- see out\t2139\gate_c_xmacro_negative.log
	popd & exit /b 1
)
echo Gate C X-macro must-reject construction correctly failed to compile, for its own reason ^(marker text confirmed, see out\t2139\gate_c_xmacro_negative.log^)

rem Gate C must-reject, SENTINEL IDENTITY mechanism specifically (S4): MUST fail to compile.
cl /nologo /std:c++20 /O2 /W4 /EHsc /Iinclude tools\t2139_gate_c_sentinel_negative.cpp /Fo:out\t2139\ /Fe:out\t2139_gate_c_sentinel_negative.exe >out\t2139\gate_c_sentinel_negative.log 2>&1
if not errorlevel 1 (
	echo Gate C sentinel must-reject construction COMPILED CLEAN -- Gate C has regressed, see out\t2139\gate_c_sentinel_negative.log
	popd & exit /b 1
)
rem Same marker-text treatment as the X-macro negative above (S3): this is the exact gap the
rem fifth confirmation review found live -- one governed append to both real headers desynchronizes
rem this construction's own extra enumerator from the real sentinels, so it still fails to compile,
rem but for an unrelated C2039 name-lookup reason, and this step's own exit-code-only check kept
rem reporting "correctly failed" with nothing showing the sentinel mechanism itself had stopped
rem firing.
rem T-2142 (M1, Claude/Poirot/aea6116-t2139-seventh-confirmation-review.md): TRUNCATED here too
rem until this fix -- the old text dropped the construction's own literal prefix
rem "registry-top divergence (deliberate one-sided append, must-reject construction): the two ".
rem Same class as the X-macro fix above, same fix (full text, matching CMakeLists.txt).
findstr /C:"registry-top divergence (deliberate one-sided append, must-reject construction): the two SSLM_STATUS_NEXT_FREE sentinels no longer agree" out\t2139\gate_c_sentinel_negative.log >nul
if errorlevel 1 (
	echo Gate C sentinel must-reject construction failed to compile, but NOT for its own assertion -- see out\t2139\gate_c_sentinel_negative.log
	popd & exit /b 1
)
echo Gate C sentinel must-reject construction correctly failed to compile, for its own reason ^(marker text confirmed, see out\t2139\gate_c_sentinel_negative.log^)

rem T-2141 (Claude/Zelda/Board.md T-2141 row; Claude/Poirot/3bcbe43-t2139-fourth-confirmation-
rem review.md O3): the t2138 red suite's OWN sslm_abi.h enum copy was guarded only by the
rem 564-cell link -- no compile-time parity check anywhere referenced it. This pair closes that
rem for sslm_status: must-accept reads the REAL tests\t2138-abi-red-suite\sslm_abi.h directly
rem (namespaced, per-name plus sentinel, same shape as Gate C's own type-identity check above);
rem must-reject proves the per-name generation mechanism itself can fail (see the .cpp files'
rem own header comments for the exact extern "C" collision this construction hit and routed
rem around via SUPERSLM_ABI_ENUM_ONLY).
cl /nologo /std:c++20 /O2 /W4 /EHsc /Iinclude /Itests tools\t2141_gate_c_t2138_suite_side_check.cpp /Fo:out\t2139\ /Fe:out\t2141_gate_c_t2138_suite_side_check.exe >out\t2139\t2141_must_accept.log 2>&1
if errorlevel 1 (
	echo T-2141 t2138-suite-side must-accept construction FAILED TO COMPILE -- this is a real regression, not expected -- see out\t2139\t2141_must_accept.log
	type out\t2139\t2141_must_accept.log
	popd & exit /b 1
)
out\t2141_gate_c_t2138_suite_side_check.exe
if errorlevel 1 (
	popd & exit /b 1
)
echo T-2141 t2138-suite-side must-accept construction: PASS

cl /nologo /std:c++20 /O2 /W4 /EHsc /Iinclude tools\t2141_gate_c_t2138_suite_side_check_negative.cpp /Fo:out\t2139\ /Fe:out\t2141_gate_c_t2138_suite_side_check_negative.exe >out\t2139\t2141_negative.log 2>&1
if not errorlevel 1 (
	echo T-2141 t2138-suite-side must-reject construction COMPILED CLEAN -- has regressed, see out\t2139\t2141_negative.log
	popd & exit /b 1
)
findstr /C:"SSLM_ARTIFACT_REJECTED diverges (deliberate corruption, must-reject construction)" out\t2139\t2141_negative.log >nul
if errorlevel 1 (
	echo T-2141 t2138-suite-side must-reject construction failed to compile, but NOT for its own assertion -- see out\t2139\t2141_negative.log
	popd & exit /b 1
)
echo T-2141 t2138-suite-side must-reject construction correctly failed to compile, for its own reason ^(marker text confirmed, see out\t2139\t2141_negative.log^)

rem count_abi_verbs.sh's own cited figure (T-2139 design Sec4): 29, for the C1-C7 scope alone.
rem RAISED to 34 (T-2132, G5-2): five new production verbs landed in
rem sslm_abi_functions_g5_comparable.inc (sslm_schema_lookup, sslm_schema_count,
rem sslm_schema_name, sslm_seq_set_schema, sslm_prefix_set_schema -- design Claude/Vitruvius/
rem t2119-g5-constrained-decoding-design-2026-08-16.md Sec5, Wizard repo) -- a deliberate,
rem documented extension of the ABI surface this counter is meant to CATCH undocumented drift
rem against, not itself an instance of that drift. T-2199 then raised it to 35 for
rem sslm_decode_step_v2 and the 1.2 candidate raises it to 36 for sslm_decode_params_init.
rem Count with native PowerShell so the gate does not depend on WSL, Git Bash's installation
rem layout, or a separately configured Unix-tool PATH on Windows.
rem T2139_VERB_COUNT is read and compared OUTSIDE any parenthesized if-block on purpose: %VAR%
rem inside a `( ... )` block expands at PARSE time (before the block's own `set /p` line has
rem run) without `setlocal enabledelayedexpansion`, which this script does not otherwise need
rem and this step does not want to turn on globally -- goto/labels sidestep it instead.
powershell -NoLogo -NoProfile -NonInteractive -Command "$n = Get-Content 'include/superslm/sslm_abi_functions.inc','include/superslm/sslm_abi_functions_g5_comparable.inc' | Select-String -AllMatches 'sslm_[a-z0-9_]+\s*\(' | ForEach-Object { $_.Matches.Value -replace '\s*\($','' } | Sort-Object -Unique; $n.Count | Set-Content -Encoding ascii 'out/t2139/verb_count.txt'"
if errorlevel 1 (
	echo ABI verb inventory command failed ^(exit %errorlevel%^) -- this is not a measured verb-count drift
	popd & exit /b 1
)
set /p T2139_VERB_COUNT=<out\t2139\verb_count.txt
if "%T2139_VERB_COUNT%"=="36" goto :t2139_verb_count_ok
echo count_abi_verbs.sh reports %T2139_VERB_COUNT%, expected 36 -- verb count drifted, see design Sec4 / T-2132 / T-2199
popd & exit /b 1
:t2139_verb_count_ok
echo count_abi_verbs.sh: 36 verbs, matches T-2139 Sec4's 29 plus T-2132/G5's five verbs and T-2199's versioned decode and params-initializer verbs
goto :t2139_verb_count_done
:t2139_verb_count_done

rem C2's own Gate B smoke: maps a real artifact, exercises C1's own construction verbs against
rem it, calls sslm_seq_state_size, unmaps. NOT auto-run against a real artifact here (matching
rem B2/B3/.../t2124's own precedent above -- needs a real .sslm this build does not assume
rem exists on every machine); built so it is ready. Usage: out\t2139_c2_smoke.exe ^<model.sslm^>.
cl /nologo /std:c++20 /O2 /W4 /fp:precise /EHsc /Iinclude ^
	src\artifact.cpp src\sha256.cpp src\tokenizer.cpp src\model.cpp src\intmath.cpp src\silu_lut.cpp src\matmul.cpp src\proof_manifest.cpp src\trace_hook.cpp ^
	src\forward\checked_chain_funnel.cpp src\forward\forward_sites.cpp src\decode_digest.cpp ^
	src\sslm_abi.cpp ^
	src\damped_greedy_antilm.cpp src\damped_greedy_topk.cpp src\damped_greedy_phaseD.cpp ^
	tools\t2139_c2_smoke.cpp /Fo:out\t2139\ /Fe:out\t2139_c2_smoke.exe
if errorlevel 1 (
	popd & exit /b 1
)
rem S9 (Claude/Poirot/2c18dab-t2139-abi-build-review.md): Gate B's own must-accept half was
rem compiled but never RUN by this build. Auto-run when a real artifact is available (set
rem T2139_MODEL, and optionally T2139_MODEL2 for the C2 pool/model-mismatch pin) -- SKIPs, never
rem silently passes, when unset, since not every machine carries these artifacts.
if defined T2139_MODEL (
	out\t2139_c2_smoke.exe %T2139_MODEL% %T2139_MODEL2%
	if errorlevel 1 ( popd & exit /b 1 )
) else (
	echo t2139_c2_smoke: built, NOT run ^(set T2139_MODEL=path\to\real.sslm to run^)
)
cl /nologo /std:c++20 /O2 /W4 /fp:precise /EHsc /Iinclude ^
	src\artifact.cpp src\sha256.cpp src\tokenizer.cpp src\model.cpp src\intmath.cpp src\silu_lut.cpp src\matmul.cpp src\proof_manifest.cpp src\trace_hook.cpp ^
	src\forward\checked_chain_funnel.cpp src\forward\forward_sites.cpp src\decode_digest.cpp ^
	src\sslm_abi.cpp ^
	src\damped_greedy_antilm.cpp src\damped_greedy_topk.cpp src\damped_greedy_phaseD.cpp ^
	tools\t2139_c2_smoke_negative.cpp /Fo:out\t2139\ /Fe:out\t2139_c2_smoke_negative.exe
if errorlevel 1 (
	popd & exit /b 1
)
if defined T2139_MODEL (
	out\t2139_c2_smoke_negative.exe %T2139_MODEL%
	if errorlevel 1 ( popd & exit /b 1 )
) else (
	echo t2139_c2_smoke_negative: built, NOT run ^(set T2139_MODEL=path\to\real.sslm to run^)
)

rem C3's own Gate B smoke: begins a prefix, prefills, freezes, creates a sequence, adopts the
rem prefix, releases both (design Sec9's own stated C3 smoke shape), plus pool-exhaustion and
rem free-count-exactness paths. NOT auto-run here (same precedent as C2's smoke above). Usage:
rem out\t2139_c3_smoke.exe ^<model.sslm^>.
cl /nologo /std:c++20 /O2 /W4 /fp:precise /EHsc /Iinclude ^
	src\artifact.cpp src\sha256.cpp src\tokenizer.cpp src\model.cpp src\intmath.cpp src\silu_lut.cpp src\matmul.cpp src\proof_manifest.cpp src\trace_hook.cpp ^
	src\forward\checked_chain_funnel.cpp src\forward\forward_sites.cpp src\decode_digest.cpp ^
	src\sslm_abi.cpp ^
	src\damped_greedy_antilm.cpp src\damped_greedy_topk.cpp src\damped_greedy_phaseD.cpp ^
	tools\t2139_c3_smoke.cpp /Fo:out\t2139\ /Fe:out\t2139_c3_smoke.exe
if errorlevel 1 (
	popd & exit /b 1
)
if defined T2139_MODEL (
	out\t2139_c3_smoke.exe %T2139_MODEL%
	if errorlevel 1 ( popd & exit /b 1 )
) else (
	echo t2139_c3_smoke: built, NOT run ^(set T2139_MODEL=path\to\real.sslm to run^)
)
cl /nologo /std:c++20 /O2 /W4 /fp:precise /EHsc /Iinclude ^
	src\artifact.cpp src\sha256.cpp src\tokenizer.cpp src\model.cpp src\intmath.cpp src\silu_lut.cpp src\matmul.cpp src\proof_manifest.cpp src\trace_hook.cpp ^
	src\forward\checked_chain_funnel.cpp src\forward\forward_sites.cpp src\decode_digest.cpp ^
	src\sslm_abi.cpp ^
	src\damped_greedy_antilm.cpp src\damped_greedy_topk.cpp src\damped_greedy_phaseD.cpp ^
	tools\t2139_c3_smoke_negative.cpp /Fo:out\t2139\ /Fe:out\t2139_c3_smoke_negative.exe
if errorlevel 1 (
	popd & exit /b 1
)
if defined T2139_MODEL (
	out\t2139_c3_smoke_negative.exe %T2139_MODEL%
	if errorlevel 1 ( popd & exit /b 1 )
) else (
	echo t2139_c3_smoke_negative: built, NOT run ^(set T2139_MODEL=path\to\real.sslm to run^)
)

rem C4's own gate (design Sec9): sslm_prefill + sslm_decode_step through this ABI reproduces
rem RunGreedyDecodeLoop's own direct-call output bit-for-bit -- token sequence AND KV bytes,
rem checked against tests/t2138-abi-red-suite/fixture_common.h's own CpuOracleModel/
rem RunGreedyOracle construction (the suite's own already-reviewed oracle, reused rather than
rem re-derived). NOT auto-run here (same precedent as C2/C3's smokes above). Usage:
rem out\t2139_c4_oracle.exe ^<model.sslm^> [max_new_tokens].
cl /nologo /std:c++20 /O2 /W4 /fp:precise /EHsc /Iinclude /Itests\t2138-abi-red-suite ^
	src\artifact.cpp src\sha256.cpp src\tokenizer.cpp src\model.cpp src\intmath.cpp src\silu_lut.cpp src\matmul.cpp src\proof_manifest.cpp src\trace_hook.cpp ^
	src\forward\checked_chain_funnel.cpp src\forward\forward_sites.cpp src\decode_digest.cpp ^
	src\sslm_abi.cpp ^
	src\damped_greedy_antilm.cpp src\damped_greedy_topk.cpp src\damped_greedy_phaseD.cpp ^
	tools\t2139_c4_oracle.cpp /Fo:out\t2139\ /Fe:out\t2139_c4_oracle.exe
if errorlevel 1 (
	popd & exit /b 1
)
if defined T2139_MODEL (
	out\t2139_c4_oracle.exe %T2139_MODEL%
	if errorlevel 1 ( popd & exit /b 1 )
) else (
	echo t2139_c4_oracle: built, NOT run ^(set T2139_MODEL=path\to\real.sslm to run^)
)
cl /nologo /std:c++20 /O2 /W4 /fp:precise /EHsc /Iinclude ^
	src\artifact.cpp src\sha256.cpp src\tokenizer.cpp src\model.cpp src\intmath.cpp src\silu_lut.cpp src\matmul.cpp src\proof_manifest.cpp src\trace_hook.cpp ^
	src\forward\checked_chain_funnel.cpp src\forward\forward_sites.cpp src\decode_digest.cpp ^
	src\sslm_abi.cpp ^
	src\damped_greedy_antilm.cpp src\damped_greedy_topk.cpp src\damped_greedy_phaseD.cpp ^
	tools\t2139_c4_smoke_negative.cpp /Fo:out\t2139\ /Fe:out\t2139_c4_smoke_negative.exe
if errorlevel 1 (
	popd & exit /b 1
)
if defined T2139_MODEL (
	out\t2139_c4_smoke_negative.exe %T2139_MODEL%
	if errorlevel 1 ( popd & exit /b 1 )
) else (
	echo t2139_c4_smoke_negative: built, NOT run ^(set T2139_MODEL=path\to\real.sslm to run^)
)

rem C5's own Gate B smoke: saves a real sequence mid-generation, restores it into a fresh
rem handle, decodes to the next token on each, compares -- plus the hostile-blob rejections
rem (corrupted magic/model_hash/kv_precision) and the two-call sizing convention. NOT auto-run
rem here (same precedent as C2/C3's smokes above). Usage: out\t2139_c5_smoke.exe ^<model.sslm^>.
cl /nologo /std:c++20 /O2 /W4 /fp:precise /EHsc /Iinclude ^
	src\artifact.cpp src\sha256.cpp src\tokenizer.cpp src\model.cpp src\intmath.cpp src\silu_lut.cpp src\matmul.cpp src\proof_manifest.cpp src\trace_hook.cpp ^
	src\forward\checked_chain_funnel.cpp src\forward\forward_sites.cpp src\decode_digest.cpp ^
	src\sslm_abi.cpp ^
	src\damped_greedy_antilm.cpp src\damped_greedy_topk.cpp src\damped_greedy_phaseD.cpp ^
	tools\t2139_c5_smoke.cpp /Fo:out\t2139\ /Fe:out\t2139_c5_smoke.exe
if errorlevel 1 (
	popd & exit /b 1
)
if defined T2139_MODEL (
	out\t2139_c5_smoke.exe %T2139_MODEL%
	if errorlevel 1 ( popd & exit /b 1 )
) else (
	echo t2139_c5_smoke: built, NOT run ^(set T2139_MODEL=path\to\real.sslm to run^)
)
cl /nologo /std:c++20 /O2 /W4 /fp:precise /EHsc /Iinclude ^
	src\artifact.cpp src\sha256.cpp src\tokenizer.cpp src\model.cpp src\intmath.cpp src\silu_lut.cpp src\matmul.cpp src\proof_manifest.cpp src\trace_hook.cpp ^
	src\forward\checked_chain_funnel.cpp src\forward\forward_sites.cpp src\decode_digest.cpp ^
	src\sslm_abi.cpp ^
	src\damped_greedy_antilm.cpp src\damped_greedy_topk.cpp src\damped_greedy_phaseD.cpp ^
	tools\t2139_c5_smoke_negative.cpp /Fo:out\t2139\ /Fe:out\t2139_c5_smoke_negative.exe
if errorlevel 1 (
	popd & exit /b 1
)
if defined T2139_MODEL (
	out\t2139_c5_smoke_negative.exe %T2139_MODEL%
	if errorlevel 1 ( popd & exit /b 1 )
) else (
	echo t2139_c5_smoke_negative: built, NOT run ^(set T2139_MODEL=path\to\real.sslm to run^)
)

rem C6's own Gate B smoke: maps a real adapter, binds it to a decoding sequence, decodes,
rem releases (design's own stated C6 smoke shape) -- plus foreign-base mismatch, mid-token swap
rem rejection, and the product-scale lifecycle-guard cell (design Sec10 dim11) reached through
rem this smoke's own ordinary call order. NOT auto-run here (same precedent as C2/C3/C5's
rem smokes above). Usage: out\t2139_c6_smoke.exe ^<base.sslm^> ^<adapter.sslm^>
rem [foreign-base.sslm].
cl /nologo /std:c++20 /O2 /W4 /fp:precise /EHsc /Iinclude ^
	src\artifact.cpp src\sha256.cpp src\tokenizer.cpp src\model.cpp src\intmath.cpp src\silu_lut.cpp src\matmul.cpp src\proof_manifest.cpp src\trace_hook.cpp ^
	src\forward\checked_chain_funnel.cpp src\forward\forward_sites.cpp src\decode_digest.cpp ^
	src\sslm_abi.cpp ^
	src\damped_greedy_antilm.cpp src\damped_greedy_topk.cpp src\damped_greedy_phaseD.cpp ^
	tools\t2139_c6_smoke.cpp /Fo:out\t2139\ /Fe:out\t2139_c6_smoke.exe
if errorlevel 1 (
	popd & exit /b 1
)
rem Conductor observation (Claude/Bach/t2139-battery-varsunset-failure-2026-08-16.md): a chained
rem `if defined A if defined B (block) else (block)` binds its `else` to the SECOND `if` only --
rem when A is undefined, the whole compound statement is skipped without ever reaching the else,
rem so the announced-skip echo below never printed even on a healthy run. Sequential guard
rem instead: set a plain flag only when both are defined, then a single if/else on that flag.
set "c6_ready="
if defined T2139_MODEL if defined T2139_ADAPTER set "c6_ready=1"
if defined c6_ready (
	out\t2139_c6_smoke.exe %T2139_MODEL% %T2139_ADAPTER% %T2139_FOREIGN%
	if errorlevel 1 ( popd & exit /b 1 )
) else (
	echo t2139_c6_smoke: built, NOT run ^(set T2139_MODEL and T2139_ADAPTER to run^)
)
cl /nologo /std:c++20 /O2 /W4 /fp:precise /EHsc /Iinclude ^
	src\artifact.cpp src\sha256.cpp src\tokenizer.cpp src\model.cpp src\intmath.cpp src\silu_lut.cpp src\matmul.cpp src\proof_manifest.cpp src\trace_hook.cpp ^
	src\forward\checked_chain_funnel.cpp src\forward\forward_sites.cpp src\decode_digest.cpp ^
	src\sslm_abi.cpp ^
	src\damped_greedy_antilm.cpp src\damped_greedy_topk.cpp src\damped_greedy_phaseD.cpp ^
	tools\t2139_c6_smoke_negative.cpp /Fo:out\t2139\ /Fe:out\t2139_c6_smoke_negative.exe
if errorlevel 1 (
	popd & exit /b 1
)
rem Same chained-if-defined defect as C6's own smoke above -- same sequential-guard fix.
set "c6neg_ready="
if defined T2139_MODEL if defined T2139_ADAPTER if defined T2139_FOREIGN set "c6neg_ready=1"
if defined c6neg_ready (
	out\t2139_c6_smoke_negative.exe %T2139_MODEL% %T2139_ADAPTER% %T2139_FOREIGN%
	if errorlevel 1 ( popd & exit /b 1 )
) else (
	echo t2139_c6_smoke_negative: built, NOT run ^(set T2139_MODEL, T2139_ADAPTER, T2139_FOREIGN to run^)
)

rem C7's own self-contained smoke (design Sec9: no Gate A/B -- sslm_g5.h declares neither verb):
rem encode a fixed string, decode the result, compare, plus Forge W4's own incremental split-
rem boundary safety obligation (every possible split point reassembles identically). NOT
rem auto-run here (same precedent as C2/C3/C5/C6's smokes above). Usage:
rem out\t2139_c7_smoke.exe ^<model.sslm^>.
cl /nologo /std:c++20 /O2 /W4 /fp:precise /EHsc /Iinclude ^
	src\artifact.cpp src\sha256.cpp src\tokenizer.cpp src\model.cpp src\intmath.cpp src\silu_lut.cpp src\matmul.cpp src\proof_manifest.cpp src\trace_hook.cpp ^
	src\forward\checked_chain_funnel.cpp src\forward\forward_sites.cpp src\decode_digest.cpp ^
	src\sslm_abi.cpp ^
	src\damped_greedy_antilm.cpp src\damped_greedy_topk.cpp src\damped_greedy_phaseD.cpp ^
	tools\t2139_c7_smoke.cpp /Fo:out\t2139\ /Fe:out\t2139_c7_smoke.exe
if errorlevel 1 (
	popd & exit /b 1
)
rem T2139_MODEL_TOK (distinct from T2139_MODEL): C7's own verbs need a REAL bound tokenizer,
rem which the plain base artifact T2139_MODEL names for C2-C6 does not carry (adapter-compat
rem constrains T2139_MODEL to the exact artifact the adapter was compiled against, which has no
rem tokenizer section) -- set T2139_MODEL_TOK to a combined model+tokenizer .sslm (e.g. this
rem ticket's own tools/t2139_build_combined_fixture.py output) to run C7's own smokes/pins.
if defined T2139_MODEL_TOK (
	out\t2139_c7_smoke.exe %T2139_MODEL_TOK%
	if errorlevel 1 ( popd & exit /b 1 )
) else (
	echo t2139_c7_smoke: built, NOT run ^(set T2139_MODEL_TOK=path\to\a-model+tokenizer.sslm to run^)
)

rem Design commit 212de7742c's own same-round pin: a forced out-of-tokenizer-range id
rem ([tok_vocab, cfg_vocab)) fed to sslm_detokenize_stream is SSLM_TOKEN_ID_UNMAPPED, output/
rem state unperturbed, distinct from a plain SSLM_INVALID_ARGUMENT at id >= cfg_vocab. NOT
rem auto-run here (same precedent as this file's other real-artifact tools). Usage:
rem out\t2139_c7_unmapped_pin.exe ^<model.sslm^>.
cl /nologo /std:c++20 /O2 /W4 /fp:precise /EHsc /Iinclude ^
	src\artifact.cpp src\sha256.cpp src\tokenizer.cpp src\model.cpp src\intmath.cpp src\silu_lut.cpp src\matmul.cpp src\proof_manifest.cpp src\trace_hook.cpp ^
	src\forward\checked_chain_funnel.cpp src\forward\forward_sites.cpp src\decode_digest.cpp ^
	src\sslm_abi.cpp ^
	src\damped_greedy_antilm.cpp src\damped_greedy_topk.cpp src\damped_greedy_phaseD.cpp ^
	tools\t2139_c7_unmapped_pin.cpp /Fo:out\t2139\ /Fe:out\t2139_c7_unmapped_pin.exe
if errorlevel 1 (
	popd & exit /b 1
)
if defined T2139_MODEL_TOK (
	out\t2139_c7_unmapped_pin.exe %T2139_MODEL_TOK%
	if errorlevel 1 ( popd & exit /b 1 )
) else (
	echo t2139_c7_unmapped_pin: built, NOT run ^(set T2139_MODEL_TOK=path\to\a-model+tokenizer.sslm to run^)
)

rem T-2199 Phase D closing round, item 3, conductor's commission, 2026-08-20 -- see
rem Claude/Curie/t2199-phaseD-red-2026-08-20.md item 3's own "spec'd back" finding: a minimal
rem synthetic fixture passes the converter/verifier but is rejected by the runtime engine
rem construction, BuildEngineCache, for want of the full named-tensor set. This step generates
rem a COMPLETE, real, runnable synthetic .sslm -- every WGT1/BIA1/WSC1/ROP1/KVC1 tensor
rem BuildEngineCache requires -- reusing tools/reference_pipeline/pipeline.py's own "Sec11
rem fixture model," the same small transformer that suite's entire test estate already
rem exercises the Python reference forward pass against -- so the model-gated Phase D cells,
rem D2/D2a/D3, and the C1-discriminating pin, t2139_dim9_current_token_pin, run for REAL on a
rem bare tree, not merely SKIP. Regenerated fresh every run, S-HARDEN-5's hermetic-fixture
rem discipline, same as tools/sslm_pinned_calibration_fixture.py -- never committed as a binary.
rem Guarded, non-fatal if python is absent, this file's own established convention for optional
rem tooling, e.g. the CI checkers below -- absent python, this step is skipped and every
rem model-gated cell below reverts to its pre-existing SKIP-without-a-checkpoint behavior --
rem never a regression, only the pre-existing state. Placed HERE (not at file top): check_gpu_
rem guard_status_parity.py's own O11-gate check anchors on this file's FIRST "cl /nologo"
rem invocation by TEXT POSITION -- a cl invocation placed before that anchor (this block's own
rem sslm_verify build, below) would misdirect that unrelated check, so this whole block runs
rem after the real anchor instead. Absolute paths throughout (%CD%, not a bare "out\..." relative
rem path): tests\t2199-damped-greedy-red-suite\build_green_phaseD.bat -- the Phase D gate this
rem fixture provisions, further below -- cd /d's into its own directory before using the model
rem path it is passed, so a relative path set HERE would resolve wrong THERE.
if not exist out\t2199 mkdir out\t2199
where python >nul 2>nul
if not errorlevel 1 (
	python tools\_t2199_s8_synthetic_full_model_fixture.py out\t2199_s8_fixture.sslm out\t2243_s8_plain.sslm
	if errorlevel 1 (
		echo T-2199 S8 fixture generation FAILED -- model-gated Phase D cells and the C1 pin
		echo will SKIP below ^(same as if python/numpy were absent^), not silently pass.
	) else (
		if not defined T2199_PHASED_MODEL set T2199_PHASED_MODEL=%CD%\out\t2199_s8_fixture.sslm
		if not defined T2199_DIM9_MODEL set T2199_DIM9_MODEL=%CD%\out\t2199_s8_fixture.sslm
		rem T-2234 (SuperSLM 1.2.1): the same fixture with NO damped-greedy opt-in -- the
		rem pre-1.2 artifact shape the workspace-layout golden cells (dim7 C4/C5) read.
		if not defined T2138_MODEL_PLAIN set T2138_MODEL_PLAIN=%CD%\out\t2243_s8_plain.sslm
		echo T-2199 S8 fixture: out\t2199_s8_fixture.sslm generated -- T2199_PHASED_MODEL/
		echo T2199_DIM9_MODEL provisioned by default, override either to use a real checkpoint.
		cl /nologo /std:c++20 /O2 /W4 /fp:precise /EHsc /Iinclude ^
			src\artifact.cpp src\sha256.cpp src\tokenizer.cpp src\model.cpp src\intmath.cpp src\silu_lut.cpp src\matmul.cpp src\proof_manifest.cpp src\trace_hook.cpp ^
			src\forward\checked_chain_funnel.cpp src\forward\forward_sites.cpp src\decode_digest.cpp ^
			src\sslm_abi.cpp ^
			src\damped_greedy_antilm.cpp src\damped_greedy_topk.cpp src\damped_greedy_phaseD.cpp src\damped_greedy_phaseD_loop.cpp ^
			tools\sslm_verify.cpp /Fo:out\t2199\ /Fe:out\sslm_verify.exe
		if errorlevel 1 (
			echo sslm_verify build FAILED -- the S8 fixture's own independent-verifier confirmation
			echo is skipped ^(non-fatal^); the runtime pin below is this round's own primary gate.
		) else (
			out\sslm_verify.exe out\t2199_s8_fixture.sslm out\t2199_s8_fixture.sslm.manifest.json
			if errorlevel 1 (
				echo sslm_verify REJECTED the S8 fixture -- a real defect in the generated artifact,
				echo surfaced here rather than silently trusted just because the runtime pin below
				echo happens to accept it.
				popd & exit /b 1
			)
			echo sslm_verify: S8 fixture confirmed byte-valid.
		)
	)
) else (
	echo python not found on PATH -- skipping T-2199 S8 synthetic fixture generation; Phase D's
	echo model-gated cells and the C1 pin SKIP below unless T2199_PHASED_MODEL/T2139_MODEL are
	echo set to a real checkpoint by hand ^(non-fatal, matching this file's own convention^).
)

rem Design commit 9e2995f4e7's own same-round pin (Sec10 dim 9): a sequence saved resting
rem BETWEEN decode steps, restored, live and restored both driven one further step -- produced
rem tokens must be bit-identical. NOT auto-run here (same precedent as this file's other
rem real-artifact tools). Usage: out\t2139_dim9_current_token_pin.exe ^<model.sslm^>.
cl /nologo /std:c++20 /O2 /W4 /fp:precise /EHsc /Iinclude ^
	src\artifact.cpp src\sha256.cpp src\tokenizer.cpp src\model.cpp src\intmath.cpp src\silu_lut.cpp src\matmul.cpp src\proof_manifest.cpp src\trace_hook.cpp ^
	src\forward\checked_chain_funnel.cpp src\forward\forward_sites.cpp src\decode_digest.cpp ^
	src\sslm_abi.cpp ^
	src\damped_greedy_antilm.cpp src\damped_greedy_topk.cpp src\damped_greedy_phaseD.cpp ^
	tools\t2139_dim9_current_token_pin.cpp /Fo:out\t2139\ /Fe:out\t2139_dim9_current_token_pin.exe
if errorlevel 1 (
	popd & exit /b 1
)
rem T-2199 Phase D closing round, item 3: T2139_MODEL (a real checkpoint) wins if the caller set
rem it; otherwise falls back to T2199_DIM9_MODEL (the S8 synthetic fixture provisioned above,
rem this file's own item-3 block) -- this is THE C1-discriminating pin the conductor's own
rem acceptance criterion names by name ("goes red under the C1 scratch-reversion check"), so a
rem bare-tree run must not silently SKIP it the way every OTHER T2139_MODEL-gated tool in this
rem file still does (unchanged, out of this round's own scope).
set DIM9_MODEL=%T2139_MODEL%
if not defined DIM9_MODEL set DIM9_MODEL=%T2199_DIM9_MODEL%
if defined DIM9_MODEL (
	out\t2139_dim9_current_token_pin.exe %DIM9_MODEL%
	if errorlevel 1 ( popd & exit /b 1 )
) else (
	echo t2139_dim9_current_token_pin: built, NOT run ^(set T2139_MODEL=path\to\real.sslm to run^)
)

rem N2 pin (Claude/Poirot/2c18dab-t2139-abi-build-review.md Sec6.3): an ODD max_chunk_budget --
rem the real S-FREEZE prompt's own token count -- through a full prefill+decode, under
rem src/sslm_abi.cpp's own compiled-in alignment asserts at wide_logits/rms_wide's point of use.
rem Usage: out\t2139_n2_odd_budget_smoke.exe ^<model.sslm^>.
cl /nologo /std:c++20 /O2 /W4 /fp:precise /EHsc /Iinclude ^
	src\artifact.cpp src\sha256.cpp src\tokenizer.cpp src\model.cpp src\intmath.cpp src\silu_lut.cpp src\matmul.cpp src\proof_manifest.cpp src\trace_hook.cpp ^
	src\forward\checked_chain_funnel.cpp src\forward\forward_sites.cpp src\decode_digest.cpp ^
	src\sslm_abi.cpp ^
	src\damped_greedy_antilm.cpp src\damped_greedy_topk.cpp src\damped_greedy_phaseD.cpp ^
	tools\t2139_n2_odd_budget_smoke.cpp /Fo:out\t2139\ /Fe:out\t2139_n2_odd_budget_smoke.exe
if errorlevel 1 (
	popd & exit /b 1
)
if defined T2139_MODEL (
	out\t2139_n2_odd_budget_smoke.exe %T2139_MODEL%
	if errorlevel 1 ( popd & exit /b 1 )
) else (
	echo t2139_n2_odd_budget_smoke: built, NOT run ^(set T2139_MODEL=path\to\real.sslm to run^)
)

rem N3 pin (Claude/Poirot/2c18dab-t2139-abi-build-review.md Sec6.3): sslm_model_map returns
rem SSLM_ALLOCATION_FAILED, not UB, when a genuine std::bad_alloc crosses its own try/catch --
rem via the SAME test-only fault-injection seam (tests/support/bad_alloc_injection.h) S-HARDEN-7's
rem own population already trusts. Needs SUPERSLM_ENABLE_BAD_ALLOC_INJECTION + /Itests, matching
rem the test-injection build's own convention (see the C5 block above). Usage:
rem out\t2139_n3_bad_alloc_pin.exe ^<model.sslm^>.
cl /nologo /std:c++20 /O2 /W4 /fp:precise /EHsc /Iinclude /Itests /DSUPERSLM_ENABLE_BAD_ALLOC_INJECTION ^
	src\artifact.cpp src\sha256.cpp src\tokenizer.cpp src\model.cpp src\intmath.cpp src\silu_lut.cpp src\matmul.cpp src\proof_manifest.cpp src\trace_hook.cpp ^
	src\forward\checked_chain_funnel.cpp src\forward\forward_sites.cpp src\decode_digest.cpp ^
	src\sslm_abi.cpp ^
	src\damped_greedy_antilm.cpp src\damped_greedy_topk.cpp src\damped_greedy_phaseD.cpp ^
	tools\t2139_n3_bad_alloc_pin.cpp /Fo:out\t2139\ /Fe:out\t2139_n3_bad_alloc_pin.exe
if errorlevel 1 (
	popd & exit /b 1
)
if defined T2139_MODEL (
	out\t2139_n3_bad_alloc_pin.exe %T2139_MODEL%
	if errorlevel 1 ( popd & exit /b 1 )
) else (
	echo t2139_n3_bad_alloc_pin: built, NOT run ^(set T2139_MODEL=path\to\real.sslm to run^)
)

rem D-SLM3464 pin (Claude/Vitruvius/t2133-layer1-c-abi-design-2026-08-16.md Sec6, fold
rem 2026-08-17 second pass): sslm_model_map/sslm_adapter_map return SSLM_ARTIFACT_REJECTED, not
rem SSLM_ALLOCATION_FAILED and not UB, when a throw NOT derived from std::exception (ForeignFault)
rem crosses the real SslmModel::Load-fronted call path -- the one class WrapBadAllocContract's own
rem narrowing does not intercept. Same seam/build convention as the N3 pin above. Usage:
rem out\t2139_d3464_foreignfault_pin.exe ^<model.sslm^> [adapter.sslm].
cl /nologo /std:c++20 /O2 /W4 /fp:precise /EHsc /Iinclude /Itests /DSUPERSLM_ENABLE_BAD_ALLOC_INJECTION ^
	src\artifact.cpp src\sha256.cpp src\tokenizer.cpp src\model.cpp src\intmath.cpp src\silu_lut.cpp src\matmul.cpp src\proof_manifest.cpp src\trace_hook.cpp ^
	src\forward\checked_chain_funnel.cpp src\forward\forward_sites.cpp src\decode_digest.cpp ^
	src\sslm_abi.cpp ^
	src\damped_greedy_antilm.cpp src\damped_greedy_topk.cpp src\damped_greedy_phaseD.cpp ^
	tools\t2139_d3464_foreignfault_pin.cpp /Fo:out\t2139\ /Fe:out\t2139_d3464_foreignfault_pin.exe
if errorlevel 1 (
	popd & exit /b 1
)
rem Sequential-guard form (see the C6-smoke fix above) -- the adapter arg is OPTIONAL for this
rem pin (sslm_model_map's own cell runs with T2139_MODEL alone), so this is a plain single-var
rem gate, not a chained one; wired the same way regardless, for consistency.
if defined T2139_MODEL (
	out\t2139_d3464_foreignfault_pin.exe %T2139_MODEL% %T2139_ADAPTER%
	if errorlevel 1 ( popd & exit /b 1 )
) else (
	echo t2139_d3464_foreignfault_pin: built, NOT run ^(set T2139_MODEL=path\to\real.sslm to run; optionally also T2139_ADAPTER=path\to\real-adapter.sslm^)
)

rem F2 pin item 2 (Claude/Brunel/t2139-abi-build-2026-08-16.md Sec22, Claude/Poirot/
rem 3bcbe43-t2139-fourth-confirmation-review.md S3): std::length_error also narrows to
rem SSLM_ALLOCATION_FAILED through SslmModel::Load's own WrapBadAllocContract narrowing. Committed
rem at beb2355 with NO build recipe until this round (S3's own named finding) -- wired here on the
rem N3/D-SLM3464 pins' own convention (SUPERSLM_ENABLE_BAD_ALLOC_INJECTION + /Itests).
cl /nologo /std:c++20 /O2 /W4 /fp:precise /EHsc /Iinclude /Itests /DSUPERSLM_ENABLE_BAD_ALLOC_INJECTION ^
	src\artifact.cpp src\sha256.cpp src\tokenizer.cpp src\model.cpp src\intmath.cpp src\silu_lut.cpp src\matmul.cpp src\proof_manifest.cpp src\trace_hook.cpp ^
	src\forward\checked_chain_funnel.cpp src\forward\forward_sites.cpp src\decode_digest.cpp ^
	src\sslm_abi.cpp ^
	src\damped_greedy_antilm.cpp src\damped_greedy_topk.cpp src\damped_greedy_phaseD.cpp ^
	tools\t2139_f2_length_error_pin.cpp /Fo:out\t2139\ /Fe:out\t2139_f2_length_error_pin.exe
if errorlevel 1 (
	popd & exit /b 1
)
if defined T2139_MODEL (
	out\t2139_f2_length_error_pin.exe %T2139_MODEL%
	if errorlevel 1 ( popd & exit /b 1 )
) else (
	echo t2139_f2_length_error_pin: built, NOT run ^(set T2139_MODEL=path\to\real.sslm to run^)
)

rem F2 pin item 1 (Claude/Brunel/t2139-abi-build-2026-08-16.md Sec22, Claude/Poirot/
rem 3bcbe43-t2139-fourth-confirmation-review.md S3): a MECHANISM CHECK, not a real-path pin (its
rem own header comment states this plainly -- CatchAllocationFailure has internal linkage and no
rem real call path today reaches its own catch(...) arm with an unnarrowed exception type). No
rem real .sslm artifact needed -- self-contained, always built AND run. Committed at beb2355 with
rem NO build recipe until this round (S3's own named finding).
cl /nologo /std:c++20 /O2 /W4 /EHsc tools\t2139_f2_catchall_construction_pin.cpp /Fo:out\t2139\ /Fe:out\t2139_f2_catchall_construction_pin.exe
if errorlevel 1 (
	popd & exit /b 1
)
out\t2139_f2_catchall_construction_pin.exe
if errorlevel 1 (
	popd & exit /b 1
)

rem D-SLM3466's owed pin (Claude/Poirot/3bcbe43-t2139-fourth-confirmation-review.md S2/S3):
rem proves a non-allocation cause in sslm_model_map's/sslm_adapter_map's own POST-Load
rem construction step (BuildEngineCache/PopulateAdapterFromView) returns SSLM_ARTIFACT_REJECTED --
rem via the NEW site-specific arming slot (tests/support/bad_alloc_injection.h), independent of the
rem plain slot SslmModel::Load's own *Impl consults, closing the isolation gap the N3 pin's own
rem header comment named. Same seam/build convention as the N3/D-SLM3464 pins above.
cl /nologo /std:c++20 /O2 /W4 /fp:precise /EHsc /Iinclude /Itests /DSUPERSLM_ENABLE_BAD_ALLOC_INJECTION ^
	src\artifact.cpp src\sha256.cpp src\tokenizer.cpp src\model.cpp src\intmath.cpp src\silu_lut.cpp src\matmul.cpp src\proof_manifest.cpp src\trace_hook.cpp ^
	src\forward\checked_chain_funnel.cpp src\forward\forward_sites.cpp src\decode_digest.cpp ^
	src\sslm_abi.cpp ^
	src\damped_greedy_antilm.cpp src\damped_greedy_topk.cpp src\damped_greedy_phaseD.cpp ^
	tools\t2139_d3466_postload_region_pin.cpp /Fo:out\t2139\ /Fe:out\t2139_d3466_postload_region_pin.exe
if errorlevel 1 (
	popd & exit /b 1
)
if defined T2139_MODEL (
	out\t2139_d3466_postload_region_pin.exe %T2139_MODEL% %T2139_ADAPTER%
	if errorlevel 1 ( popd & exit /b 1 )
) else (
	echo t2139_d3466_postload_region_pin: built, NOT run ^(set T2139_MODEL=path\to\real.sslm to run; optionally also T2139_ADAPTER=path\to\real-adapter.sslm^)
)

rem S-FREEZE-EXAMPLE (design Sec9's own gate, D-SLM13): builds with NO internal include path --
rem /Iinclude ONLY, the frozen public header, no /Itests, no /Isrc-internal, no Unreal, no
rem test-harness affordance. Uses every verb C1-C7 ship, real text I/O (D-SLM3452), explicit
rem pool capacity (D-SLM3454). NOT auto-run here (needs a real .sslm this build does not assume
rem exists on every machine); built so it is ready. Usage:
rem out\t2139_sfreeze_example.exe ^<model.sslm^> "prompt".
cl /nologo /std:c++20 /O2 /W4 /fp:precise /EHsc /Iinclude ^
	src\artifact.cpp src\sha256.cpp src\tokenizer.cpp src\model.cpp src\intmath.cpp src\silu_lut.cpp src\matmul.cpp src\proof_manifest.cpp src\trace_hook.cpp ^
	src\forward\checked_chain_funnel.cpp src\forward\forward_sites.cpp src\decode_digest.cpp ^
	src\sslm_abi.cpp ^
	src\damped_greedy_antilm.cpp src\damped_greedy_topk.cpp src\damped_greedy_phaseD.cpp ^
	tools\t2139_sfreeze_example.cpp /Fo:out\t2139\ /Fe:out\t2139_sfreeze_example.exe
if errorlevel 1 (
	popd & exit /b 1
)
rem S-FREEZE needs a REAL bound tokenizer (real text in, real text out, D-SLM3452) -- same
rem T2139_MODEL_TOK distinction as C7's own smokes/pins above.
if defined T2139_MODEL_TOK (
	out\t2139_sfreeze_example.exe %T2139_MODEL_TOK% "The old wizard said"
	if errorlevel 1 ( popd & exit /b 1 )
) else (
	echo t2139_sfreeze_example: built, NOT run ^(set T2139_MODEL_TOK=path\to\a-model+tokenizer.sslm to run^)
)

rem T-2132 (G5-2): the StandardsDocument Sec5.4 real-workload demonstration -- maps a real G5
rem fixture (tools/t2132_build_g5_fixture.py's own output: a real weights+tokenizer .sslm plus
rem a real SchemaMasks section), binds a compiled schema, prefills a real tokenized prompt, and
rem decodes under the mask. Same NO-internal-include-path posture as S-FREEZE-EXAMPLE above --
rem /Iinclude only. NOT auto-run here (needs a real G5 fixture this build does not assume exists
rem on every machine); built so it is ready. Usage:
rem out\t2132_g5_smoke.exe ^<g5-fixture.sslm^> "prompt" [schema_name].
cl /nologo /std:c++20 /O2 /W4 /fp:precise /EHsc /Iinclude ^
	src\artifact.cpp src\sha256.cpp src\tokenizer.cpp src\model.cpp src\intmath.cpp src\silu_lut.cpp src\matmul.cpp src\proof_manifest.cpp src\trace_hook.cpp ^
	src\forward\checked_chain_funnel.cpp src\forward\forward_sites.cpp src\decode_digest.cpp ^
	src\sslm_abi.cpp ^
	src\damped_greedy_antilm.cpp src\damped_greedy_topk.cpp src\damped_greedy_phaseD.cpp ^
	tools\t2132_g5_smoke.cpp /Fo:out\t2139\ /Fe:out\t2132_g5_smoke.exe
if errorlevel 1 (
	popd & exit /b 1
)
rem G5 needs a real fixture WITH a compiled SchemaMasks section (tools/t2132_build_g5_fixture.py) --
rem a distinct env var from T2139_MODEL_TOK, since an ordinary combined model+tokenizer artifact
rem has no schema to bind.
if defined T2132_G5_FIXTURE (
	out\t2132_g5_smoke.exe %T2132_G5_FIXTURE% "Book me with Rin, Thursday afternoon."
	if errorlevel 1 ( popd & exit /b 1 )
) else (
	echo t2132_g5_smoke: built, NOT run ^(set T2132_G5_FIXTURE=path\to\a-g5-fixture.sslm to run^)
)

rem T-2147 (design Sec15.2, D-SLM3481/D-SLM3488): the chunk-batched prefill proof-cell +
rem speedup-measurement tool -- per-size (a), boundary-split (b) bit-identity against the real
rem G5 fixture, plus --speedup for the actual streamed-weight-bandwidth measurement and
rem --dump-blob=PATH for cross-source-tree cell (c) comparisons. Same NOT-auto-run posture as
rem t2132_g5_smoke.exe above (needs a real G5 fixture). Usage:
rem out\t2147_chunk_batched_pins.exe ^<g5-fixture.sslm^> "prompt" [--speedup] [--boundary-sweep=K] [--dump-blob=PATH].
if not exist out\t2147 mkdir out\t2147
cl /nologo /std:c++20 /O2 /W4 /fp:precise /EHsc /Iinclude ^
	src\artifact.cpp src\sha256.cpp src\tokenizer.cpp src\model.cpp src\intmath.cpp src\silu_lut.cpp src\matmul.cpp src\proof_manifest.cpp src\trace_hook.cpp ^
	src\forward\checked_chain_funnel.cpp src\forward\forward_sites.cpp src\decode_digest.cpp ^
	src\sslm_abi.cpp ^
	src\damped_greedy_antilm.cpp src\damped_greedy_topk.cpp src\damped_greedy_phaseD.cpp ^
	tools\t2147_chunk_batched_pins.cpp /Fo:out\t2147\ /Fe:out\t2147_chunk_batched_pins.exe
if errorlevel 1 (
	popd & exit /b 1
)
if defined T2132_G5_FIXTURE (
	out\t2147_chunk_batched_pins.exe %T2132_G5_FIXTURE% "Book me with Rin, Thursday afternoon."
	if errorlevel 1 ( popd & exit /b 1 )
) else (
	echo t2147_chunk_batched_pins: built, NOT run ^(set T2132_G5_FIXTURE=path\to\a-g5-fixture.sslm to run^)
)

rem T-2132 (G5-5, Brunel, design Sec6 Slot G5-5 -- REQUIRED 1.0 gate, D-SLM3443): the executed
rem CPU-vs-GPU parity proof for schema-constrained decode and jump-forward. Needs the SAME real
rem G5 fixture as G5-2's own smoke above plus real D3D12 hardware -- built here (both the CPU ABI
rem and the GPU-1.0 TUs, per this tool's own #include list) but NOT auto-run (matching every
rem other real-artifact GPU tool's own precedent above: B2/B3/.../t2124). Usage:
rem out\t2132_g5_gpu_parity.exe ^<g5-fixture.sslm^> "prompt" [schema_name] [num_decode_steps].
rem NOTE: split into three tool TUs (t2132_g5_gpu_parity.cpp / _cpu.cpp / _gpu.cpp) -- sslm_abi.h's
rem `sslm_status` and gpu_1p0.h's `SslmGpuStatus` share several enumerator spellings at GLOBAL C
rem scope (SSLM_OK, SSLM_ADAPTER_MODEL_MISMATCH, SSLM_MODEL_HAS_LIVE_SEQUENCES,
rem SSLM_TOKEN_ID_OUT_OF_RANGE, SSLM_RESTORE_MODEL_MISMATCH) -- a real collision this tool is the
rem first target to trip (needing both APIs at once), fixed by keeping the two headers out of the
rem same TU rather than editing either (both declaration-pinned). /Itools resolves the shared POD
rem header (t2132_g5_gpu_parity_shared.h) between the three.
if not exist out\t2132g5gpu mkdir out\t2132g5gpu
cl /nologo /std:c++20 /O2 /W4 /fp:precise /EHsc /Iinclude /Itools ^
	src\artifact.cpp src\sha256.cpp src\tokenizer.cpp src\model.cpp src\intmath.cpp src\silu_lut.cpp src\matmul.cpp src\proof_manifest.cpp src\trace_hook.cpp ^
	src\forward\checked_chain_funnel.cpp src\forward\forward_sites.cpp src\decode_digest.cpp ^
	src\sslm_abi.cpp ^
	src\damped_greedy_antilm.cpp src\damped_greedy_topk.cpp src\damped_greedy_phaseD.cpp ^
	src\gpu\superslm_gpu.cpp src\gpu\gpu_1p0.cpp ^
	tools\t2132_g5_gpu_parity.cpp tools\t2132_g5_gpu_parity_cpu.cpp tools\t2132_g5_gpu_parity_gpu.cpp ^
	/Fo:out\t2132g5gpu\ /Fe:out\t2132_g5_gpu_parity.exe ^
	/link d3d12.lib dxgi.lib dxguid.lib
if errorlevel 1 (
	popd & exit /b 1
)
rem T-2132 (G5-5) build log finding, Claude/Brunel/t2132-g5-build-2026-08-16.md: real, reproducible,
rem dispatch-budget-independent CPU/GPU divergence found starting at decode step 19 of a real,
rem content-varying 24-step generation -- localized (walk-state replay) to the underlying GPU
rem layer-loop substrate's own numerics, NOT G5-5's masking/walk-state/jump-forward mechanism
rem (which stays state-for-state identical to the CPU oracle through and past the divergence).
rem Auto-run here at 18 steps -- the largest length this build has real, executed, clean evidence
rem for -- so this step gives a real PASS/FAIL signal instead of a known-red one; the diverging
rem >18-step run is documented, not silently dropped, in the build log above.
if defined T2132_G5_FIXTURE (
	out\t2132_g5_gpu_parity.exe %T2132_G5_FIXTURE% "Book me with Rin, Thursday afternoon." shopkeeper_intent_extraction 18
	if errorlevel 1 ( popd & exit /b 1 )
) else (
	echo t2132_g5_gpu_parity: built, NOT run ^(set T2132_G5_FIXTURE=path\to\a-g5-fixture.sslm to run^)
)

rem C1 pin (Claude/Poirot/9bc9ec6-t2132-g5-arc-review.md): sslm_seq_restore's own restored
rem dfa_walk_state, bounded against the resolved schema's state_count -- out-of-range and
rem sentinel-as-bound-state both rejected, against a real corrupted save-blob from the real G5
rem fixture. Usage: out\t2132_c1_restore_walk_state_pin.exe ^<g5-fixture.sslm^> [schema_name].
cl /nologo /std:c++20 /O2 /W4 /fp:precise /EHsc /Iinclude ^
	src\artifact.cpp src\sha256.cpp src\tokenizer.cpp src\model.cpp src\intmath.cpp src\silu_lut.cpp src\matmul.cpp src\proof_manifest.cpp src\trace_hook.cpp ^
	src\forward\checked_chain_funnel.cpp src\forward\forward_sites.cpp src\decode_digest.cpp ^
	src\sslm_abi.cpp ^
	src\damped_greedy_antilm.cpp src\damped_greedy_topk.cpp src\damped_greedy_phaseD.cpp ^
	tools\t2132_c1_restore_walk_state_pin.cpp /Fo:out\t2139\ /Fe:out\t2132_c1_restore_walk_state_pin.exe
if errorlevel 1 (
	popd & exit /b 1
)
if defined T2132_G5_FIXTURE (
	out\t2132_c1_restore_walk_state_pin.exe %T2132_G5_FIXTURE%
	if errorlevel 1 ( popd & exit /b 1 )
) else (
	echo t2132_c1_restore_walk_state_pin: built, NOT run ^(set T2132_G5_FIXTURE=path\to\a-g5-fixture.sslm to run^)
)

rem C2 pin (Claude/Poirot/9bc9ec6-t2132-g5-arc-review.md): sslm_seq_set_schema/
rem sslm_prefix_set_schema now reject a schema handle bound to a DIFFERENT model. Needs a SECOND,
rem independently-mapped artifact (the 0.5B fixture -- no schema section needed on that side, the
rem schema comes from model A). Usage: out\t2132_c2_cross_model_schema_pin.exe
rem ^<g5-fixture-1.5b.sslm^> ^<plain-0.5b.sslm^> [schema_name].
cl /nologo /std:c++20 /O2 /W4 /fp:precise /EHsc /Iinclude ^
	src\artifact.cpp src\sha256.cpp src\tokenizer.cpp src\model.cpp src\intmath.cpp src\silu_lut.cpp src\matmul.cpp src\proof_manifest.cpp src\trace_hook.cpp ^
	src\forward\checked_chain_funnel.cpp src\forward\forward_sites.cpp src\decode_digest.cpp ^
	src\sslm_abi.cpp ^
	src\damped_greedy_antilm.cpp src\damped_greedy_topk.cpp src\damped_greedy_phaseD.cpp ^
	tools\t2132_c2_cross_model_schema_pin.cpp /Fo:out\t2139\ /Fe:out\t2132_c2_cross_model_schema_pin.exe
if errorlevel 1 (
	popd & exit /b 1
)
if defined T2132_G5_FIXTURE (
if defined T2132_MODEL_0P5B (
	out\t2132_c2_cross_model_schema_pin.exe %T2132_G5_FIXTURE% %T2132_MODEL_0P5B%
	if errorlevel 1 ( popd & exit /b 1 )
) else (
	echo t2132_c2_cross_model_schema_pin: built, NOT run ^(set T2132_MODEL_0P5B=path\to\a-0.5b-plain.sslm too^)
)
) else (
	echo t2132_c2_cross_model_schema_pin: built, NOT run ^(set T2132_G5_FIXTURE=path\to\a-g5-fixture.sslm to run^)
)

rem S7 pin (Claude/Poirot/9bc9ec6-t2132-g5-arc-review.md): sslm_seq_set_schema's own guard now
rem matches its shipped header's freshness condition (current_token == -1), not merely
rem dfa_walk_state -- a first-time bind attempted after real unconstrained generation rejects.
rem Usage: out\t2132_s7_set_schema_freshness_pin.exe ^<g5-fixture.sslm^> [schema_name].
cl /nologo /std:c++20 /O2 /W4 /fp:precise /EHsc /Iinclude ^
	src\artifact.cpp src\sha256.cpp src\tokenizer.cpp src\model.cpp src\intmath.cpp src\silu_lut.cpp src\matmul.cpp src\proof_manifest.cpp src\trace_hook.cpp ^
	src\forward\checked_chain_funnel.cpp src\forward\forward_sites.cpp src\decode_digest.cpp ^
	src\sslm_abi.cpp ^
	src\damped_greedy_antilm.cpp src\damped_greedy_topk.cpp src\damped_greedy_phaseD.cpp ^
	tools\t2132_s7_set_schema_freshness_pin.cpp /Fo:out\t2139\ /Fe:out\t2132_s7_set_schema_freshness_pin.exe
if errorlevel 1 (
	popd & exit /b 1
)
if defined T2132_G5_FIXTURE (
	out\t2132_s7_set_schema_freshness_pin.exe %T2132_G5_FIXTURE%
	if errorlevel 1 ( popd & exit /b 1 )
) else (
	echo t2132_s7_set_schema_freshness_pin: built, NOT run ^(set T2132_G5_FIXTURE=path\to\a-g5-fixture.sslm to run^)
)

rem S2/D-SLM3476 pin (design Sec14.1, Claude/Poirot/9bc9ec6-t2132-g5-arc-review.md): out_tokens[i]
rem == -2 reserved for a schema-bound walk with no legal continuation -- found by BFS'ing the
rem REAL compiled reference schema for a genuinely dead-end state, not a hand-built artifact.
rem Usage: out\t2132_s2_dead_end_sentinel_pin.exe ^<g5-fixture.sslm^> [schema_name].
cl /nologo /std:c++20 /O2 /W4 /fp:precise /EHsc /Iinclude ^
	src\artifact.cpp src\sha256.cpp src\tokenizer.cpp src\model.cpp src\intmath.cpp src\silu_lut.cpp src\matmul.cpp src\proof_manifest.cpp src\trace_hook.cpp ^
	src\forward\checked_chain_funnel.cpp src\forward\forward_sites.cpp src\decode_digest.cpp ^
	src\sslm_abi.cpp ^
	src\damped_greedy_antilm.cpp src\damped_greedy_topk.cpp src\damped_greedy_phaseD.cpp ^
	tools\t2132_s2_dead_end_sentinel_pin.cpp /Fo:out\t2139\ /Fe:out\t2132_s2_dead_end_sentinel_pin.exe
if errorlevel 1 (
	popd & exit /b 1
)
if defined T2132_G5_FIXTURE (
	out\t2132_s2_dead_end_sentinel_pin.exe %T2132_G5_FIXTURE%
	if errorlevel 1 ( popd & exit /b 1 )
) else (
	echo t2132_s2_dead_end_sentinel_pin: built, NOT run ^(set T2132_G5_FIXTURE=path\to\a-g5-fixture.sslm to run^)
)

rem S4 pin (Claude/Poirot/9bc9ec6-t2132-g5-arc-review.md): restores a discriminating mechanism
rem for the dimension-1 leak guard T-2132's own DrawBlock zero-fill made unobservable -- a
rem test-only pool-peek hook (src\sslm_abi.cpp, sslm_g5_test_only_peek_kv_block_bytes) bypasses
rem DrawBlock's zero-fill to observe ReturnBlock's own poison-fill directly.
rem Usage: out\t2132_s4_leak_guard_mutation_pin.exe ^<g5-fixture.sslm^>.
cl /nologo /std:c++20 /O2 /W4 /fp:precise /EHsc /Iinclude ^
	src\artifact.cpp src\sha256.cpp src\tokenizer.cpp src\model.cpp src\intmath.cpp src\silu_lut.cpp src\matmul.cpp src\proof_manifest.cpp src\trace_hook.cpp ^
	src\forward\checked_chain_funnel.cpp src\forward\forward_sites.cpp src\decode_digest.cpp ^
	src\sslm_abi.cpp ^
	src\damped_greedy_antilm.cpp src\damped_greedy_topk.cpp src\damped_greedy_phaseD.cpp ^
	tools\t2132_s4_leak_guard_mutation_pin.cpp /Fo:out\t2139\ /Fe:out\t2132_s4_leak_guard_mutation_pin.exe
if errorlevel 1 (
	popd & exit /b 1
)
if defined T2132_G5_FIXTURE (
	out\t2132_s4_leak_guard_mutation_pin.exe %T2132_G5_FIXTURE%
	if errorlevel 1 ( popd & exit /b 1 )
) else (
	echo t2132_s4_leak_guard_mutation_pin: built, NOT run ^(set T2132_G5_FIXTURE=path\to\a-g5-fixture.sslm to run^)
)

rem M4 pin (design Sec7.3, D-SLM3486, Claude/Vitruvius/t2133-layer1-c-abi-design-2026-08-16.md):
rem forced_token_count now survives save/restore under the 'SSB2' blob format, proven for a
rem sequence with REAL jump-forward-admitted tokens against the real fixture; also pins the
rem 'SSB1'-rejection half (a hand-built pre-fold-shaped blob rejects outright on the magic check).
rem Usage: out\t2132_m4_forced_token_count_pin.exe ^<g5-fixture.sslm^> [schema_name].
cl /nologo /std:c++20 /O2 /W4 /fp:precise /EHsc /Iinclude ^
	src\artifact.cpp src\sha256.cpp src\tokenizer.cpp src\model.cpp src\intmath.cpp src\silu_lut.cpp src\matmul.cpp src\proof_manifest.cpp src\trace_hook.cpp ^
	src\forward\checked_chain_funnel.cpp src\forward\forward_sites.cpp src\decode_digest.cpp ^
	src\sslm_abi.cpp ^
	src\damped_greedy_antilm.cpp src\damped_greedy_topk.cpp src\damped_greedy_phaseD.cpp ^
	tools\t2132_m4_forced_token_count_pin.cpp /Fo:out\t2139\ /Fe:out\t2132_m4_forced_token_count_pin.exe
if errorlevel 1 (
	popd & exit /b 1
)
if defined T2132_G5_FIXTURE (
	out\t2132_m4_forced_token_count_pin.exe %T2132_G5_FIXTURE%
	if errorlevel 1 ( popd & exit /b 1 )
) else (
	echo t2132_m4_forced_token_count_pin: built, NOT run ^(set T2132_G5_FIXTURE=path\to\a-g5-fixture.sslm to run^)
)

rem T-2113 (B9, design Sec10 B9/Sec11 dim7): the compile-the-declared-interface check
rem (tests\t2112-gpu-1p0-red-suite\interface_probe\build_probe.bat), promoted from a T-2111
rem strike instrument to a standing suite fixture (design Sec10 B9) and wired here as a real
rem build-time gate -- a hand-mutated guard or a broken declaration in gpu_1p0.h fails THIS
rem build, not merely a separately-run script nobody invokes. Non-fatal-if-absent would defeat
rem the point (unlike the Python CI checkers below, which degrade gracefully because they are
rem genuinely optional tooling); the probe's own toolchain (cl, already required above) is not
rem optional, so a failure here fails the whole build. build_probe.bat's own `cd /d %HEREDIR%`
rem persists into THIS script (batch `call` shares the process, not a subshell) -- wrapped in
rem its own pushd/popd so this script's own relative paths below still resolve from repo root.
pushd .
call tests\t2112-gpu-1p0-red-suite\interface_probe\build_probe.bat
set probe_ec=%errorlevel%
popd
if not %probe_ec%==0 (
	popd & exit /b 1
)

rem T-2199 Phase D review fix S8 (Claude/Poirot/7a3b10a-t2199-phaseD-review.md), 2026-08-20:
rem "nothing gates Phase D" -- neither build_green.bat (Phase A/C) nor build_green_phaseD.bat
rem (Phase D) was ever CALLED from this file, so a hand-mutated guard or a broken declaration in
rem either suite failed nothing but a script nobody in this build's own gate invokes. Wired here
rem as a real build-time gate, the SAME shape as the T-2112 probe immediately above (a suite
rem that fails to build/link/run fails THIS build, not a separately-run script). Both suites'
rem own scripts `cd /d` internally (matching build_probe.bat's own documented behavior) --
rem wrapped in the same pushd/popd discipline. Set T2199_PHASED_MODEL=path\to\real.sslm to
rem exercise D2/D2a/D3's own product cells and t2139_dim9_current_token_pin's own C1 pin below;
rem absent it, D1 still runs and gates (needs no model), and D2/D2a/D3 SKIP honestly (their own
rem script's own established convention, matching every T2139_MODEL-gated pin elsewhere in this
rem file) rather than failing for want of a checkpoint this CI environment may not have.
pushd .
call tests\t2199-damped-greedy-red-suite\build_green.bat
set phaseAC_ec=%errorlevel%
popd
if not %phaseAC_ec%==0 (
	popd & exit /b 1
)
pushd .
call tests\t2199-damped-greedy-red-suite\build_green_phaseD.bat %T2199_PHASED_MODEL%
set phaseD_ec=%errorlevel%
popd
if not %phaseD_ec%==0 (
	popd & exit /b 1
)

rem D-SLM3798 suite-wiring closure: the T-2138 ABI suite previously had a correct executable
rem driver in every cell but root build only compiled unrelated header probes. S8's hermetic
rem fixture is now sized to the suite's real token/context population (vocab 128, context 64),
rem so build, link, and execute all eleven cells here. Adapter/tokenizer/foreign-model cells
rem activate when their optional T2138_* artifact variables are supplied and otherwise report
rem counted skips; every model-only cell runs on every ordinary build.
pushd .
call tests\t2138-abi-red-suite\run_green.bat %T2199_PHASED_MODEL%
set t2138_ec=%errorlevel%
popd
if not %t2138_ec%==0 (
	popd & exit /b 1
)

rem T-2314 (gate-reachability sweep, Claude/Brunel/t2314-gate-reachability-2026-08-27.md): three
rem suites -- this one, T-2178, and T-2296 below -- had zero references anywhere in this file,
rem CMakeLists.txt, or .github/workflows/tests.yml, found by an independent hand sweep and
rem reproduced by tools/ci/check_tests_have_build_recipe.py (wired further down). Wired here the
rem same shape as the T-2112/T-2199/T-2138 suites above: tests\t2018-slora-serial\ has three
rem independent build scripts, none of which run their own output (they only compile) -- so each
rem is called and the resulting .exe is run directly too, matching this file's own
rem out\superslm_tests.exe precedent below. All three exes need no external artifact; verified
rem this round fully green (75/0, 37/0, 21/0).
pushd .
call tests\t2018-slora-serial\build.bat
set t2018_offline_ec=%errorlevel%
popd
if not %t2018_offline_ec%==0 (
	popd & exit /b 1
)
tests\t2018-slora-serial\t2018_offline_red.exe
if errorlevel 1 (
	popd & exit /b 1
)
pushd .
call tests\t2018-slora-serial\build_b0b.bat
set t2018_b0b_ec=%errorlevel%
popd
if not %t2018_b0b_ec%==0 (
	popd & exit /b 1
)
tests\t2018-slora-serial\t2029_b0b_red.exe
if errorlevel 1 (
	popd & exit /b 1
)
pushd .
call tests\t2018-slora-serial\build_b2.bat
set t2018_b2_ec=%errorlevel%
popd
if not %t2018_b2_ec%==0 (
	popd & exit /b 1
)
tests\t2018-slora-serial\t2029_b2_red.exe
if errorlevel 1 (
	popd & exit /b 1
)

rem T-2314: tests\t2178-gpu-batched-prefill-red-suite\ wired calling its own single entry point,
rem build_red_suite.bat, exactly as that script's own header and tail document its exit-code
rem contract: exit 2 is "an unexpected error class was found" (a real regression -- a COMPILE
rem ERROR or an UNEXPECTED ERROR CLASS); exit 1 covers BOTH "RED BY LINK" (a probe/bound symbol
rem not yet wired) and "LINKED CLEAN" (a cell that needs none) -- both named explicitly in that
rem script's own tail as acceptable, non-failing states, UNLIKE tests\t2296-fp-free-open-red-
rem suite\ below, where exit 1 IS a real, gating failure. Verified this round:
rem kT2169TdrSafeMaxChunkTokens (src\gpu\superslm_gpu.cpp, defined since T-2180) is no longer the
rem unresolved symbol the suite's own header comments describe -- all five cells now link clean,
rem and running each resulting .exe directly (obj\<cell>.exe, no arguments) shows 4 checks/0
rem failures/5 skips, every skip for want of --model1p5b=PATH / --g5fixture=PATH, matching this
rem suite's own documented shape (D-SLM3432: no GPU CI runner -- "run directly"). Its own
rem build_red_suite.bat never runs the cells itself (only compiles/links them), so this gate
rem proves the suite still compiles and links against the current engine; it does NOT re-verify
rem cell runtime pass/fail, which needs a real model artifact and a GPU this environment does not
rem assume -- the per-cell .exe runs above were a one-time manual verification this round, not
rem part of this gate, matching every other wired suite's one-call convention.
pushd .
call tests\t2178-gpu-batched-prefill-red-suite\build_red_suite.bat
set t2178_ec=%errorlevel%
popd
if %t2178_ec%==2 (
	popd & exit /b 1
)

rem T-2314: tests\t2296-fp-free-open-red-suite\ -- this arc's own pin, six cells each
rem mutation-proved to discriminate, reverting any one of which left every automated gate green
rem until now. Wired here the same shape as the T-2138/T-2199 suites above: both of its own
rem scripts run and check every cell's real pass/fail via their own exit code (0=GREEN, 1=RED,
rem 2=COMPILE/LINK ERROR -- unlike tests\t2178-gpu-batched-prefill-red-suite\ above, 1 here IS a
rem real, gating failure). Verified this round: both scripts GREEN (build_link_red.bat's own
rem dim7_capacity_red.cpp cell alone reported 162 checks/0 failures at wiring time, T-2314;
rem Cells I and J landed two commits later in this same range and brought that cell's own count
rem to 9,532 checks/0 failures, T-2321/T-2322; Cell F's own round-budget derivation (T-2324, per
rem Claude/Poirot/t-2323-fold28-confirmation-review-2026-08-27.md S1) raised it again to 12,484
rem checks/0 failures -- corrected here rather than left at either stale figure.
rem build_liveness_red.bat's two cells report 2/0 and 1/0 checks, the rest SKIP for want of a
rem full-model artifact this environment does not assume).
pushd .
call tests\t2296-fp-free-open-red-suite\build_link_red.bat
set t2296_link_ec=%errorlevel%
popd
if not %t2296_link_ec%==0 (
	popd & exit /b 1
)
pushd .
call tests\t2296-fp-free-open-red-suite\build_liveness_red.bat
set t2296_liveness_ec=%errorlevel%
popd
if not %t2296_liveness_ec%==0 (
	popd & exit /b 1
)

rem T-2326/T-2333 (Curie): tests\t2296-fp-free-open-red-suite\test_check_fp_free_scan.py -- the red
rem suite for design Sec4.1's own deciding instrument (Sec7 dimension 11's fourteen commissioning
rem populations, plus ci_gate's own enforcement contract: Claude/Vitruvius/
rem t2265-superslm-fp-free-open-design-2026-08-24.md, as ratified through fold round 33). The
rem instrument itself is a Python CI tool that does not exist yet anywhere in this tree
rem (tests\ci\check_fp_free_scan.py) -- a separate, substantial deliverable T-2296's own dim11_
rem guard_red.cpp named as a specification-scale follow-on rather than authoring it as a C++ TU
rem (Python test code for a Python instrument, not C++ test code against an existing/soon-to-exist
rem C++ API). Every cell in this suite independently verifies its own fixture first -- a fresh
rem compile or byte-level synthesis, plus a raw capstone decode or disasm-text parse that is NOT the
rem absent instrument's own logic (fp_scan_common.py's own docstring) -- before asserting the
rem instrument's documented verdict, so every cell fails today for exactly one reason (the
rem check_fp_free_scan import failing), never a broken fixture: 25 cells across the fourteen
rem populations plus ci_gate's own absent-report leg, 0 unexpected exceptions, 0 fixture-verification
rem failures, verified this ticket's own session (25 of 25 red for the stated reason, real
rem toolchains -- MSVC cl.exe/ml64.exe/dumpbin, MSVC's own AArch64 cross-compiler, clang/clang++,
rem capstone -- every one confirmed present and working, 0 skips). T-2333 repoints this suite at the
rem RATIFIED production contract (scan_object(path, isa) -> ScanResult{object_format, refuse,
rem unclassified_bytes, verdicts}; ci_gate; enumerate_scan_targets; diagnostic_walk;
rem derive_core_sources; build_call_graph/flagged_symbols/diagnostic_fp_report) after T-2326's own
rem proposed contract (D-SLM4825) went through the planner and both review rungs and moved
rem (D-SLM4826/D-SLM4827/D-SLM4830/D-SLM4834) -- populations one and two are now real reproductions
rem of TE-32's own construction (that scope was denied to T-2326, granted to T-2333), and population
rem six is graded end-to-end against build_call_graph/flagged_symbols/diagnostic_fp_report, not a
rem TU-name-list diff. THIS IS WHY build.bat NOW EXITS NONZERO ON A CLEAN CHECKOUT, identically in
rem shape to how dim6_determinism_red.cpp/dim7_contract_red.cpp gated this same build red before
rem src\detail\int_hash.h existed (T-2296) -- see this file's own "NON-ZERO-EXIT PATHS" note below,
rem extended with this path (4). Guarded on pytest being importable by whatever `python` resolves to
rem on PATH: an environment with python but no pytest SKIPs this step loudly, non-fatal, rather than
rem failing the build on a missing dev-only dependency -- distinct from the instrument itself being
rem absent, which IS fatal, per this suite's own red-first charter.
rem
rem T-2342 (Curie): design fold round 34 (Dan's ruling: every leg proven before v1.3.0) closed five
rem specification gaps (a)-(e) against Sec4.1 -- ci_gate_corpus (a REJECT anywhere fails the job,
rem D-SLM4856), scan_object's own corpus_symbols index (D-SLM4857), the external-edge default-deny
rem policy ruled explicit (D-SLM4858), enumerate_scan_targets's duplicate-stem refusal (D-SLM4859),
rem and the BFloat16 exclusion on both ISAs (D-SLM4860) -- plus Poirot's own Critical C1 (movsd
rem missing from the movement allowlist) and Popper's own clang/ELF/x86-64 finding (the padding
rem detector recognises only repeated 0x90/0xCC, not clang/GCC's multi-byte NOP forms). 16 new cells
rem (12 genuinely red today; 4 confirmatory controls, stated as such rather than forced red) realize
rem all of this, none of it built yet -- see this file's own "NON-ZERO-EXIT PATHS" note below for the
rem exact list, and Claude/Curie/t2342-fp-scan-instrument-red-suite-fold34-2026-08-27.md for the full
rem disposition. Suite total: 41 cells (25 T-2326/T-2333 + 16 T-2342), 39 executed by the gating run
rem below (2 pre-existing deselects, unrelated to T-2342).
rem
rem T-2338 (Brunel): tests\ci\check_fp_free_scan.py BUILT to the ratified contract above -- this
rem paragraph describes the suite's own state AT THAT BUILD (25 cells, before T-2342 added 16 more).
rem 23 of the suite's 25 cells were graded through it and green at that time; two were DESELECTED
rem from the gating run, each for a stated reason recorded here and in
rem Claude/Brunel/t2338-fp-scan-instrument-build-*.md (the
rem build record) -- neither is "moved" (the test file is untouched):
rem   - test_population_08_real_corpus_whole_sweep: its OWN docstring states it is "Not gradable by
rem     this single pytest cell even once the instrument exists -- a full-corpus build-and-scan is
rem     CI-scale, not a unit cell," and its own body calls pytest.fail() UNCONDITIONALLY after
rem     verifying the 17-file population is real -- it cannot pass under pytest by construction,
rem     for any instrument. Its own real occupant is
rem     tests\ci\run_fp_free_scan_real_corpus.py, invoked separately below.
rem   - test_population_10_arm_differential_historical: asserts `not result.refuse` on
rem     pop10_arm_site.cpp (BuildMerges/DedupNames/BodyDivide/BodyConvertCompare) compiled whole via
rem     MSVC's own AArch64 cross-compiler. Design Sec4.1's OWN executed fold-round-9 finding, for
rem     this identical construction, is that the WHOLE OBJECT correctly REFUSES ("the same object
rem     drops to 28 of 3933 unclassified... the scan still correctly REFUSES... filed as its own
rem     residual") -- fold round 10's own "unclassified=0" reading is expressly scoped to the four
rem     named target symbols in isolation, not the whole object scan_object() itself performs.
rem     Reproduced fresh this ticket: the object's own `$LN22` label (COFF storage class STATIC,
rem     type 0 -- confirmed by direct read of the compiled object's own symbol table) marks the
rem     start of the identical FNV-1a hash-constant literal fold round 9 disclosed; Sec4.1's own
rem     ratified text (fold round 14) witnesses no STATIC/LABEL symbol and grounds no data extent
rem     without relocation/unwind metadata, so this reader -- built to that text -- REFUSES on the
rem     whole object exactly as the design's own probe did. This is a found tension between the red
rem     suite's own assertion and the ratified design text's own disclosed, executed finding for the
rem     identical construction, filed as an open finding rather than resolved by editing either side.
rem T-2343 (Brunel): fold round 34's own five specification gaps (a)-(e), Poirot's Critical C1
rem (movsd), Popper's clang/ELF/x86-64 finding, and the routed review findings (S1/S2/S4/S5/S6/
rem M1-M6/O3-O6) are ALL BUILT into tests\ci\check_fp_free_scan.py this round. Re-executed this
rem session: 39 passed, 2 deselected (the identical two named below, unrelated to fold round 34) --
rem every one of T-2342's own 12 newly-red cells now passes for a genuine grading reason, and no
rem cell was edited to reach that state. Population ten remains an open, disclosed tension (below);
rem population eight remains ungradable by any pytest cell by its own docstring, unchanged.
rem S5 (Poirot): a `--deselect` in a batch-file string is silent the moment it stops matching (an
rem unmatched nodeid deselects nothing, with no error -- this file's own prior session confirmed
rem that directly). Poirot's own recommended structural fix (`xfail(strict=True)` inside the test
rem file) needs an edit to test_check_fp_free_scan.py, which is Curie's own writable domain, not
rem this build round's -- routed back rather than done here. What this round adds, inside this
rem build's own writable scope: a `--collect-only` count check, immediately below, that FAILS THE
rem BUILD the moment the number of deselected cells is not exactly 2 -- so a third cell silently
rem ceasing to match (or a fourth deselect quietly added) is a loud build failure, not a permanently
rem invisible batch-file string.
rem T-2347 (Curie): fold round 35 (Claude/Vitruvius/t2265-fold35-delta-manifest.md) -- design
rem Sec4.1's two new refusal contracts (derive_core_sources raises CoreSourcesDerivationError,
rem D-SLM4887; ci_gate_corpus closes its own vacuous-True side, D-SLM4888), Sec7 dim 11's six new
rem populations (sixteenth-twenty-first, D-SLM4889: self-zeroing vxorps/vxorpd; check (C)'s
rem relocation-resolve-then-classify carve-out, D-SLM4886; the diagnostic surface's fail-closed
rem behavior; refuse=True on an unrecognised ISA/format; per-section local_starts keying; the
rem empty-extent byte charge), and the three findings Poirot's 8a28460-t2344 casebook routes to the
rem test author (M4: fp_scan_common.compile_cl_release's own hardcoded flag duplication -- FIXED IN
rem PLACE this ticket, since it is this suite's own test-helper file; O4: populations eight and ten
rem migrated to xfail(strict=True), S5's own structural fix, now actually landed; O5:
rem run_fp_free_scan_real_corpus._msvc_target_flags's silent truncation on an embedded ), documented
rem as a disclosed residual, not fixed -- no ratified correction exists for it yet). 20 new cells:
rem 7 newly RED for the newly-specified behaviour (check (C)'s carve-out must-accept; all three
rem derive_core_sources raise reproductions; both ci_gate_corpus vacuous-True cells; the empty-extent
rem multi-charge cell) and 13 confirmatory/green (the prior fix round's own remedies for
rem vxorps/xorps, the diagnostic surface's MissingDisassemblyError, refuse=True on an unrecognised
rem ISA/format, and the per-section local_starts keying were all already correct but carried ZERO
rem cells anywhere in this suite -- Poirot's own S2: reverting all four together left the gating
rem suite at 39 passed, unchanged). Re-executed this session: 61 collected, 7 failed (the newly-red
rem cells above), 52 passed, 2 xfailed (populations eight/ten, now marked in the test file itself
rem rather than only in this batch file's own --deselect strings) -- reproduced twice, stable.
rem NOTE, corrected T-2348 (Brunel), O4's own build-round half (8a28460-t2344-fp-scan-fix-round-
rem confirmation.md): fold round 35's own text above described the --deselect flags and the
rem --collect-only count guard as still present, routed back for removal once xfail(strict=True)
rem was actually landed. It IS landed (T-2347, populations eight/ten, test_check_fp_free_scan.py
rem :766,929) -- both flags and the guard are REMOVED below this round: xfail(strict=True) now
rem provides the identical drift protection structurally, inside the test file itself (a deselect
rem nodeid that stops matching, or a swapped/added deselect, was only ever detectable via the
rem --collect-only count this guard computed; a cell that stops being an expected failure now fails
rem the gating run directly, the same way any other genuine regression does). The gating run below
rem now COLLECTS AND RUNS all 61 of this suite's own cells (39 pre-existing + 20 fold round 35, T-
rem 2347) -- expected 59 passed, 2 xfailed, 0 failed, reproduced this session (see this file's own
rem NON-ZERO-EXIT PATHS note, item 4, below, for the up-to-date accounting).
rem T-2366 (Curie), D-SLM5001 -- the invocation below now targets the whole SUITE DIRECTORY, not
rem only test_check_fp_free_scan.py: a new sibling file, test_ci_gate_wiring.py (D-SLM5001 item 6,
rem CI reachability -- this ticket's own writable scope names "tests/t2296-fp-free-open-red-suite/
rem and any new test file it needs," never tools/ci/, which is where this repo's own analogous
rem check_tests_have_build_recipe.py precedent otherwise lives), would sit uncollected under the
rem prior single-file invocation. This is fold round 39's own red suite (design Sec4.1/Sec5.4/
rem Sec5.5/Sec7 dim 11 as amended by fold round 39, routed by T-2364's strike and T-2365's coverage
rem audit, D-SLM5001): 9 newly RED for the newly-specified behaviour the fold wrote as accomplished
rem but never built (the gate reading check (C) at all; check (A)'s eight-mnemonic bitwise-family
rem widening, including population sixteen's own reconciled collision; the p/vp-prefix rule's
rem vitality census; D-SLM4359's seven switch-jump-table symbols; the CI wiring, two cells), 4 newly
rem green (already-correct regression guards carrying zero cells until this ticket: the gate's own
rem must-reject control, the p/vp deny-list's own mutation proof, and scan_build_output.py's
rem fail-closed membership discipline, both legs). See this file's own NON-ZERO-EXIT PATHS note,
rem item 4, below, for the up-to-date accounting.
rem T-2380 (Curie), design Sec4.1/Sec5.4/Sec5.5/Sec7 dim 11 as amended by fold round 42 -- the red
rem suite for the archive-based ship gate's own reader, classifier, and composition stages (T-2378's
rem strike found 16 of 18 archive-member positions invisible to fold round 41's own acceptance
rem form). Two new sibling files, both collected by the same directory-level invocation below:
rem test_archive_gate.py (thirty-seventh through forty-sixth populations: magic/header validation,
rem member classification, zero-object refusal, the even-byte padding rule, the notrack/shrd-shld/
rem vextract-vinsert classifier corrections, Mach-O REFUSE-not-crash, and scan_object's corrected
rem signature) and test_archive_composition.py (the forty-seventh population, corrected to resolving
rem power zero over membership, and the forty-eighth -- membership swept at resolving power dn = 1
rem across all eighteen real archive-member positions, adopting T-2378's own fp_at_k.lib/SKIPPER_k
rem construction). None of the archive-reading or archive-composition surface these two files test
rem is built yet at this ticket's own pinned commit (b2b7aef): check_fp_free_scan.py has no
rem iterate_archive_members/enumerate_archive_objects, and scan_build_output.py's find_target_
rem objects still globs a directory, never an archive. Every cell asserting that not-yet-built
rem behaviour is marked xfail(strict=True) in the test file itself, so this round's own 51 new
rem cells (16 passed -- fixture-verification pins and controls already true today; 1 skipped -- a
rem documented model gap, capstone 5.0.7 never renders a "data16" mnemonic prefix, see the test
rem file's own docstring; 34 xfailed -- genuinely red-unimplemented) do not fail the gating run
rem below. Re-executed this session: 143 collected (92 prior + 51 new), 105 passed, 1 skipped, 37
rem xfailed, 0 failed -- reproduced twice, stable. The object-directory gate two paragraphs below
rem stays green and untouched (D-SLM5047's own mutation proof is unrevisited): 17 objects, 1950
rem ACCEPT, 0 REJECT, 0 REFUSE, exit 0, confirmed this session against the identical real corpus
rem this suite's own real_build_dir fixture builds.
rem T-2381 (Brunel), design Sec4.1/Sec5.4/Sec5.5/Sec7 dim 11 as amended by fold round 42 -- the
rem archive reader (iterate_archive_members/enumerate_archive_objects/MalformedArchiveError/
rem ArchiveHasNoObjectsError), scan_object's corrected signature (corpus_symbols AND data, both),
rem the four ELF/GCC classifier corrections (notrack strip, shrd/shld, the full sixteen-mnemonic
rem vextract/vinsert lane-movement family), and scan_build_output.py's own corpus retargeting from
rem the object-directory glob to the archive (find_target_archive, checked BEFORE the directory;
rem the directory path is kept, used only when no archive exists at any candidate location, solely
rem because six pre-existing cells in this suite's own test_check_fp_free_scan.py build a bare
rem <target>.dir layout with no archive at all -- every real CI leg always produces an archive, so
rem that path is inert on every real leg) ARE ALL BUILT this round. 34 of the 37 xfail(strict=True)
rem cells the two new files above carried now pass for the built reason and their markers are
rem removed -- no cell's own assertion was edited; two new pins close production changes this round
rem landed with no existing cell (the four ELF/GCC corrections' combined effect against the real
rem GCC-built archive, 523 ACCEPT/0 REJECT/0 REFUSE up from 505/18/0 pre-correction; the malformed-
rem archive-exits-2 composition path, five parametrized cases). One stale literal-count regression
rem guard in test_check_fp_free_scan.py (test_check_a_p_vp_structural_accept_census_and_violation)
rem is rebaselined 555/101/454 -> 571/117/454, the exact +16 shift the vextract/vinsert widening
rem produces -- the identical shape that file's own D-SLM5009a comment already documents once, not
rem a weakened assertion. Re-executed this session: 149 collected (143 prior + 6 new), 145 passed,
rem 1 skipped, 3 xfailed, 0 failed -- the 3 remaining xfailed and the 1 skipped are pre-existing,
rem unrelated to this round (D-SLM5009b's own open question, two long-standing population 8/10
rem tensions, the data16-jmp model gap). tests\ci unaffected at 377 passed. The object-directory
rem gate's own real-corpus reading (D-SLM5047, two paragraphs above) is reproduced exactly by the
rem archive path now built: 17 objects, 1950 ACCEPT, 0 REJECT, 0 REFUSE, exit 0, against the
rem identical real corpus, now read via out\t2368_fp_scan_corpus_build\Release\superslm.lib rather
rem than the retired directory glob. Full engine regression (fresh CMake Release build,
rem BuildTools-pinned toolset): 34213 checks, 0 failures, unchanged. See
rem Claude/Brunel/t2381-archive-gate-build-2026-08-28.md for the full accounting.
rem T-2385 (Brunel), design Sec4.1 fold round 43 (D-SLM5099-D-SLM5103), closing T-2382's own
rem FIX-THEN-SHIP review of T-2381's build (S1/S2/S3/S4, M1/M2). S1/S4: the retained
rem object-directory fallback in scan_build_output.py's main() is REMOVED, not kept and graded --
rem main() now searches find_target_archive's own four candidate locations and, finding none,
rem prints the search and exits 2; the object-directory scan is never called. The stated
rem six-cell justification for keeping the fallback measured one cell at source (T-2382 S4);
rem that one cell (test_gate_must_not_fail_on_a_check_c_only_reject) and its must-reject sibling
rem (test_gate_still_fails_on_a_genuine_check_ab_violation) now build a real lib.exe archive for
rem their own scratch build via archive_fixtures.run_lib_exe, so neither reads a corpus the
rem corrected driver would refuse to open. S2: design Sec7 dim 11's fortieth population, retired
rem at fold round 41 on a premise the built driver falsifies (main() DOES open a directory, the
rem two reader functions alone do not), is restored at the design (D-SLM5099) and built here as
rem two live cells in test_archive_gate.py -- a clean object directory and an FP-carrying one,
rem both with no archive present, both required to exit 2, because the directory's own content
rem must have zero effect on a missing-archive disposition. S3: the nine present-tense
rem "not yet built" claims T-2382 found recurring on this same suite (D-SLM5018's own class,
rem structurally invisible to check_present_tense_defect_comments.py's severity-label-only
rem pattern) are corrected to current truth in test_archive_composition.py and
rem test_archive_gate.py, and the class is closed structurally: check_present_tense_defect_
rem comments.py gains a second, independent check (find_stale_unbuilt_claim/find_stale_
rem unbuilt_claims/scan_unbuilt_claims) that fails when a file in tests/t2296-fp-free-open-red-
rem suite/ carries a not-yet-built-shaped phrase with no live @pytest.mark.xfail marker anywhere
rem in it -- validated (before any fix was applied) against the real historical population at
rem this ticket's own starting commit 7a77a07: both files flag; both are clean after the fix.
rem Its own test pin, test_check_present_tense_unbuilt_class.py, lives in tests/t2296-fp-free-
rem open-red-suite/ rather than beside the module, since this ticket's writable scope names the
rem check module alone, not its existing test file. M1/M2: conftest.py's docstring corrected --
rem the archive, not the object directory, is the real ship gate's own production entry point
rem since T-2381/T-2385; archive_fixtures.py's docstring corrected -- fp_carrier.cpp's function
rem body is byte-identical to the adversary's probe, a 15-line adoption header differs (verified
rem by diff this session). Five counts re-run this session, all from D:\SuperSLM\.worktrees\
rem t2348-build: `pytest tests/t2296-fp-free-open-red-suite -q` 180 collected (149 baseline + 2
rem pop40 cells + 29 in the new present-tense pin), 176 passed, 1 skipped, 3 xfailed, 0 failed --
rem every one of the 149 baseline cells still passes; `pytest tests/ci -q` 377 passed, unchanged;
rem the real COFF archive (D:/SuperSLM/.worktrees/t2367-bld/Release/superslm.lib) 17 objects, 1950
rem ACCEPT, 0 REJECT, 0 REFUSE, exit 0, unchanged; the real ELF archive
rem (D:/SuperSLM/.worktrees/elf-leg/libsuperslm.a) 17 objects, 523 ACCEPT, 0 REJECT, 0 REFUSE,
rem exit 0, unchanged; full engine regression (fresh CMake Release build, BuildTools-pinned
rem toolset) 34213 checks, 0 failures, unchanged -- no C++ production source is touched this
rem round. Fail-closed behavior confirmed by direct execution: a synthetic build directory
rem carrying real, uncompiled-but-present objects under <target>.dir and no archive at any
rem candidate location now exits 2 with the search printed, where the pre-fix driver would have
rem scanned the directory and exited 0 or 1 depending on its contents. See
rem Claude/Brunel/t2385-archive-gate-fix-round-2026-08-28.md for the full accounting.
pushd .
set t2326_scan_ec=0
rem T-2348 (Brunel): initialized here, OUTSIDE every nested if-block below, so the sentinel is
rem correctly in place however far this block gets (python absent, pytest absent, or a real run) --
rem a sentinel set only INSIDE the innermost block would read as empty/undefined from outside it on
rem every path that never reaches that block, which the later "was it actually run" check below
rem would misread as a captured (empty) exit code rather than "never ran at all."
set t2343_runner_ec=not-run
where python >nul 2>nul
if not errorlevel 1 (
	python -c "import pytest" >nul 2>nul
	if not errorlevel 1 (
		python -m pytest tests\t2296-fp-free-open-red-suite -q --basetemp out\pytest-t2296
		if errorlevel 1 (
			set t2326_scan_ec=1
		)
		rem Population eight's own real occupant -- a whole-corpus build-and-scan of the real 17-file
		rem SUPERSLM_CORE_SOURCES, reported (never gating: a REFUSE/REJECT on the real corpus is the
		rem design's own disclosed, unclosed residual, not a build regression -- see the script's own
		rem module docstring). M4 (Poirot): non-gating means reported, not unobserved -- the runner's
		rem own exit code is captured and surfaced; it stays non-fatal to this build (a genuine
		rem infrastructure failure inside the runner is loud in its own printed output either way).
		rem T-2348 (Brunel), 8a28460-t2344's own S1: the fold-34 remedy echoed %errorlevel% INSIDE
		rem this same parenthesized block, which expands every %VAR% reference in the block ONCE when
		rem the block is first PARSED, before any line in it runs -- so the echo always reported the
		rem ERRORLEVEL left by the preceding `where`/`if` test, never the runner's own real exit code
		rem (confirmed by direct execution this session: "runner exited 0" printed here while the
		rem process genuinely exited 3, in the identical two-level nesting this file uses). This file
		rem does not use `setlocal enabledelayedexpansion` (see this file's own note near line 540)
		rem and this fix does not introduce it for the whole script's own single setlocal scope --
		rem `call set VAR=%%errorlevel%%` is the standard escape: CALL re-parses its own argument line
		rem as a fresh command, so the doubled %% (a literal % after THIS block's one parse pass)
		rem becomes %errorlevel% again in that second pass and is expanded THEN, against the real,
		rem current value the python command just left, not the value frozen at this block's own
		rem parse time. The value is captured into a variable and echoed only AFTER the block closes
		rem (below), the same read-after-close discipline t2326_scan_ec already uses. The sentinel
		rem itself is initialized once, above, outside every nested block (see that comment).
		python tests\ci\run_fp_free_scan_real_corpus.py
		call set t2343_runner_ec=%%errorlevel%%
	) else (
		echo pytest not installed for this Python -- skipping test_check_fp_free_scan.py ^(non-fatal^)
	)
) else (
	echo python not found on PATH -- skipping test_check_fp_free_scan.py ^(non-fatal^)
)
popd
if not "%t2343_runner_ec%"=="not-run" (
	echo run_fp_free_scan_real_corpus.py exited %t2343_runner_ec% ^(non-gating; see the script's own module docstring^)
)
rem T-2371 (Brunel), D-SLM5018 O1: the pytest run above, when it reaches
rem test_check_fp_free_scan.py's `real_build_dir`-dependent cells, causes
rem conftest.py's own session fixture to configure and build a real
rem `superslm` CMake target at out\t2368_fp_scan_corpus_build (D-SLM5008) --
rem the IDENTICAL object layout the ship gate's own production driver,
rem scan_build_output.py, reads. Before this round, that driver was never
rem invoked here at all: build.bat could pass locally while the real ship
rem gate -- checks (A)/(B) alone, the driver CI actually runs -- would have
rem failed, an asymmetry the code review filed as O1. If the fixture
rem produced that build directory, this build now ALSO gates on the same
rem driver CI runs against it; if the fixture skipped (no VS 2022 install
rem found, D-SLM5018 O2) or was never reached, this step is silently
rem skipped rather than treated as a failure, matching the fixture's own
rem stated disposition ("no cell in this suite tests whether superslm
rem builds").
set t2371_gate_ec=0
if exist "out\t2368_fp_scan_corpus_build" (
	where python >nul 2>nul
	if not errorlevel 1 (
		python tests\ci\scan_build_output.py --build-dir out\t2368_fp_scan_corpus_build --target superslm --isa x86-64
		if errorlevel 1 (
			set t2371_gate_ec=1
		)
	)
)
if not %t2371_gate_ec%==0 (
	echo scan_build_output.py FAILED against the real corpus this build's own red suite produced -- this IS the ship gate CI runs.
	popd & exit /b 1
)
if not %t2326_scan_ec%==0 (
	popd & exit /b 1
)

rem NON-ZERO-EXIT PATHS (O2, Claude/Poirot/aea6116-t2139-seventh-confirmation-review.md; RoPE
rem baseline retired T-2153, Claude/Curie/t2153-rope-fix-2026-08-17.md): this build's own
rem non-zero-exit paths, named so "build.bat exits 1" is never read as "no code defect" without
rem checking WHICH path fired --
rem   1. out\superslm_tests.exe's own exit code, below -- the suite is expected FULLY GREEN
rem      (34184 checks, 0 failures at T-2153's close; the prior RoPE@k=3/6/7 "pre-existing
rem      baseline" of three permanently-red cells was a cell-construction defect in
rem      TestT2019_B11_SequenceLayerStateComplete_Rope itself, not a product defect --
rem      root-caused and fixed T-2153 -- so this build no longer carries a disclosed, accepted
rem      failure count; ANY nonzero exit here is a real regression).
rem   2. tools\ci\check_abi_header_inventory.py, further down -- environment-dependent: a real
rem      identifier missing its Sec8 inventory line (a genuine content defect) OR, before T-2142's
rem      S1 fix, the records repo's mtime-selected copy resolving to one with no '## 8.' section
rem      (an apparatus defect, not a code defect -- now a SKIP, exit 0, never this path).
rem   3. tools\ci\gate_c_third_tu_can_fail_probe.py, above -- before T-2142's S3 fix, ANY dirty
rem      working tree aborted this path regardless of what was dirty or why (an apparatus defect,
rem      not a code defect); now compares before/after and only fires on a real regression.
rem   4. tests\t2296-fp-free-open-red-suite\test_check_fp_free_scan.py, immediately above
rem      (T-2326/T-2333/T-2338/T-2342/T-2343) -- T-2343 (Brunel) BUILT fold round 34's own five
rem      specification gaps, Poirot's Critical C1 (movsd), Popper's clang/ELF/x86-64 finding, and
rem      the routed review findings. Re-executed this round: 39 passed, 2 deselected -- all 12 of
rem      T-2342's own newly-red cells now pass for a genuine grading reason, and the 4 confirmatory
rem      cells are unregressed. The gating run above still deselects exactly the same two cells named
rem      in the comment above it (population eight, never gradable by a pytest cell by its own
rem      docstring; population ten, the found tension against the ratified design text's own
rem      disclosed fold-9 finding, still open -- see Claude/Brunel/
rem      t2343-fp-scan-instrument-fix-round-2026-08-27.md for the full accounting) -- a `--collect-
rem      only` count guard immediately above this pytest invocation fails the build the moment that
rem      count is not exactly 2, per S5's own structural fix (a full `xfail(strict=True))` still
rem      needs an edit to this test file, out of this build round's own writable scope, and is routed
rem      back to whoever next touches it). ANY nonzero exit from the 39 cells this run actually
rem      executes is a real regression.
rem      T-2347 (Curie): fold round 35's own two new refusal contracts and six new Coverage Model
rem      populations (sixteenth-twenty-first) are AUTHORED, not built, this round -- 7 newly RED for
rem      the newly-specified behaviour (D-SLM4886/4887/4888), 13 confirmatory/green (D-SLM4889's
rem      remedies, already correct, previously uncovered). THIS CHANGES WHAT "a real regression"
rem      MEANS UNTIL THE NEXT BUILD ROUND: the gating pytest invocation above deselects ONLY
rem      populations eight and ten by name, so it NOW COLLECTS AND RUNS all 20 of this ticket's own
rem      new cells alongside the pre-existing 39 -- 59 executed, EXPECTED 7 failed / 52 passed, which
rem      WILL set t2326_scan_ec=1 and fail this build, deliberately, exactly as T-2333's and T-2342's
rem      own new red cells each did in their own rounds until the following build round closed them.
rem      This is this ticket's own intended EXIT state (red for the newly specified behaviour and
rem      nothing else), not a regression to chase -- the next build round closes it by building the
rem      carve-out (D-SLM4886), the two refusal contracts (D-SLM4887/D-SLM4888), and the empty-extent
rem      once-per-section charge, at which point all 7 should flip to passing with no cell edited to
rem      reach that state. A nonzero exit here should be read against this comment's own 7-failure
rem      list, above, before being treated as an unrelated new regression.
rem      T-2348 (Brunel): the carve-out (D-SLM4886), both refusal contracts (D-SLM4887/D-SLM4888),
rem      and the empty-extent once-per-section charge (Poirot's M2, 8a28460-t2344 confirmation) are
rem      ALL BUILT this round, plus 8a28460-t2344's own S1 (build.bat's own %errorlevel% parse-time-
rem      expansion bug, above), S3 (the runner's main() now returns 0 if ci_gate_corpus else 1), M3
rem      (the runner's own /std:c++<N> now derived, CMake's own platform defines added), M5
rem      (scan_object no longer reports object_format="unknown" when only the ISA lacked a decoder),
rem      O2 (a documentation-accuracy correction), and O3 (enumerate_scan_targets's own object
rem      extension is now a parameter, `.obj` by default). The gating pytest invocation above no
rem      longer deselects anything: O4's own build-round half (the now-redundant --deselect flags
rem      and the --collect-only count guard, both routed back by fold round 35's own delta manifest)
rem      is REMOVED -- xfail(strict=True) inside the test file itself (T-2347) now provides the
rem      identical drift protection structurally. Re-executed this session: 61 collected, 59 passed,
rem      2 xfailed, 0 failed -- every one of fold round 35's own 7 newly-red cells now passes for a
rem      genuine grading reason, no cell was edited to reach that state, and the 52 previously-green
rem      cells (T-2326 through T-2343) are unregressed. ANY nonzero exit from this pytest invocation
rem      is now a real regression against this 61-collected/59-passed/2-xfailed/0-failed baseline.
rem      T-2366 (Curie), D-SLM5001: the invocation above now runs the whole suite directory
rem      (test_check_fp_free_scan.py plus a new sibling file, test_ci_gate_wiring.py -- D-SLM5001
rem      item 6, CI reachability). This ticket authors tests only, builds nothing, and adds 12 new
rem      cells realizing the six gaps T-2364's strike and T-2365's coverage audit both found in fold
rem      round 39's own text: (1) the gate's verdict must not read check (C) -- one RED cell
rem      (test_gate_must_not_fail_on_a_check_c_only_reject) plus its own GREEN must-reject control;
rem      (2) check (A)'s eight-mnemonic bitwise-family widening, D-SLM4987 -- two RED cells (the
rem      sixteen-mnemonic sweep and its real-corpus leg), plus population sixteen's own committed
rem      cell reconciled in place rather than left asserting the reversed verdict (now RED -- it was
rem      GREEN before this ticket, since D-SLM4987 requires the opposite verdict from what that cell
rem      asserted pre-fold-39); (3) the p/vp-prefix rule's vitality pin, D-SLM4999 -- two RED cells
rem      (the census-and-violation cell, the future-mnemonic mutation proof), plus one GREEN
rem      mutation proof that the deny list is load-bearing for KNOWN escapes; (4) D-SLM4359's seven
rem      switch-jump-table symbols -- one RED real-corpus sweep (7 of 7 still block the gate); (5)
rem      scan_build_output.py's fail-closed membership discipline, design Sec7 dim 11's thirty-sixth
rem      population -- two GREEN cells, already correct; (6) CI reachability -- two RED cells in the
rem      new sibling file (scan_build_output.py is referenced nowhere under .github/workflows/). Net
rem      this round: 73 collected (61 prior + 12 new), 62 passed, 2 xfailed, 9 failed. THIS CHANGES
rem      WHAT "a real regression" MEANS UNTIL THE NEXT BUILD ROUND, exactly as fold round 35's own
rem      red cells did above: the gating pytest invocation now executes 9 cells expected to fail,
rem      deliberately, until the next build round builds the gate's own check-(C) exclusion, check
rem      (A)'s widening, D-SLM4359's full seven-symbol restructure, and the CI rewiring -- at which
rem      point all 9 should flip to passing with no cell edited to reach that state (per this
rem      ticket's own casebook, Claude/Curie/t2366-fp-gate-red-suite-2026-08-28.md, for the full
rem      per-cell accounting). A nonzero exit here should be read against this comment's own
rem      9-failure list before being treated as an unrelated new regression.
rem      T-2367 (Brunel): the gate's own check-(C) exclusion (a new `ScanResult.ab_verdicts` field,
rem      read by scan_build_output.py's own pass/fail decision instead of the combined `verdicts`
rem      field; check (C) keeps running and reporting as a non-gating diagnostic), check (A)'s
rem      eight-mnemonic bitwise-family widening (D-SLM4987), D-SLM4359's seven-symbol switch-to-
rem      if-chain restructure (src/artifact.cpp, src/model.cpp, src/proof_manifest.cpp,
rem      src/forward/checked_chain_funnel.cpp), and the CI rewiring (.github/workflows/tests.yml's
rem      fp-free-scan-report renamed fp-free-scan-gate, now a real CMake build + a gating
rem      scan_build_output.py invocation, no continue-on-error, no trailing exit 0) are ALL BUILT
rem      this round, plus one new pin file (test_dslm4359_switch_restructure_pin.py, three cells,
rem      a source-level regression guard on the seven-symbol restructure that does not depend on
rem      the read-only D:/SuperSLM/.worktrees/optb-build reference corpus staying current). Item
rem      (4) of D-SLM5001 (populations 17/26's demotion notice) is a Coverage Model edit to the
rem      design document, outside this build round's own writable scope (Claude/Vitruvius/... is
rem      read-only here) -- not touched, routed back to the planner, exactly as T-2365's own
rem      coverage audit already routed it. Re-executed this session: 76 collected (73 prior + 3
rem      new), 4 failed, 70 passed, 2 xfailed -- five of the nine prior failures now pass for a
rem      genuine grading reason (population sixteen's reconciliation, the check-(C)-exclusion pair,
rem      the bitwise-family synthetic sweep, both CI-wiring cells); the new pin's own three cells
rem      pass. FOUR CELLS REMAIN RED, and per this file's own standing law they are NOT edited to
rem      reach green, because each is a defect in the red suite itself rather than in this round's
rem      production code (Claude/Brunel/t2367-fp-gate-build-round-2026-08-28.md carries the full
rem      per-cell reasoning): `test_check_a_p_vp_structural_accept_census_and_violation` pins three
rem      literal counts (539/101/[]) that are mutually exclusive by the census's own partition
rem      arithmetic (539 != 101+0) and can never all hold, independent of any production change;
rem      `test_check_a_p_vp_rule_fails_open_on_a_future_fp_mnemonic` requires check (A)'s p/vp
rem      branch to become a closed allow-list, which is achievable in isolation but contradicts the
rem      currently-GREEN `test_check_a_p_prefix_exclude_list_is_load_bearing`'s own premise (that
rem      removing a name from the deny list flips an unnamed mnemonic to ACCEPT, true only under a
rem      fail-open fallback) -- the two assert incompatible architectures for the same function and
rem      need a planner ruling, not a builder's unilateral pick; `test_check_a_bitwise_family_real_
rem      corpus_leg` calls scan_object without the corpus_symbols the production driver always
rem      supplies, so its own read of the COMBINED verdict (not ab_verdicts) still shows REJECT from
rem      check (C)'s unrelated cross-TU-call vetting, confirmed by direct execution with
rem      corpus_symbols supplied (ACCEPT under both ab_verdicts and verdicts); and `test_dslm4359_
rem      seven_switch_jump_table_symbols_must_not_block_gate` reads the read-only D:/SuperSLM/
rem      .worktrees/optb-build corpus, compiled from engine a1df129 before this round's own source
rem      changes, so it cannot observe this remedy until that shared directory is rebuilt from a
rem      commit including this round's work -- verified instead against a freshly configured build
rem      (D:/SuperSLM/.worktrees/t2367-bld): 0 REJECT, 0 REFUSE under checks (A)/(B) across all 17
rem      objects, down from 264 REJECT / 1 REFUSE pre-remedy. A nonzero exit here should be read
rem      against this comment's own 4-failure list before being treated as an unrelated new
rem      regression.
rem      T-2368 (Curie), D-SLM5008/D-SLM5009: this round's own four red cells are CLOSED, none by
rem      editing production code. D-SLM5008 (corpus-path class): `test_check_fp_free_scan.py` no
rem      longer binds `_REAL_BUILD_ROOT`/`_REAL_BUILD_OBJ_DIR` to the hand-configured
rem      D:/SuperSLM/.worktrees/optb-build path at all -- every corpus-dependent cell now takes a
rem      new `real_build_dir` session fixture (test_t2296-fp-free-open-red-suite/conftest.py) that
rem      configures and builds `superslm` fresh, once per pytest session, into
rem      out/t2368_fp_scan_corpus_build, and SKIPS (never fails) every dependent cell with a stated
rem      reason when no usable corpus can be produced (toolchain absent, or the configure/build
rem      itself fails) -- proven both ways this session (corpus built fresh: 0 skipped; corpus
rem      forced unavailable via the fixture's own SUPERSLM_FP_SCAN_BUILD_DIR override: the 3
rem      corpus-dependent cells skip, the other 74 collected cells are unaffected). Fixing this also
rem      closed `test_check_a_bitwise_family_real_corpus_leg` and
rem      `test_dslm4359_seven_switch_jump_table_symbols_must_not_block_gate`, which now read
rem      `ab_verdicts` (checks (A)/(B) alone, matching `scan_build_output.py`'s own gating decision,
rem      D-SLM5004) instead of the combined `verdicts` field check (C) still populates as a
rem      non-gating diagnostic. D-SLM5009a (stale pin): `test_check_a_p_vp_structural_accept_
rem      census_and_violation` is rebaselined to the reproduced 1523/555/101/454 (539+16/438+16,
rem      D-SLM4987's sixteen-mnemonic widening landing entirely in the structural-only population);
rem      its own former fourth assertion (`structural_only == []`, permanently unsatisfiable by the
rem      partition's own arithmetic once accept_a/named_accept are pinned separately) is split into
rem      its own `xfail(strict=True)` cell,
rem      `test_check_a_p_vp_structural_only_nonempty_violates_fail_closed_claim`, reason D-SLM5009b.
rem      D-SLM5009b (unruled requirement): `test_check_a_p_vp_rule_fails_open_on_a_future_fp_mnemonic`
rem      (asserted a FABRICATED mnemonic must REJECT, demanding a production change no decision
rem      authorizes) is RETIRED and replaced by the vitality cell D-SLM5001 item (5) actually
rem      specified,
rem      `test_check_a_p_vp_structural_only_set_is_pinned_against_vocabulary_growth`: a membership
rem      pin (fp_scan_fixtures/p_vp_structural_only_pinned.txt, 454 mnemonics) against the REAL
rem      decoder's own structural-only accept set, deliberately not attempting an FP-vs-packed-
rem      integer classifier (this suite's own `_is_x86_fp_arith` was confirmed this session to
rem      misclassify real packed-integer mnemonics `pmaxsd`/`pminsd`/`vpcmpd`/`vpmaxsd`/`vpminsd` as
rem      floating-point-shaped) -- whether check (A) should be made fail-closed over the whole p/vp
rem      class stays OPEN, waiting on Dan (D-SLM5009), and is not this ticket's to answer or build.
rem      CLOSED 2026-08-29 (D-SLM5155/D-SLM5156, T-2404 R3): the p/vp class's fail-closed question
rem      the paragraph above leaves OPEN is resolved. check_fp_free_scan.py's own
rem      _X86_P_VP_STRUCTURAL_ALLOW replaces the deny-list-guarded structural rule with a frozen
rem      allow-list; check (A) REJECTs an unrecognized p/vp mnemonic instead of accepting it by the
rem      old rule's own silence. The paragraph above is left standing as a record of the round it
rem      describes, not as a claim about the class's current state.
rem      Re-executed this session: 77 collected, 74 passed, 3 xfailed, 0 failed (corpus available);
rem      77 collected, 71 passed, 3 skipped, 3 xfailed, 0 failed (corpus forced unavailable). ANY
rem      nonzero exit from this pytest invocation is now a real regression against this baseline.
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
	rem T-2441 (Poirot 327ee29-t2438-ask5-tracka-review.md, Significant 3, D-SLM5437): the
	rem geometry site census (design Claude/Vitruvius/t2408-superslm-ask5-qwen3-arch-design-
	rem 2026-08-29.md Sec2.5) is specified as "run as a test-suite cell, not a human step" --
	rem T-2432's own build committed the registry and the script but wired them into nothing
	rem local: not this file, not CMakeLists.txt, not .github/workflows/tests.yml (also wired,
	rem separately, in the same fix round). A tree-wide search for `geometry_site_census`
	rem before this line found only the script's own self-references.
	python tools\geometry_site_census.py
	if errorlevel 1 (
		echo geometry_site_census.py FAILED -- see output above
		set ec=1
	)
	rem T-2139 (Finding 3 class-closer, Claude/Poirot/3bcbe43-t2139-fourth-confirmation-review.md
	rem S3): every tools/*.cpp must have a build recipe SOMEWHERE in HEAD (build.bat,
	rem CMakeLists.txt, tests/*/build_link_red.bat, or tools/build_*.bat) or be an explicitly
	rem justified allowlist entry -- the T-2045/S5 scar this file's own C5 comment already names
	rem once, closed as a class here rather than fixed instance-by-instance again.
	python tools\ci\check_tools_have_build_recipe.py
	if errorlevel 1 (
		echo check_tools_have_build_recipe.py FAILED -- see output above
		set ec=1
	)
	rem T-2314 (gate-reachability sweep, Claude/Brunel/t2314-gate-reachability-2026-08-27.md): the
	rem symmetric class-closer to check_tools_have_build_recipe.py immediately above, for test
	rem SUITE DIRECTORIES rather than tools/*.cpp files -- tests/t2296-fp-free-open-red-suite/, this
	rem arc's entire pin, had zero references anywhere in this file, CMakeLists.txt, or
	rem .github/workflows/tests.yml until the T-2314 wiring above; two siblings
	rem (tests/t2018-slora-serial/, tests/t2178-gpu-batched-prefill-red-suite/) were found orphaned
	rem the same way by the same hand sweep. Validated pre-wiring (Claude/Brunel/t2314-gate-
	rem reachability-raw/check_pre_wiring.txt) to reproduce exactly that three-suite population
	rem before any of the three were wired above.
	python tools\ci\check_tests_have_build_recipe.py
	if errorlevel 1 (
		echo check_tests_have_build_recipe.py FAILED -- see output above
		set ec=1
	)
	rem T-2139 sixth confirmation review item 5 (Claude/Poirot/5fbd04d-t2139-sixth-confirmation-
	rem review.md, ruled in Claude/Vitruvius/t2133-layer1-c-abi-design-2026-08-16.md Sec8's
	rem "structural half" paragraph): M3's class-closer -- extracts every #define/#ifndef/
	rem #if !defined identifier from include/superslm/sslm_abi.h AND every first-party .inc it
	rem transitively includes (M2), and fails when one has no Sec8 inventory line as a WHOLE,
	rem WORD-BOUNDARY-MATCHED token (S2 -- a bare substring test reported false accounted-for on
	rem any new identifier that is a PREFIX of an already-narrated one) outside a code block, same
	rem allowlist-escape-hatch shape as check_tools_have_build_recipe.py above. Population source
	rem is resolved from a COMMITTED GIT REF of the Wizard records repo -- SSLM_ABI_DESIGN_DOC_REPO
	rem (repo path, default: sibling-checkout search) and SSLM_ABI_DESIGN_DOC_REF (ref, default
	rem "develop") -- never filesystem mtime (S1: mtime tracks when a worktree was last WRITTEN
	rem to, never whether its content was ratified). SKIPS gracefully, loudly -- exit 0, never a
	rem build failure -- both when the records repo cannot be found at all, and when the document
	rem resolves but carries no '## 8.' section at the ref read (S1: that state is the same as no
	rem document, so landing Sec8 for the first time is never blocked by a check that depends on
	rem it existing).
	python tools\ci\check_abi_header_inventory.py
	if errorlevel 1 (
		echo check_abi_header_inventory.py FAILED -- see output above
		set ec=1
	)
) else (
	echo python not found on PATH -- skipping tests\ci\check_gpu_guard_status_parity.py, check_gemm_site_thread_width_parity.py, tools\ci\check_tools_have_build_recipe.py, tools\ci\check_tests_have_build_recipe.py, and tools\ci\check_abi_header_inventory.py ^(non-fatal^)
)

popd
exit /b %ec%
