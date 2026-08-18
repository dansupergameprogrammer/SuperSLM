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
rem against, not itself an instance of that drift. Checked only when bash is on PATH (git-bash
rem on a typical Windows dev box) -- non-fatal if absent, matching this script's own
rem python-checker precedent below.
rem T2139_VERB_COUNT is read and compared OUTSIDE any parenthesized if-block on purpose: %VAR%
rem inside a `( ... )` block expands at PARSE time (before the block's own `set /p` line has
rem run) without `setlocal enabledelayedexpansion`, which this script does not otherwise need
rem and this step does not want to turn on globally -- goto/labels sidestep it instead.
where bash >nul 2>nul
if errorlevel 1 goto :t2139_verb_count_skip
bash -c "cat include/superslm/sslm_abi_functions.inc include/superslm/sslm_abi_functions_g5_comparable.inc | grep -oE '\bsslm_[a-z0-9_]+\s*\(' | sed -E 's/\s*\($//' | sort -u | wc -l" > out\t2139\verb_count.txt
rem T-2152 (outside strike item 7): `where bash` finding bash on PATH is not proof the
rem command above RAN successfully -- a missing cat/grep/sed/sort/wc inside a minimal
rem git-bash install, a bad working directory, or any other bash-side failure previously
rem fell straight through to the count comparison below, which reports whatever landed in
rem verb_count.txt (often empty, or "0" from a pipe with no input) as if it were a real,
rem measured verb count that happens to differ from 34 -- "verb count drifted" is the WRONG
rem diagnosis for "bash itself failed to run its command," and sends whoever reads it
rem hunting through include/superslm/*.inc for an ABI change that never happened. bash's own
rem exit status is checked FIRST, immediately after the command that produced it (same
rem reason the count read below stays outside a parenthesized block -- %ERRORLEVEL%/errorlevel
rem here reflects the bash invocation, not a later command).
if errorlevel 1 (
	echo count_abi_verbs.sh: bash was found on PATH but its own command failed ^(exit %errorlevel%^) -- this is a broken git-bash/coreutils install, not a verb-count drift; see out\t2139\verb_count.txt for whatever partial output it produced
	popd & exit /b 1
)
set /p T2139_VERB_COUNT=<out\t2139\verb_count.txt
if "%T2139_VERB_COUNT%"=="34" goto :t2139_verb_count_ok
echo count_abi_verbs.sh reports %T2139_VERB_COUNT%, expected 34 -- verb count drifted, see design Sec4 / T-2132
popd & exit /b 1
:t2139_verb_count_ok
echo count_abi_verbs.sh: 34 verbs, matches T-2139 Sec4's 29 plus T-2132/G5's five new verbs
goto :t2139_verb_count_done
:t2139_verb_count_skip
echo bash ^(git-bash^) not found on PATH -- skipping count_abi_verbs.sh ^(non-fatal: this build does not require bash. NOTE, read at source rather than assumed -- no other check in this tree re-derives the 34-verb ABI surface count, so skipping here leaves that specific invariant unchecked on this run, same as every other `where ^<tool^>`-guarded optional step in this script^)
:t2139_verb_count_done

rem C2's own Gate B smoke: maps a real artifact, exercises C1's own construction verbs against
rem it, calls sslm_seq_state_size, unmaps. NOT auto-run against a real artifact here (matching
rem B2/B3/.../t2124's own precedent above -- needs a real .sslm this build does not assume
rem exists on every machine); built so it is ready. Usage: out\t2139_c2_smoke.exe ^<model.sslm^>.
cl /nologo /std:c++20 /O2 /W4 /fp:precise /EHsc /Iinclude ^
	src\artifact.cpp src\sha256.cpp src\tokenizer.cpp src\model.cpp src\intmath.cpp src\silu_lut.cpp src\matmul.cpp src\proof_manifest.cpp src\trace_hook.cpp ^
	src\forward\checked_chain_funnel.cpp src\forward\forward_sites.cpp src\decode_digest.cpp ^
	src\sslm_abi.cpp ^
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

rem Design commit 9e2995f4e7's own same-round pin (Sec10 dim 9): a sequence saved resting
rem BETWEEN decode steps, restored, live and restored both driven one further step -- produced
rem tokens must be bit-identical. NOT auto-run here (same precedent as this file's other
rem real-artifact tools). Usage: out\t2139_dim9_current_token_pin.exe ^<model.sslm^>.
cl /nologo /std:c++20 /O2 /W4 /fp:precise /EHsc /Iinclude ^
	src\artifact.cpp src\sha256.cpp src\tokenizer.cpp src\model.cpp src\intmath.cpp src\silu_lut.cpp src\matmul.cpp src\proof_manifest.cpp src\trace_hook.cpp ^
	src\forward\checked_chain_funnel.cpp src\forward\forward_sites.cpp src\decode_digest.cpp ^
	src\sslm_abi.cpp ^
	tools\t2139_dim9_current_token_pin.cpp /Fo:out\t2139\ /Fe:out\t2139_dim9_current_token_pin.exe
if errorlevel 1 (
	popd & exit /b 1
)
if defined T2139_MODEL (
	out\t2139_dim9_current_token_pin.exe %T2139_MODEL%
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
	echo python not found on PATH -- skipping tests\ci\check_gpu_guard_status_parity.py, check_gemm_site_thread_width_parity.py, tools\ci\check_tools_have_build_recipe.py, and tools\ci\check_abi_header_inventory.py ^(non-fatal^)
)

popd
exit /b %ec%
