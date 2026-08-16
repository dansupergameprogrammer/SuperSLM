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
echo Gate A must-reject construction correctly failed to compile ^(see out\t2139\gate_a_negative.log^)

rem Gate C must-accept: compiles, links, runs to exit 0.
cl /nologo /std:c++20 /O2 /W4 /EHsc /Iinclude tools\t2139_gate_c_type_identity_check.cpp /Fo:out\t2139\ /Fe:out\t2139_gate_c_type_identity_check.exe
if errorlevel 1 (
	echo Gate C must-accept construction FAILED TO COMPILE -- this is a real regression, not expected
	popd & exit /b 1
)
out\t2139_gate_c_type_identity_check.exe
if errorlevel 1 (
	popd & exit /b 1
)

rem Gate C must-reject: MUST fail to compile. errorlevel 0 here is the regression.
cl /nologo /std:c++20 /O2 /W4 /EHsc /Iinclude tools\t2139_gate_c_type_identity_check_negative.cpp /Fo:out\t2139\ /Fe:out\t2139_gate_c_type_identity_check_negative.exe >out\t2139\gate_c_negative.log 2>&1
if not errorlevel 1 (
	echo Gate C must-reject construction COMPILED CLEAN -- Gate C has regressed, see out\t2139\gate_c_negative.log
	popd & exit /b 1
)
echo Gate C must-reject construction correctly failed to compile ^(see out\t2139\gate_c_negative.log^)

rem count_abi_verbs.sh's own cited figure (design Sec4): 29. Checked only when bash is on PATH
rem (git-bash on a typical Windows dev box) -- non-fatal if absent, matching this script's own
rem python-checker precedent below.
rem T2139_VERB_COUNT is read and compared OUTSIDE any parenthesized if-block on purpose: %VAR%
rem inside a `( ... )` block expands at PARSE time (before the block's own `set /p` line has
rem run) without `setlocal enabledelayedexpansion`, which this script does not otherwise need
rem and this step does not want to turn on globally -- goto/labels sidestep it instead.
where bash >nul 2>nul
if errorlevel 1 goto :t2139_verb_count_skip
bash -c "cat include/superslm/sslm_abi_functions.inc include/superslm/sslm_abi_functions_g5_comparable.inc | grep -oE '\bsslm_[a-z0-9_]+\s*\(' | sed -E 's/\s*\($//' | sort -u | wc -l" > out\t2139\verb_count.txt
set /p T2139_VERB_COUNT=<out\t2139\verb_count.txt
if "%T2139_VERB_COUNT%"=="29" goto :t2139_verb_count_ok
echo count_abi_verbs.sh reports %T2139_VERB_COUNT%, expected 29 -- verb count drifted, see design Sec4
popd & exit /b 1
:t2139_verb_count_ok
echo count_abi_verbs.sh: 29 verbs, matches design Sec4's own cited figure
goto :t2139_verb_count_done
:t2139_verb_count_skip
echo bash not found on PATH -- skipping count_abi_verbs.sh ^(non-fatal^)
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
