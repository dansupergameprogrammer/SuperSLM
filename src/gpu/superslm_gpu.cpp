// T-1986 GPU-serial port -- implementation of the `superslm_gpu::*` contract
// surface T-2024's red suite (Claude/Curie/t2019-gpu-serial-red-suite-2026-08-
// 13.md, include/superslm/gpu_port.h) link-gates against. Built stepwise, in
// Sec11's own dependency order (B1 -> B2 -> B3 -> B4 -> B5 -> B6/B7 -> B8 ->
// B11 -> B12); see the build log (Claude/Brunel/t2025-gpu-serial-build-2026-
// 08-13.md) for which steps are real GPU-verified ports and which are named,
// dated stand-ins for a step not yet reached.
//
// B1 (Sec7.1): the four primitive-battery symbols below are real, GPU-
// dispatched, bit-exact ports of the shipped CPU functions of the same name
// (src/intmath.cpp, src/forward/checked_chain_funnel.cpp), executed on real
// D3D12 hardware via src/gpu/d3d12_harness.h and src/gpu/shaders/*.hlsl.
//
// Every symbol below B1 is a STUB pending its own B-step, marked "// STUB" at
// its definition, so this translation unit provides all 32 symbols the red
// suite link-gates against (StandardsDocument.md Sec5.6: a build must not
// silently omit part of what a plan/suite requires) -- allowing the suite to
// LINK and RUN from B1 forward, with the not-yet-built steps' own cells
// failing an assertion (visible, honest, ASSERTION-RED) rather than the whole
// binary failing to link. A stub never crashes the process and never returns
// a value it is not entitled to return without being wrong in an OBSERVABLE
// way -- CHECK_MSG must be able to catch it, not be accidentally satisfied by
// it.
#include "superslm/gpu_port.h"
#include "superslm/silu_lut_canonical.h"  // kSiluLutCanonicalTable (T-2035 mlp_act_site upload)

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "d3d12_harness.h"

// T-2169 (Rung 2b, design Sec5/Sec8/Sec9): the two chunk-dispatch instrumentation counters
// (tests/support/gpu_chunk_dispatch_instrument.h) -- this suite's own gating symbols
// (Claude/Curie/t2178-t2169-gpu-batched-prefill-red-suite-2026-08-18.md Sec2's "RED BY LINK"
// mechanism) and, simultaneously, the Guard-vitality row's own "actual GPU dispatch count
// issued" instrument. Included only under SUPERSLM_ENABLE_GPU_CHUNK_DISPATCH_INSTRUMENT so a
// normal production build (build.bat, no such define) never sees this test-only header at all;
// tests/t2178-gpu-batched-prefill-red-suite/build_red_suite.bat passes the define so its own
// cells' `extern` references resolve.
#if defined(SUPERSLM_ENABLE_GPU_CHUNK_DISPATCH_INSTRUMENT)
#include "support/gpu_chunk_dispatch_instrument.h"
// T-2169 (Rung 2b): the real global symbols the header above declares `extern` -- this
// definition is what resolves every red-suite cell's own LNK2019 the moment this build
// carries SUPERSLM_ENABLE_GPU_CHUNK_DISPATCH_INSTRUMENT (tests/t2178-gpu-batched-prefill-
// red-suite/build_red_suite.bat, updated this rung to pass it). Incremented at the two sites
// SubmitChunkToFullDepthForG5Bridge's own header comment names (below), each guarded by the
// identical macro so a normal production build (build.bat, no such define) carries neither
// the declaration, the definition, nor the increment.
namespace superslm_test {
std::atomic<int64_t> g_gpu_chunk_submit_count_probe{0};
std::atomic<int64_t> g_gpu_chunk_dispatch_count_probe{0};
}  // namespace superslm_test
#endif

namespace superslm_gpu {
namespace harness {

// GetModuleFileNameA-derived directory of the running executable, plus
// "shaders\\<name>.cso". build.bat (Sec5.7) places the compiled shaders
// there alongside its own built test binary (out\superslm_tests.exe).
// CMakeLists.txt (T-2115, D-SLM3432) also compiles these shaders, behind
// SUPERSLM_BUILD_GPU, into a location matching wherever a GPU-linked
// executable's own default output directory would be -- but does not itself
// build any such executable (superslm_tests links only the CPU-only
// superslm_test_injection), so nothing in the CMake tree currently ends up
// beside them the way build.bat's test binary does. An installed
// superslm::superslm_gpu package exposes their location as
// superslm_GPU_SHADER_DIR (cmake/superslmConfig.cmake.in) for a consumer to
// copy next to its own binary.
std::string ShaderPath(const std::string& name) {
	char path[MAX_PATH]{};
	DWORD n = GetModuleFileNameA(nullptr, path, MAX_PATH);
	std::string dir;
	if (n > 0 && n < MAX_PATH) {
		std::string full(path, n);
		size_t slash = full.find_last_of("\\/");
		dir = (slash == std::string::npos) ? "." : full.substr(0, slash);
	} else {
		dir = ".";
	}
	return dir + "\\shaders\\" + name + ".cso";
}

}  // namespace harness
}  // namespace superslm_gpu

namespace superslm_gpu {

using harness::Device;
using harness::GetDevice;
using harness::GetOrBuildPipeline;

namespace {

void PutI64(std::vector<uint8_t>& buf, int64_t v) {
	uint64_t u = static_cast<uint64_t>(v);
	uint8_t b[8];
	std::memcpy(b, &u, 8);
	buf.insert(buf.end(), b, b + 8);
}

int64_t GetI64(const std::vector<uint8_t>& buf, size_t off) {
	uint64_t u = 0;
	std::memcpy(&u, buf.data() + off, 8);
	return static_cast<int64_t>(u);
}

}  // namespace

// ===========================================================================
// B1 (Sec7.1/Sec11 B1): per-primitive battery. Real GPU dispatch, bit-exact
// against the shipped CPU function of the same name.
// ===========================================================================

int64_t DynamicScaleReciprocalGpu(int64_t dn) {
	auto& pipe = GetOrBuildPipeline("dyn_recip");
	std::vector<uint8_t> in;
	PutI64(in, dn);
	auto out = GetDevice().DispatchOne(pipe.root_sig.Get(), pipe.pso.Get(), in, 8);
	return GetI64(out, 0);
}

int64_t RequantTokenCodeWideGpu(int64_t x_i, int64_t r, int s) {
	auto& pipe = GetOrBuildPipeline("requant_wide");
	std::vector<uint8_t> in;
	PutI64(in, x_i);
	PutI64(in, r);
	PutI64(in, static_cast<int64_t>(s));
	auto out = GetDevice().DispatchOne(pipe.root_sig.Get(), pipe.pso.Get(), in, 8);
	return GetI64(out, 0);
}

bool BiasReconcileWideGpu(int64_t b, int64_t q_b, int64_t r_a, int64_t e_a, int64_t* out) {
	auto& pipe = GetOrBuildPipeline("bias_wide");
	std::vector<uint8_t> in;
	PutI64(in, b);
	PutI64(in, q_b);
	PutI64(in, r_a);
	PutI64(in, e_a);
	auto result = GetDevice().DispatchOne(pipe.root_sig.Get(), pipe.pso.Get(), in, 16);
	bool ok = GetI64(result, 0) != 0;
	if (out) *out = GetI64(result, 8);
	return ok;
}

superslm::CarriedScale CombineCarriedScaleGpu(superslm::CarriedScale a, superslm::CarriedScale b) {
	auto& pipe = GetOrBuildPipeline("combine_carried");
	std::vector<uint8_t> in;
	PutI64(in, a.m);
	PutI64(in, a.e);
	PutI64(in, b.m);
	PutI64(in, b.e);
	auto out = GetDevice().DispatchOne(pipe.root_sig.Get(), pipe.pso.Get(), in, 16);
	superslm::CarriedScale r;
	r.m = GetI64(out, 0);
	r.e = GetI64(out, 8);
	return r;
}

// ===========================================================================
// B5 (Sec5.5/Sec11 B5): the two-schedule int64 abs-max reduction. Real GPU
// dispatch, ported from the proven Claude/Loki/t1993-probe/
// reduce_max_abs_i64.hlsl (T-1993's own remedy, D-SLM2924) -- SCHEME 0
// (sequential) and SCHEME 1 (shared-memory tree) only; SCHEME 2/3 are not
// built for this operator/width at this design's pinned compile target
// (Sec5.5's own ruling: no groupshared 64-bit atomic compiles clean at -WX).
// ===========================================================================

namespace {
int64_t MaxAbsReduceWideGpuDispatch(const char* shader_name, const int64_t* x, size_t n) {
	auto& pipe = GetOrBuildPipeline(shader_name);
	std::vector<uint8_t> in;
	PutI64(in, static_cast<int64_t>(n));
	for (size_t i = 0; i < n; ++i) PutI64(in, x[i]);
	auto out = GetDevice().DispatchOne(pipe.root_sig.Get(), pipe.pso.Get(), in, 8);
	return GetI64(out, 0);
}
}  // namespace

int64_t MaxAbsReduceWideGpuScheme0(const int64_t* x, size_t n) {
	return MaxAbsReduceWideGpuDispatch("reduce_max_abs_i64_scheme0", x, n);
}

int64_t MaxAbsReduceWideGpuScheme1(const int64_t* x, size_t n) {
	return MaxAbsReduceWideGpuDispatch("reduce_max_abs_i64_scheme1", x, n);
}

// ===========================================================================
// B2 (Sec6/Sec11 B2): the guard-path port, isolated tier. Real GPU dispatch,
// bit-exact against the shipped CPU predicate of the same name. B2 does not
// depend on B5 (Sec11's own dependency graph: B1 -> B2 -> B4 -> ...; B1 -> B5
// -> B4), so RequantChainCheckedGpu's own reduction is a self-contained,
// single-thread guard-tier port (requant_chain_checked.hlsl's own, not B5's
// two-schedule production reduction).
// ===========================================================================

superslm::SslmForwardStatus CheckRoundingDivideByPotExponentDomainGpu(int64_t q_B, int64_t e_a) {
	auto& pipe = GetOrBuildPipeline("check_rdp_exponent");
	std::vector<uint8_t> in;
	PutI64(in, q_B);
	PutI64(in, e_a);
	auto out = GetDevice().DispatchOne(pipe.root_sig.Get(), pipe.pso.Get(), in, 8);
	const int64_t tag = GetI64(out, 0);
	switch (tag) {
		case 3:
			return superslm::SslmForwardStatus::RoundingDivideByPotExponentOutOfDomain;
		default:
			return superslm::SslmForwardStatus::Ok;
	}
}

superslm::SslmForwardStatus CheckBiasAccumulateMagnitudeDomainGpu(int64_t acc_i, int64_t b,
                                                                    int64_t q_b, int64_t r_a,
                                                                    int64_t e_a) {
	// Not exercised by any T-2019 cell (grep of tests/test_main.cpp finds no
	// caller; the design's own B2(3) construction routes the equivalent
	// coverage through RunLayerLoopGpu instead, Curie casebook Sec6). Built
	// for real anyway, per B2's own declared scope (gpu_port.h) -- reuses the
	// already-proven BiasReconcileWideGpu dispatch (B1) for the wide
	// reconcile term, then reproduces the second-stage overflow test in
	// `CheckBiasAccumulateMagnitudeDomain` (`checked_chain_funnel.cpp`) in the
	// identical unsigned two's-complement form the CPU reference uses (no UB,
	// same technique host-side as device-side would use).
	int64_t term = 0;
	const bool fits = BiasReconcileWideGpu(b, q_b, r_a, e_a, &term);
	if (!fits) return superslm::SslmForwardStatus::BiasReconcileProductOutOfDomain;
	const uint64_t ua = static_cast<uint64_t>(acc_i);
	const uint64_t ub = static_cast<uint64_t>(term);
	const uint64_t sum = ua + ub;
	const bool same_sign_operands = ((ua ^ ub) >> 63) == 0;
	const bool sum_sign_differs = ((ua ^ sum) >> 63) != 0;
	if (same_sign_operands && sum_sign_differs) {
		return superslm::SslmForwardStatus::BiasReconcileProductOutOfDomain;
	}
	return superslm::SslmForwardStatus::Ok;
}

superslm::ChainResult RequantChainCheckedGpu(const int64_t* wide_row, size_t n,
                                              const superslm::CarriedScale* incoming,
                                              size_t n_incoming,
                                              superslm::CarriedScale site_constant) {
	static constexpr size_t kMaxRow = 32;
	static constexpr size_t kMaxIncoming = 32;
	if (n > kMaxRow || n_incoming > kMaxIncoming) {
		throw std::runtime_error(
		    "RequantChainCheckedGpu: fixture exceeds this guard-tier kernel's fixed capacity "
		    "(32 row elements / 32 incoming factors) -- B4's production shader (not this B2 "
		    "guard-only tier) is where the real per-site row widths are handled");
	}
	std::vector<uint8_t> in(16 + kMaxRow * 8 + kMaxIncoming * 16 + 16, 0);
	const int64_t n_val = static_cast<int64_t>(n);            // n <= kMaxRow, always representable
	const int64_t n_inc_val = static_cast<int64_t>(n_incoming);
	std::memcpy(in.data() + 0, &n_val, 8);
	std::memcpy(in.data() + 8, &n_inc_val, 8);
	for (size_t i = 0; i < n; ++i) {
		std::memcpy(in.data() + 16 + i * 8, &wide_row[i], 8);
	}
	const size_t incoming_off = 16 + kMaxRow * 8;
	for (size_t i = 0; i < n_incoming; ++i) {
		std::memcpy(in.data() + incoming_off + i * 16 + 0, &incoming[i].m, 8);
		std::memcpy(in.data() + incoming_off + i * 16 + 8, &incoming[i].e, 8);
	}
	const size_t site_off = incoming_off + kMaxIncoming * 16;
	std::memcpy(in.data() + site_off + 0, &site_constant.m, 8);
	std::memcpy(in.data() + site_off + 8, &site_constant.e, 8);

	auto& pipe = GetOrBuildPipeline("requant_chain_checked");
	auto out = GetDevice().DispatchOne(pipe.root_sig.Get(), pipe.pso.Get(), in, 8);
	const int64_t tag = GetI64(out, 0);
	switch (tag) {
		case 1:
			return superslm::ChainResult{superslm::SslmForwardStatus::CarriedScaleMantissaOutOfDomain};
		case 2:
			return superslm::ChainResult{superslm::SslmForwardStatus::ChainInputOutOfDomain};
		default:
			return superslm::ChainResult{superslm::SslmForwardStatus::Ok};
	}
}

// ===========================================================================
// B4/B7/B11 (Sec5.6/Sec11): the composed, device-resident, multi-layer
// forward, real GPU dispatches throughout. T-2032 built sites 1-4 (attn_norm,
// q_proj, kv_proj fused, RoPE's own guard); T-2035 completes RoPE's own
// rotation and builds sites 5-16 plus the real per-layer commit dispatch --
// the full 14-dispatch-per-layer composition (Claude/Vitruvius/t1986-...-
// 2026-08-13.md Sec4's own site order; the per-layer loop body of
// `RunLayerLoopImpl` (`forward_sites.cpp`)).
// ===========================================================================

namespace {

uint32_t Align8U32(uint32_t x) { return (x + 7u) & ~7u; }

// T-2045 (S3, Claude/Poirot/82cfca7-gpu-serial-port-build-review.md): a
// single-slot weight-residency cache -- see RunLayerLoopGpu's own header
// comment at its use site for the full rationale. Process-lifetime storage,
// matching this design's own single-execution, no-concurrent-model-handle
// build target (T-811); a real multi-model-handle ABI surface would key this
// per handle rather than globally, which is a residual for whichever
// dispatch first builds that surface.
//
// Keyed on CONTENT (a byte-for-byte comparison of the packed row, `bytes`
// below), never on the `layers` pointer alone. A pointer-only key is unsound:
// this project's own test harness constructs a fresh `NLayerFixture` on each
// loop iteration of a `for (uint32_t k : positions) { NLayerFixture<8>
// fixture; ... }`-shaped test (every T-2019 B11 test in this suite), and a
// short-lived stack object's own address is routinely reused across loop
// iterations by the compiler -- confirmed by execution this ticket's own
// build: a pointer-keyed first cut of this cache served k=0's stale, already-
// uploaded weights back on k=3's call (an entirely different fixture
// mutation, same `fixture.layers` address), regressing 17 previously-green
// cells. Reusing STALE weights across genuinely different models is a
// correctness defect, strictly worse than the re-upload cost S3 exists to
// remove -- a content comparison is the only sound invalidation signal a
// same-signature cache (no separate "map model" handle exists in gpu_port.h's
// own contract) can use.
struct ResidentWeights {
	std::vector<uint8_t> bytes;  // the last-uploaded packed row, kept for comparison
	Microsoft::WRL::ComPtr<ID3D12Resource> lw_buf;
	bool valid = false;
	// T-2100 (throughput): the IDENTITY of the source this packed row was built from. The content
	// comparison above is the only SOUND invalidation signal available without a model handle, and
	// it is also ruinously expensive: it requires packing ~1.31 GiB byte-by-byte and memcmp-ing it
	// on EVERY call, so the cache that exists to skip a 1.31 GiB PCIe transfer spends ~4 GiB of
	// host memory traffic to decide not to do it. Measured: 3.24 s/token, of which the GPU is a
	// small fraction. These three fields let the common case -- the same `layers` array driving
	// every token of one decode session, which is the usage this file's own header comment
	// describes -- skip the pack and the compare entirely. A different pointer, count, or stride
	// falls back to the full content comparison, so the sound signal is retained rather than
	// replaced.
	const void* src_layers = nullptr;  // opaque: identity only, no LayerWeights visibility needed here
	uint32_t src_n = 0;
	uint32_t src_stride = 0;
};
ResidentWeights g_resident_weights;
// T-2052 (item 3, Claude/Curie/t2019-gpu-serial-red-suite-2026-08-13.md
// §13.2): backs the public `LastWeightUploadWasSkipped()` accessor
// (gpu_port.h) -- set every `RunLayerLoopGpu` call, read by the caller after
// it returns.
bool g_last_weight_upload_was_skipped = false;

// T-2101 (dispatch-overhead decomposition): backs the public `LastCallTiming()` accessor
// (`gpu_port.h`) -- see that declaration's own header comment for what each field measures.
GpuCallTiming g_last_call_timing;
// T-2101 (per-site decomposition, follow-up to D-SLM3312): backs the public
// `LastCallPerDispatchTimingsMs()` accessor (`gpu_port.h`) -- one GPU-measured millisecond figure
// per dispatch this call issued, in dispatch order.
std::vector<double> g_last_call_per_dispatch_ms;

// T-2101 (throughput, D-SLM3301/D-SLM3294): the K/V workspace's own device residency, mirroring
// ResidentWeights' shape exactly. Before this: `superslm_gpu.cpp`'s call site copied the WHOLE
// workspace (448 MiB at the 1.5B tier) into a fresh host vector, uploaded it, and read the whole
// thing back, EVERY call -- ~1.8 GiB of host<->device traffic to land one new K/V row per layer.
// Keyed on the `workspace` pointer identity, exactly like ResidentWeights is keyed on `layers`:
// this design's own real call pattern (`tools/sslm_generate.cpp`'s decode loop, this ticket's own
// `tools/t2100_gpu_throughput.cpp`) drives every token of one decode session through the SAME
// caller-owned workspace buffer, so a pointer+size identity is a sound, cheap invalidation signal
// for the common case; a different pointer or size (a different sequence's workspace) falls back
// to a full re-upload, exactly as ResidentWeights falls back to its full content comparison on a
// `layers` mismatch. Process-lifetime storage -- the same single-execution, no-concurrent-model-
// handle residual ResidentWeights' own header comment already names for D-SLM3294's future model-
// owned residency to key per handle instead of globally.
struct ResidentKv {
	Microsoft::WRL::ComPtr<ID3D12Resource> kv_buf;
	bool valid = false;
	const uint8_t* src_workspace = nullptr;  // opaque identity: the caller's own workspace pointer
	size_t src_size = 0;
};
ResidentKv g_resident_kv;

// T-2113 (B4, design Sec3/Sec6.1, re-derived from Claude/Laplace/t2105-gpu-speed-ceiling-
// 2026-08-14.md Sec2 change 1 -- "RoPE cos/sin tables made resident, keyed on source-tensor
// identity"). NOT superseded by B2's own SslmGpuModelHandle::rope_cos_buf/rope_sin_buf
// members in gpu_1p0.cpp: that residency is reachable only through the 1.0 handle API, and this
// function -- `RunLayerLoopGpu`, the pre-1.0 substrate's own shared entry point every caller
// (C5, T-2100's own throughput harness, B1/B3's own bench bridges, and until B5/B6 route
// production calls through handles, everything else) still goes through -- had no cache for
// these two tables at all: the shipped code repacked (host memcpy) and re-uploaded
// (UPLOAD-heap `dev.Upload`) both tables, in full, on EVERY call. At the real 1.5B tier
// (context_cap=32768, head_dim=128 -> pairs=64) that is 2,097,152 elements * 8 bytes = 16 MiB
// EACH for cos and sin, 32 MiB of host memcpy plus a fresh UPLOAD-heap resource creation and
// CPU-side copy, per call -- T-2105's own comment on this exact construction measured it at
// ~11.04 ms/token of host memcpy alone (their own 32-step baseline). This mirrors
// `ResidentWeights`/`ResidentKv`'s own shape exactly: process-lifetime storage, keyed on the
// source tensor's own address and byte count (a sound, cheap invalidation signal for the
// common case -- every token of one decode session reads the SAME model-wide-constant
// `SslmTensorView`), never written by any dispatch (read-only SRV contents), so a hit needs
// no state transition and no invalidation signal beyond the identity check. This is a
// pre-1.0-substrate-scoped process-global, the same footing as `g_resident_weights`/
// `g_resident_kv` above -- D-SLM3294's "no process-global device state in the 1.0 backend"
// requirement (design Sec1) targets the 1.0 API's own backend objects (`SslmGpuContext` and
// what it owns), not this pre-1.0 function, which already carries two such caches.
struct ResidentRopeTables {
	const void* cos_src = nullptr;
	const void* sin_src = nullptr;
	uint64_t cos_bytes = 0;
	uint64_t sin_bytes = 0;
	Microsoft::WRL::ComPtr<ID3D12Resource> cos_buf;
	Microsoft::WRL::ComPtr<ID3D12Resource> sin_buf;
	bool valid = false;
};
ResidentRopeTables g_resident_rope;

// T-2071 (O11's own instrument, retiring the long-named gap: "no way to
// force `CreateCommittedResource` to fail from the suite" -- Claude/Poirot/
// db73b22-.../a3d44e7-.../b543abe-gpu-serial-port-ship-*-review.md, every
// round since T-2055's own P3): deterministic failure injection.
//
// CORRECTED 2026-08-14 (T-2080, Claude/Poirot/
// 94ebee3-gpu-serial-port-closing-review.md, S1/S2; D-SLM3241, superseding
// this paragraph's own original text -- left standing below, not rewritten,
// per this tree's own append-only discipline): the paragraph originally
// said this instrument was scoped to the weight DEFAULT-heap allocation
// "specifically -- the single site S1 (T-2062) fixed," and declined B12's
// own index-parameterized shape on the grounds that it "targets ONE named
// site, not an enumerable sequence." T-2075's own S1 fix then MOVED the arm
// site to `work_scratch_uav` (making the M-b pin live), which this review
// measured as a SWAP, not a widening: with one arm point, the weight
// DEFAULT-heap allocation -- T-2062's own S1 remedy's own site -- became
// permanently unreachable by any test. Fixed here by taking the
// index-parameterized shape after all: `ArmO11AllocationFailureInjection`
// now takes a `site` selector
// (`kO11AllocInjectionSiteWeightDefaultHeap`/`kO11AllocInjectionSite
// WorkScratchUav`, `gpu_port.h`), and BOTH call sites below carry the check
// -- either remedy can be pinned, independently, by arming the site it
// lives at. The original paragraph's own "targets ONE named site... takes
// no index" reasoning is exactly what produced the swap this correction
// closes.
//
// CORRECTED 2026-08-14 (T-2084, Claude/Poirot/
// 42ecf79-gpu-serial-port-round9-review.md, S2; D-SLM3247): the correction
// above claimed the original text was "left standing below, not rewritten"
// -- it was DELETED, not left standing, taking with it the tree's only
// explanation of why the call site inside `RunLayerLoopGpu` carries no
// `#ifdef` at all. Restored below, verbatim, exactly as `gpu_port.h`'s own
// sibling correction (the "T-2076 note" paragraph, beside its own T-2091
// CORRECTED block) already does this correctly on the same day: original
// text standing, correction beside it, not one
// replacing the other.
//
// T-2071 note (original text, superseded above by T-2080's own S1/S2, kept
// standing for the design rationale it carries -- not a claim about the
// CURRENT arm scope, which is now two sites, not one; see the correction
// above for that). Quoted verbatim, so it keeps the T-2071-era symbol names
// (`MaybeThrowInjectedWeightAllocFault`, `SUPERSLM_O11_WEIGHT_ALLOC_
// INJECTION`) that T-2084's own M2 rename swept everywhere else in this
// file -- a quote of history is accurate only if it still says what history
// actually said; see the M2 entry in the T-2084 build-log section for the
// live-code names these map to today:
//
// T-2071 (O11's own instrument, retiring the long-named gap: "no way to
// force `CreateCommittedResource` to fail from the suite" -- Claude/Poirot/
// db73b22-.../a3d44e7-.../b543abe-gpu-serial-port-ship-*-review.md, every
// round since T-2055's own P3): deterministic failure injection scoped to
// the weight DEFAULT-heap allocation specifically -- the single site S1
// (T-2062) fixed and T-2063's own pin cell
// (`TestT2063_S1Mb_WeightAllocationThrow_ReturnsGpuAllocationFailed_
// SkippedFalse`, `tests/test_main.cpp`) exercises. Mirrors
// `src/bad_alloc_wrap.h`'s own established discipline exactly, not B12's
// index-parameterized `ArmAllocationFailureInjection(uint32_t)` (this
// instrument targets ONE named site, not an enumerable sequence, so it
// takes no index): the CALL SITE inside `RunLayerLoopGpu` below is never
// `#ifdef`-guarded -- `MaybeThrowInjectedWeightAllocFault()` is always
// callable and is a no-op that costs nothing once inlined and optimized
// away when `SUPERSLM_O11_WEIGHT_ALLOC_INJECTION` is undefined ("zero
// overhead unarmed"); only the flag, the throw body, and the public
// Arm/Clear functions below are compiled at all when the macro is defined
// (`build.bat`, beside `SUPERSLM_ENABLE_BAD_ALLOC_INJECTION`) -- exactly
// `gpu_port.h`'s own `#ifdef SUPERSLM_O11_WEIGHT_ALLOC_INJECTION` guard
// around these two declarations, which is what makes this definition link
// against them rather than reproducing T-2063's own deliberate LINK-RED
// proof.
namespace {
#ifdef SUPERSLM_O11_ALLOC_INJECTION
bool g_o11_alloc_injection_armed = false;
uint32_t g_o11_alloc_injection_site = 0;
#endif  // SUPERSLM_O11_ALLOC_INJECTION

// Always defined, always callable -- see the header comment above for why
// the call site never needs its own `#ifdef`. Internal linkage (this
// anonymous namespace): only ever called from within this TU
// (`RunLayerLoopGpu`, below), so it need not -- and, to avoid an unused
// external symbol in the unarmed default build, should not -- be visible
// outside it.
//
// T-2080: takes `this_site` -- one of `gpu_port.h`'s own named site
// constants, passed by each call site naming itself. Fires only when the
// ARMED site matches the site currently executing, so arming one site
// never fires at the other.
inline void MaybeThrowInjectedO11AllocFault(uint32_t this_site) {
	// M3 (T-2084, Claude/Poirot/42ecf79-gpu-serial-port-round9-review.md;
	// D-SLM3247): `this_site` is read only inside the macro-gated block below,
	// so every macro-undefined compile -- including the default C5-harness
	// build, which never defines SUPERSLM_O11_ALLOC_INJECTION -- leaves the
	// parameter unreferenced (C4100). Standard idiom: mark it used
	// unconditionally, outside the `#ifdef`, rather than gating the parameter
	// name itself and disturbing the signature between compiles.
	(void)this_site;
#ifdef SUPERSLM_O11_ALLOC_INJECTION
	if (g_o11_alloc_injection_armed && g_o11_alloc_injection_site == this_site) {
		// Single-shot: fires exactly once per `Arm` call, matching the pin
		// cell's own "call 4 (no injection) recovers cleanly" expectation --
		// a re-armed-forever flag would make every subsequent call in the
		// same process fail too, which is not what "the next matching
		// allocation" means.
		g_o11_alloc_injection_armed = false;
		throw std::runtime_error("T2080 O11: injected allocation failure at site " +
		                          std::to_string(this_site));
	}
#endif  // SUPERSLM_O11_ALLOC_INJECTION
}
}  // namespace

// The composed pipeline's own per-layer LayerWeights byte layout -- computed
// ONCE per call from (hidden_size, kv_hidden_size, num_kv_heads) and uploaded
// as the "Layout" SRV every real site shader reads offsets from (never
// re-derived shader-side), so the host packer and the shader reader can never
// drift out of sync with each other. Index order matches every *_site.hlsl's
// own `Layout.Load<uint>(N * 4)` calls exactly; see each shader's own header
// comment for which indices it reads.
// T-2035 (final composed checkpoint): extended to carry every LayerWeights
// field sites 5-16 read. Index 0-24 unchanged from T-2032 (attn_norm/q_proj/
// kv_proj); 25-55 new (o_proj, ctx_fold, attn_residual, mlp_norm, gate/up/
// down_proj, mlp_act, mlp_residual, iexp_softmax_khead); 56 is the stride.
// T-2113 (B2): GpuLayerLayout is now declared in include/superslm/gpu_port.h (promoted so
// gpu_1p0.cpp's own model-handle upload path can share it), and ComputeLayerLayout's
// DEFINITION moved with it -- from HERE (inside this anonymous namespace, internal linkage)
// to just after this namespace's own closing brace below (external linkage, matching the
// header's declaration; same reason PackLayerWeightsBytes moved too, immediately below this
// comment's own sibling). Body unchanged; see the relocated definition, right after this
// anonymous namespace closes.

void PutBytesAt(std::vector<uint8_t>& buf, size_t off, const void* data, size_t n) {
	std::memcpy(buf.data() + off, data, n);
}
void PutI32At(std::vector<uint8_t>& buf, size_t off, int32_t v) { PutBytesAt(buf, off, &v, 4); }
void PutI64At(std::vector<uint8_t>& buf, size_t off, int64_t v) { PutBytesAt(buf, off, &v, 8); }

// T-2113 (B2): PackLayerWeightsBytes's DEFINITION also moved -- to just after this
// anonymous namespace's own closing brace below, alongside ComputeLayerLayout (same reason:
// external linkage, matching its declaration in include/superslm/gpu_port.h, so
// gpu_1p0.cpp's own sslm_gpu_model_map can call it too). Body unchanged; it still uses
// PutI32At/PutI64At (immediately above, internal linkage) via ordinary unqualified lookup --
// an anonymous namespace's members stay visible, unqualified, for the rest of this
// translation unit after the namespace itself closes, exactly as this file's own T-2071
// precedent (below) already relies on for a different pair of symbols.

// T-2039 (real-capacity shader geometry): LayerScratch's own per-call, per-
// real-dims byte layout -- computed ONCE per call from (hidden_size,
// intermediate_size), exactly the same "host computes, shader never re-
// derives" discipline GpuLayerLayout already established for LayerWeights,
// so the two can never drift apart. Every offset is a codes block
// (width * 4 bytes, one int32 per code, 8-byte aligned) or a scale block (16
// bytes, CarriedScale.m/e). Superseding T-2035's own fixed 640-byte layout
// (Claude/Brunel/t2025-gpu-serial-build-2026-08-13.md Sec13.5's own named
// blocker): that layout's per-field 32-byte reservations were sized for the
// T-2019 suite's own hidden_size<=8 fixture and overflow at real widths;
// this layout is driven entirely by the real per-call hidden_size/
// intermediate_size root-constant values.
struct GpuScratchLayout {
	uint32_t normed = 0, normed_scale = 0;
	uint32_t q_codes = 0, q_scale = 0;
	uint32_t q_rot = 0;
	uint32_t ctx_codes = 0, ctx_scale = 0;
	uint32_t o_codes = 0, o_scale = 0;
	uint32_t attn_stream = 0, attn_stream_scale = 0;
	uint32_t gate_codes = 0, gate_scale = 0;
	uint32_t up_codes = 0, up_scale = 0;
	uint32_t act_codes = 0, act_scale = 0;
	uint32_t down_codes = 0, down_scale = 0;
	uint32_t stream_next = 0, stream_next_scale = 0;
	// T-2045 (C3, Claude/Poirot/82cfca7-gpu-serial-port-build-review.md): a
	// persistent scores/probs region, `num_attention_heads * context_cap`
	// int64 elements -- the ratified 16-site decomposition splits attention
	// into four separate dispatches (attention-score, softmax, context-
	// accumulate, ctx_fold), so the score row one dispatch computes and the
	// probs row the next reads/overwrites in place must survive ACROSS
	// dispatches, unlike WorkScratch's transient per-dispatch regions.
	uint32_t scores = 0;
	uint32_t total = 0;
};

// T-2039: SeqState's own per-call, per-real-hidden_size byte layout -- the
// IDENTICAL one-line formula site_common.hlsli's SeqScaleOffGpu/.../
// SeqStickyOffGpu family computes shader-side from the same real
// g_hidden_size root constant (a trivial, single-field-order layout, safe to
// compute independently on both sides rather than round-trip through
// another SRV table, unlike LayerWeights'/LayerScratch's own multi-field
// packing). Superseding T-2035's fixed hidden_codes[8]/32-byte assumption.
uint32_t SeqScaleOff(uint32_t hidden_size) { return Align8U32(hidden_size * 4); }
uint32_t SeqLayerIdxOff(uint32_t hidden_size) { return SeqScaleOff(hidden_size) + 16u; }
uint32_t SeqSatLoOff(uint32_t hidden_size) { return SeqLayerIdxOff(hidden_size) + 8u; }
uint32_t SeqSatHiOff(uint32_t hidden_size) { return SeqSatLoOff(hidden_size) + 4u; }
uint32_t SeqCtxLenOff(uint32_t hidden_size) { return SeqSatHiOff(hidden_size) + 4u; }
uint32_t SeqStickyOff(uint32_t hidden_size) { return SeqCtxLenOff(hidden_size) + 8u; }
uint32_t SeqTotalSize(uint32_t hidden_size) { return SeqStickyOff(hidden_size) + 8u; }

GpuScratchLayout ComputeScratchLayout(uint32_t hidden_size, uint32_t intermediate_size,
                                       uint32_t num_attention_heads, uint32_t context_cap) {
	GpuScratchLayout L;
	uint32_t cur = 0;
	auto codes_block = [&](uint32_t width) {
		uint32_t o = cur;
		cur += Align8U32(width * 4);
		return o;
	};
	auto scale_block = [&]() {
		uint32_t o = cur;
		cur += 16;
		return o;
	};
	L.normed = codes_block(hidden_size); L.normed_scale = scale_block();
	L.q_codes = codes_block(hidden_size); L.q_scale = scale_block();
	L.q_rot = codes_block(hidden_size);
	L.ctx_codes = codes_block(hidden_size); L.ctx_scale = scale_block();
	L.o_codes = codes_block(hidden_size); L.o_scale = scale_block();
	L.attn_stream = codes_block(hidden_size); L.attn_stream_scale = scale_block();
	L.gate_codes = codes_block(intermediate_size); L.gate_scale = scale_block();
	L.up_codes = codes_block(intermediate_size); L.up_scale = scale_block();
	L.act_codes = codes_block(intermediate_size); L.act_scale = scale_block();
	L.down_codes = codes_block(hidden_size); L.down_scale = scale_block();
	L.stream_next = codes_block(hidden_size); L.stream_next_scale = scale_block();
	L.scores = cur;
	cur += static_cast<uint32_t>(static_cast<uint64_t>(num_attention_heads) *
	                              static_cast<uint64_t>(context_cap) * 8ull);
	L.total = cur;
	return L;
}

// T-2032's own local status-tag encoding (site_common.hlsli's kTag* family),
// mapped to the real superslm::SslmForwardStatus by name -- the same
// switch-mapped convention check_rdp_exponent.hlsl/requant_chain_checked.hlsl
// already established for B2.
superslm::SslmForwardStatus DecodeStickyTag(int64_t tag) {
	using S = superslm::SslmForwardStatus;
	switch (tag) {
		case 0: return S::Ok;
		case 1: return S::CarriedScaleMantissaOutOfDomain;
		case 2: return S::ChainInputOutOfDomain;
		case 3: return S::RoundingDivideByPotExponentOutOfDomain;
		case 4: return S::BiasReconcileProductOutOfDomain;
		case 5: return S::RopeTableTensorMissing;
		case 6: return S::RopeTableExtentExceeded;
		case 7: return S::PositionOverCap;
		case 9: return S::SoftmaxRowWidthOutOfDomain;
		case 10: return S::IExpScaleDerivationOutOfDomain;
		case 11: return S::SoftmaxKernelRefusedAfterGateAccepted;
		case 12: return S::ResidualReconciliationMagnitudeOutOfDomain;
		case 13: return S::SiluCompositionScaleOutOfDomain;
		default: return S::KvPrecisionUnsupported;  // 8 = NotYetImplemented, and any unmapped tag
	}
}

// `keep_alive` collects the temporary upload-heap resource: the GPU's own
// CopyResource command is only RECORDED here, not executed, and a ComPtr that
// goes out of scope at this function's own return destroys the upload buffer
// before the command list is ever submitted -- the caller must hold it alive
// through ExecuteCommandLists + the fence wait, exactly like lw_buf/layout_buf/
// rope_buf already are by staying in the caller's own local scope.
Microsoft::WRL::ComPtr<ID3D12Resource> MakeInitializedUav(
    harness::Device& dev, const std::vector<uint8_t>& init,
    std::vector<Microsoft::WRL::ComPtr<ID3D12Resource>>& keep_alive) {
	auto buf = dev.MakeBuffer(init.size(), D3D12_HEAP_TYPE_DEFAULT,
	                           D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_DEST);
	auto upload = dev.Upload(init.data(), init.size());
	dev.list->CopyResource(buf.Get(), upload.Get());
	keep_alive.push_back(upload);
	D3D12_RESOURCE_BARRIER b{};
	b.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
	b.Transition.pResource = buf.Get();
	b.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_DEST;
	b.Transition.StateAfter = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
	b.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
	dev.list->ResourceBarrier(1, &b);
	return buf;
}

}  // namespace

// T-2113 (B2): relocated from inside the anonymous namespace above (internal linkage) to
// here (external linkage, `superslm_gpu::ComputeLayerLayout`) -- matching precedent
// (T-2071's own pair, immediately below) and the declaration this now satisfies,
// include/superslm/gpu_port.h. Body byte-for-byte unchanged from its original definition.
GpuLayerLayout ComputeLayerLayout(uint32_t hidden_size, uint32_t kv_hidden_size,
                                   uint32_t num_kv_heads, uint32_t num_attention_heads,
                                   uint32_t intermediate_size) {
	GpuLayerLayout L;
	uint32_t cur = 0;
	L.off[0] = cur; cur += Align8U32(hidden_size * 4);              // attn_norm_gain
	L.off[1] = cur; cur += 16;                                      // attn_norm_site_constant
	L.off[2] = cur; cur += Align8U32(hidden_size * hidden_size);    // q_weight (int8)
	L.off[3] = cur; cur += Align8U32(hidden_size * 4);              // q_fold_identity
	L.off[4] = cur; cur += Align8U32(hidden_size * 4);              // q_fold_mult
	L.off[5] = cur; cur += Align8U32(hidden_size * 4);              // q_fold_shift
	L.off[6] = cur; cur += 16;                                      // q_site_constant
	L.off[7] = cur; cur += 8;                                       // q_bias_present
	L.off[8] = cur; cur += Align8U32(hidden_size * 8);              // q_bias
	L.off[9] = cur; cur += Align8U32(kv_hidden_size * hidden_size); // k_weight (int8)
	L.off[10] = cur; cur += Align8U32(kv_hidden_size * hidden_size);// v_weight (int8)
	L.off[11] = cur; cur += Align8U32(kv_hidden_size * 4);          // k_fold_identity
	L.off[12] = cur; cur += Align8U32(kv_hidden_size * 4);          // k_fold_mult
	L.off[13] = cur; cur += Align8U32(kv_hidden_size * 4);          // k_fold_shift
	L.off[14] = cur; cur += Align8U32(kv_hidden_size * 4);          // v_fold_identity
	L.off[15] = cur; cur += Align8U32(kv_hidden_size * 4);          // v_fold_mult
	L.off[16] = cur; cur += Align8U32(kv_hidden_size * 4);          // v_fold_shift
	L.off[17] = cur; cur += 8;                                      // k_bias_present
	L.off[18] = cur; cur += Align8U32(kv_hidden_size * 8);          // k_bias
	L.off[19] = cur; cur += 8;                                      // v_bias_present
	L.off[20] = cur; cur += Align8U32(kv_hidden_size * 8);          // v_bias
	L.off[21] = cur; cur += Align8U32(num_kv_heads * 8);            // kv_landing_r_t_k
	L.off[22] = cur; cur += Align8U32(num_kv_heads * 8);            // kv_landing_e_t_k
	L.off[23] = cur; cur += Align8U32(num_kv_heads * 8);            // kv_landing_r_t_v
	L.off[24] = cur; cur += Align8U32(num_kv_heads * 8);            // kv_landing_e_t_v
	L.off[25] = cur; cur += Align8U32(hidden_size * hidden_size);   // o_weight (int8)
	L.off[26] = cur; cur += Align8U32(hidden_size * 4);             // o_fold_identity
	L.off[27] = cur; cur += Align8U32(hidden_size * 4);             // o_fold_mult
	L.off[28] = cur; cur += Align8U32(hidden_size * 4);             // o_fold_shift
	L.off[29] = cur; cur += 16;                                     // o_site_constant
	L.off[30] = cur; cur += Align8U32(num_attention_heads * 4);     // ctx_fold_identity
	L.off[31] = cur; cur += Align8U32(num_attention_heads * 4);     // ctx_fold_mult
	L.off[32] = cur; cur += Align8U32(num_attention_heads * 4);     // ctx_fold_shift
	L.off[33] = cur; cur += 16;                                     // ctx_fold_site_constant
	L.off[34] = cur; cur += 16;                                     // attn_residual_site_constant
	L.off[35] = cur; cur += Align8U32(hidden_size * 4);             // mlp_norm_gain
	L.off[36] = cur; cur += 16;                                     // mlp_norm_site_constant
	L.off[37] = cur; cur += Align8U32(intermediate_size * hidden_size);  // gate_weight (int8)
	L.off[38] = cur; cur += Align8U32(intermediate_size * 4);       // gate_fold_identity
	L.off[39] = cur; cur += Align8U32(intermediate_size * 4);       // gate_fold_mult
	L.off[40] = cur; cur += Align8U32(intermediate_size * 4);       // gate_fold_shift
	L.off[41] = cur; cur += 16;                                     // gate_site_constant
	L.off[42] = cur; cur += Align8U32(intermediate_size * hidden_size);  // up_weight (int8)
	L.off[43] = cur; cur += Align8U32(intermediate_size * 4);       // up_fold_identity
	L.off[44] = cur; cur += Align8U32(intermediate_size * 4);       // up_fold_mult
	L.off[45] = cur; cur += Align8U32(intermediate_size * 4);       // up_fold_shift
	L.off[46] = cur; cur += 16;                                     // up_site_constant
	L.off[47] = cur; cur += 16;                                     // mlp_act_site_constant
	L.off[48] = cur; cur += Align8U32(hidden_size * intermediate_size);  // down_weight (int8)
	L.off[49] = cur; cur += Align8U32(hidden_size * 4);             // down_fold_identity
	L.off[50] = cur; cur += Align8U32(hidden_size * 4);             // down_fold_mult
	L.off[51] = cur; cur += Align8U32(hidden_size * 4);             // down_fold_shift
	L.off[52] = cur; cur += 16;                                     // down_site_constant
	L.off[53] = cur; cur += 16;                                     // mlp_residual_site_constant
	L.off[54] = cur; cur += Align8U32(num_kv_heads * 8);            // iexp_softmax_khead_m
	L.off[55] = cur; cur += Align8U32(num_kv_heads * 8);            // iexp_softmax_khead_e
	L.stride = cur;
	return L;
}

// T-2113 (B2): relocated from inside the anonymous namespace above (internal linkage) to
// here (external linkage, `superslm_gpu::PackLayerWeightsBytes`), matching its declaration
// in include/superslm/gpu_port.h. Body byte-for-byte unchanged -- the exact loop
// RunLayerLoopGpu's own weight-residency miss path (`if (!lw_fast_hit) { ... }`, its own
// call site, below) used to contain inline; extracted so it has exactly one implementation,
// callable from both this file's own call site (unchanged behavior) and gpu_1p0.cpp's own
// sslm_gpu_model_map (design Sec10 B2).
std::vector<uint8_t> PackLayerWeightsBytes(const superslm::LayerWeights* layers, uint32_t N,
                                            const GpuLayerLayout& layout, uint32_t H, uint32_t KV,
                                            uint32_t NH, uint32_t NQH, uint32_t I) {
	std::vector<uint8_t> lw_bytes;
	lw_bytes.assign(static_cast<size_t>(layout.stride) * N, 0);
	for (uint32_t l = 0; l < N; ++l) {
		const superslm::LayerWeights& lw = layers[l];
		const size_t base = static_cast<size_t>(l) * layout.stride;
		for (uint32_t i = 0; i < H; ++i) PutI32At(lw_bytes, base + layout.off[0] + i * 4, lw.attn_norm_gain[i]);
		PutI64At(lw_bytes, base + layout.off[1] + 0, lw.attn_norm_site_constant.m);
		PutI64At(lw_bytes, base + layout.off[1] + 8, lw.attn_norm_site_constant.e);
		for (uint32_t i = 0; i < H * H; ++i) lw_bytes[base + layout.off[2] + i] = static_cast<uint8_t>(lw.q_weight[i]);
		for (uint32_t i = 0; i < H; ++i) {
			PutI32At(lw_bytes, base + layout.off[3] + i * 4, lw.q_fold_identity[i]);
			PutI32At(lw_bytes, base + layout.off[4] + i * 4, lw.q_fold_mult[i]);
			PutI32At(lw_bytes, base + layout.off[5] + i * 4, lw.q_fold_shift[i]);
		}
		PutI64At(lw_bytes, base + layout.off[6] + 0, lw.q_site_constant.m);
		PutI64At(lw_bytes, base + layout.off[6] + 8, lw.q_site_constant.e);
		PutI64At(lw_bytes, base + layout.off[7], lw.q_bias != nullptr ? 1 : 0);
		if (lw.q_bias != nullptr) {
			for (uint32_t i = 0; i < H; ++i) PutI64At(lw_bytes, base + layout.off[8] + i * 8, lw.q_bias[i]);
		}
		for (uint32_t i = 0; i < KV * H; ++i) {
			lw_bytes[base + layout.off[9] + i] = static_cast<uint8_t>(lw.k_weight[i]);
			lw_bytes[base + layout.off[10] + i] = static_cast<uint8_t>(lw.v_weight[i]);
		}
		for (uint32_t i = 0; i < KV; ++i) {
			PutI32At(lw_bytes, base + layout.off[11] + i * 4, lw.k_fold_identity[i]);
			PutI32At(lw_bytes, base + layout.off[12] + i * 4, lw.k_fold_mult[i]);
			PutI32At(lw_bytes, base + layout.off[13] + i * 4, lw.k_fold_shift[i]);
			PutI32At(lw_bytes, base + layout.off[14] + i * 4, lw.v_fold_identity[i]);
			PutI32At(lw_bytes, base + layout.off[15] + i * 4, lw.v_fold_mult[i]);
			PutI32At(lw_bytes, base + layout.off[16] + i * 4, lw.v_fold_shift[i]);
		}
		PutI64At(lw_bytes, base + layout.off[17], lw.k_bias != nullptr ? 1 : 0);
		if (lw.k_bias != nullptr) {
			for (uint32_t i = 0; i < KV; ++i) PutI64At(lw_bytes, base + layout.off[18] + i * 8, lw.k_bias[i]);
		}
		PutI64At(lw_bytes, base + layout.off[19], lw.v_bias != nullptr ? 1 : 0);
		if (lw.v_bias != nullptr) {
			for (uint32_t i = 0; i < KV; ++i) PutI64At(lw_bytes, base + layout.off[20] + i * 8, lw.v_bias[i]);
		}
		for (uint32_t i = 0; i < NH; ++i) {
			PutI64At(lw_bytes, base + layout.off[21] + i * 8, lw.kv_landing_r_t_k[i]);
			PutI64At(lw_bytes, base + layout.off[22] + i * 8, lw.kv_landing_e_t_k[i]);
			PutI64At(lw_bytes, base + layout.off[23] + i * 8, lw.kv_landing_r_t_v[i]);
			PutI64At(lw_bytes, base + layout.off[24] + i * 8, lw.kv_landing_e_t_v[i]);
		}
		// T-2035: sites 5-16's own read-resource list.
		for (uint32_t i = 0; i < H * H; ++i) lw_bytes[base + layout.off[25] + i] = static_cast<uint8_t>(lw.o_weight[i]);
		for (uint32_t i = 0; i < H; ++i) {
			PutI32At(lw_bytes, base + layout.off[26] + i * 4, lw.o_fold_identity[i]);
			PutI32At(lw_bytes, base + layout.off[27] + i * 4, lw.o_fold_mult[i]);
			PutI32At(lw_bytes, base + layout.off[28] + i * 4, lw.o_fold_shift[i]);
		}
		PutI64At(lw_bytes, base + layout.off[29] + 0, lw.o_site_constant.m);
		PutI64At(lw_bytes, base + layout.off[29] + 8, lw.o_site_constant.e);
		for (uint32_t i = 0; i < NQH; ++i) {
			PutI32At(lw_bytes, base + layout.off[30] + i * 4, lw.ctx_fold_identity[i]);
			PutI32At(lw_bytes, base + layout.off[31] + i * 4, lw.ctx_fold_mult[i]);
			PutI32At(lw_bytes, base + layout.off[32] + i * 4, lw.ctx_fold_shift[i]);
		}
		PutI64At(lw_bytes, base + layout.off[33] + 0, lw.ctx_fold_site_constant.m);
		PutI64At(lw_bytes, base + layout.off[33] + 8, lw.ctx_fold_site_constant.e);
		PutI64At(lw_bytes, base + layout.off[34] + 0, lw.attn_residual_site_constant.m);
		PutI64At(lw_bytes, base + layout.off[34] + 8, lw.attn_residual_site_constant.e);
		for (uint32_t i = 0; i < H; ++i) PutI32At(lw_bytes, base + layout.off[35] + i * 4, lw.mlp_norm_gain[i]);
		PutI64At(lw_bytes, base + layout.off[36] + 0, lw.mlp_norm_site_constant.m);
		PutI64At(lw_bytes, base + layout.off[36] + 8, lw.mlp_norm_site_constant.e);
		for (uint32_t i = 0; i < I * H; ++i) {
			lw_bytes[base + layout.off[37] + i] = static_cast<uint8_t>(lw.gate_weight[i]);
			lw_bytes[base + layout.off[42] + i] = static_cast<uint8_t>(lw.up_weight[i]);
		}
		for (uint32_t i = 0; i < I; ++i) {
			PutI32At(lw_bytes, base + layout.off[38] + i * 4, lw.gate_fold_identity[i]);
			PutI32At(lw_bytes, base + layout.off[39] + i * 4, lw.gate_fold_mult[i]);
			PutI32At(lw_bytes, base + layout.off[40] + i * 4, lw.gate_fold_shift[i]);
			PutI32At(lw_bytes, base + layout.off[43] + i * 4, lw.up_fold_identity[i]);
			PutI32At(lw_bytes, base + layout.off[44] + i * 4, lw.up_fold_mult[i]);
			PutI32At(lw_bytes, base + layout.off[45] + i * 4, lw.up_fold_shift[i]);
		}
		PutI64At(lw_bytes, base + layout.off[41] + 0, lw.gate_site_constant.m);
		PutI64At(lw_bytes, base + layout.off[41] + 8, lw.gate_site_constant.e);
		PutI64At(lw_bytes, base + layout.off[46] + 0, lw.up_site_constant.m);
		PutI64At(lw_bytes, base + layout.off[46] + 8, lw.up_site_constant.e);
		PutI64At(lw_bytes, base + layout.off[47] + 0, lw.mlp_act_site_constant.m);
		PutI64At(lw_bytes, base + layout.off[47] + 8, lw.mlp_act_site_constant.e);
		for (uint32_t i = 0; i < H * I; ++i) lw_bytes[base + layout.off[48] + i] = static_cast<uint8_t>(lw.down_weight[i]);
		for (uint32_t i = 0; i < H; ++i) {
			PutI32At(lw_bytes, base + layout.off[49] + i * 4, lw.down_fold_identity[i]);
			PutI32At(lw_bytes, base + layout.off[50] + i * 4, lw.down_fold_mult[i]);
			PutI32At(lw_bytes, base + layout.off[51] + i * 4, lw.down_fold_shift[i]);
		}
		PutI64At(lw_bytes, base + layout.off[52] + 0, lw.down_site_constant.m);
		PutI64At(lw_bytes, base + layout.off[52] + 8, lw.down_site_constant.e);
		PutI64At(lw_bytes, base + layout.off[53] + 0, lw.mlp_residual_site_constant.m);
		PutI64At(lw_bytes, base + layout.off[53] + 8, lw.mlp_residual_site_constant.e);
		for (uint32_t i = 0; i < NH; ++i) {
			PutI64At(lw_bytes, base + layout.off[54] + i * 8,
			         lw.iexp_softmax_khead_m != nullptr ? lw.iexp_softmax_khead_m[i] : 0);
			PutI64At(lw_bytes, base + layout.off[55] + i * 8,
			         lw.iexp_softmax_khead_e != nullptr ? lw.iexp_softmax_khead_e[i] : 0);
		}
	}
	return lw_bytes;
}

// T-2071: external linkage, deliberately OUTSIDE the anonymous namespace
// above (unlike `g_o11_alloc_injection_armed`/`MaybeThrowInjectedO11
// AllocFault`, which stay internal-linkage on purpose) -- these two must be
// callable as `superslm_gpu::ArmO11AllocationFailureInjection(site)` from
// `tests/test_main.cpp`, a different translation unit, matching
// `gpu_port.h`'s own gated declarations exactly.
//
// CORRECTED 2026-08-14 (T-2084, Claude/Poirot/
// 42ecf79-gpu-serial-port-round9-review.md, M1; D-SLM3247): the call form
// above was the zero-argument one -- true when this comment was written
// (T-2071), false since T-2080 gave `ArmO11AllocationFailureInjection`
// its own `site` parameter three lines below. Corrected to the real call
// shape rather than left describing a signature that no longer compiles.
#ifdef SUPERSLM_O11_ALLOC_INJECTION
void ArmO11AllocationFailureInjection(uint32_t site) {
	g_o11_alloc_injection_armed = true;
	g_o11_alloc_injection_site = site;
}
void ClearO11AllocationInjection() { g_o11_alloc_injection_armed = false; }
#endif  // SUPERSLM_O11_ALLOC_INJECTION

// T-2101 (S4, code review 6d9e04e-t2101-gpu-throughput-review.md, confirmation pass @ f7026db): a
// distinct exception type for the multi-group GEMM standing guard below -- NOT `std::runtime_error`,
// so it is never accidentally caught by the SAME generic `catch (const std::runtime_error&)` this
// function's own recording window already uses for genuine D3D12 allocation/device failures. That
// generic catch's own cache-invalidation logic (both residency caches, `LastWeightUploadWasSkipped`)
// still applies unconditionally on ANY throw in the window regardless of cause, so this type is
// caught by a SEPARATE, explicit clause inside the SAME try/catch (not hoisted above it, which
// would skip that cleanup) -- the confirmation pass's own first option: "give the check its own
// status... distinct from allocation failure." `std::logic_error`, not `std::runtime_error`: this
// is a permanent coding-arithmetic bug, never a transient or environmental condition, and the two
// exception hierarchies are unrelated by design in the standard library for exactly this
// distinction.
struct GpuGemmGroupArithmeticError : std::logic_error {
	using std::logic_error::logic_error;
};

// T-2180/T-2183 (D-SLM3655/D-SLM3660): the tenth-failure-origin fault-injection seam
// (`gpu_port.h`'s own header comment on `ArmT2169ChunkRecordingFaultInjection` carries the full
// design/casebook account). State declared here, after `GpuGemmGroupArithmeticError` (immediately
// above) rather than beside `g_o11_alloc_injection_armed` (this file, above): the throw body below
// constructs that type, which is not yet declared at the earlier point in this file O11's own
// state block occupies -- construction-order, not a stylistic choice.
namespace {
#ifdef SUPERSLM_T2169_CHUNK_RECORDING_FAULT_INJECTION
bool g_t2169_chunk_recording_fault_armed = false;
uint32_t g_t2169_chunk_recording_fault_after_token_index = 0;
#endif  // SUPERSLM_T2169_CHUNK_RECORDING_FAULT_INJECTION

// Always defined, always callable -- mirrors `MaybeThrowInjectedO11AllocFault`'s own established
// "zero overhead unarmed, call site never `#ifdef`-guarded" convention exactly. Internal linkage:
// only ever called from `SubmitOneSubChunkToFullDepthForG5Bridge`, this same translation unit.
inline void MaybeThrowInjectedT2169ChunkRecordingFault(uint32_t admitted_token_index) {
	(void)admitted_token_index;  // unreferenced when the macro is undefined -- matches
	                              // MaybeThrowInjectedO11AllocFault's own M3 fix (this file, above).
#ifdef SUPERSLM_T2169_CHUNK_RECORDING_FAULT_INJECTION
	if (g_t2169_chunk_recording_fault_armed &&
	    g_t2169_chunk_recording_fault_after_token_index == admitted_token_index) {
		// Single-shot: cleared before throwing, matching O11's own "the next matching call" idiom
		// -- a re-armed-forever flag would fire on every later, unrelated call in the same process.
		g_t2169_chunk_recording_fault_armed = false;
		throw GpuGemmGroupArithmeticError(
		    "T2183 D-SLM3660: injected mid-recording infrastructural fault after admitted token "
		    "index " +
		    std::to_string(admitted_token_index));
	}
#endif  // SUPERSLM_T2169_CHUNK_RECORDING_FAULT_INJECTION
}
}  // namespace

// External linkage, deliberately outside the anonymous namespace immediately above -- same
// reasoning as `ArmO11AllocationFailureInjection` (this file, above): `tests/t2178-gpu-batched-
// prefill-red-suite/cell_exit_census.cpp`, a different translation unit, calls these as
// `superslm_gpu::ArmT2169ChunkRecordingFaultInjection(...)`, matching `gpu_port.h`'s own gated
// declarations exactly.
#ifdef SUPERSLM_T2169_CHUNK_RECORDING_FAULT_INJECTION
void ArmT2169ChunkRecordingFaultInjection(uint32_t after_token_index) {
	g_t2169_chunk_recording_fault_armed = true;
	g_t2169_chunk_recording_fault_after_token_index = after_token_index;
}
void ClearT2169ChunkRecordingFaultInjection() { g_t2169_chunk_recording_fault_armed = false; }
#endif  // SUPERSLM_T2169_CHUNK_RECORDING_FAULT_INJECTION

// T-2113 (B5, design Sec4.2/Sec6.2/Sec10 B5): the full definition of the opaque token
// gpu_port.h forward-declares. Every field is FINISH-phase-only state that only exists
// once Submit has recorded and closed the command list -- see gpu_port.h's own comment
// on `RunLayerLoopGpuSubmit`/`RunLayerLoopGpuFinish` for the ownership contract.
struct GpuLayerLoopInFlight {
	harness::Device* dev = nullptr;
	uint64_t fence_val = 0;
	uint32_t dispatch_count_this_call = 0;
	Microsoft::WRL::ComPtr<ID3D12Resource> seq_readback;
	Microsoft::WRL::ComPtr<ID3D12Resource> kv_readback;
	std::vector<size_t> kv_row_offsets;
	size_t seq_bytes_size = 0;
	uint32_t hidden_size_h = 0;
	uint32_t head_dim_hd = 0;
	std::chrono::steady_clock::time_point t_record_end{};

	// T-2113 (B5): EVERY per-call GPU resource the recorded command list either reads
	// throughout the dispatch chain or copies FROM, that is not independently kept
	// alive by a longer-lived owner (g_resident_weights/g_resident_kv/g_resident_rope's
	// own caches, or a caller-owned external buffer), MUST survive until the fence this
	// token names has actually signaled -- the GPU does not finish reading/copying from
	// these merely because ExecuteCommandLists/Signal returned on the CPU side. In the
	// pre-split synchronous RunLayerLoopGpu, this was true BY CONSTRUCTION: these were
	// ordinary C++ locals of the one function whose stack frame did not unwind until
	// after the fence-wait. Splitting the function moved the fence-wait to a SEPARATE
	// call (RunLayerLoopGpuFinish) -- without these fields, each ComPtr below would be
	// destroyed the instant Submit's own stack frame unwinds, RELEASING (potentially
	// freeing) a D3D12 resource the GPU may still be reading from or copying out of, a
	// real use-after-free reproduced by execution (t2039_c5_harness.exe: full 672-
	// dispatch chain recorded correctly, fence-waited correctly, readback returned ALL
	// ZEROS -- the exact signature of the GPU having read/copied from resources already
	// released by the time it got to them). Captured here, released only when this
	// token itself is freed (RunLayerLoopGpuFinish, after the fence-wait completes).
	Microsoft::WRL::ComPtr<ID3D12Resource> layout_buf;
	Microsoft::WRL::ComPtr<ID3D12Resource> rope_buf;
	Microsoft::WRL::ComPtr<ID3D12Resource> model_const_buf;
	Microsoft::WRL::ComPtr<ID3D12Resource> silu_lut_buf;
	Microsoft::WRL::ComPtr<ID3D12Resource> scratch_layout_buf;
	Microsoft::WRL::ComPtr<ID3D12Resource> seq_uav;
	Microsoft::WRL::ComPtr<ID3D12Resource> scratch_uav;
	Microsoft::WRL::ComPtr<ID3D12Resource> work_scratch_uav;
	Microsoft::WRL::ComPtr<ID3D12Resource> lw_upload_keep_alive;
	std::vector<Microsoft::WRL::ComPtr<ID3D12Resource>> upload_keep_alive;
};

// T-2113 (B10 lever 1b): the group-cooperative lane count for the fused stage-1 reduction
// (site_common.hlsli's ApplyFusedAdapterDeltaGpu) -- largest power of two <= 256/rank, so `rank`
// k-values each get `stage1_lanes` threads cooperating on one in_channels-wide int64 reduction
// via a fixed groupshared tree (GemmCoalescedGpuAt's own construction, D-SLM3334's own blessed
// shape: integer addition is exactly associative/commutative, so a tree reduction returns the
// identical int64 a single-thread serial sum returns, regardless of order) instead of one thread
// alone doing the whole in_channels-wide reduction serially -- the low-occupancy driver §16.5
// (Claude/Brunel/t2113-1p0-core-build-2026-08-15.md) named and this section's own per-site
// decomposition confirmed by execution. Safe for any rank >= 1 -- the shader's own wave loop
// covers rank > 256/stage1_lanes by iterating multiple waves; rank == 0 (uncovered slot) never
// reaches the shader's reduction body, ApplyFusedAdapterDeltaGpu's own no-op sentinel
// short-circuits first. A free function, NOT a lambda local to RunLayerLoopGpuSubmit -- its own
// two `return` statements would otherwise be swept into
// tests/ci/check_gpu_guard_status_parity.py's own full-brace-matched scan of that function's
// body (that checker's own docstring: a text-level return-statement count, not a real C++
// parse), miscounting two ordinary uint32_t returns as new LastWeightUploadWasSkipped decision
// paths -- reproduced by execution during this section's own build (22 real vs 20 documented,
// the exact +2 this lambda's own two returns predict) before being moved here.
uint32_t Stage1LanesForRank(uint32_t rank) {
	if (rank == 0) return 1u;
	uint32_t want = 256u / rank;
	if (want == 0) want = 1u;
	uint32_t lanes = 1u;
	while (lanes * 2u <= want) lanes *= 2u;
	return lanes;
}

// T-2113 (B5, design Sec4.2/Sec6.2/Sec10 B5): the SUBMIT half of the async split --
// declared in gpu_port.h. This function is BYTE-FOR-BYTE what RunLayerLoopGpu's own
// body always was, from function entry through ExecuteCommandLists/Signal, MINUS the
// T-2169 (Rung 2, design Claude/Vitruvius/t2169-gpu-batched-prefill-design-2026-08-18.md Sec5/
// Sec5.1, D-SLM3595): the per-token, per-layer dispatch body, extracted from
// RunLayerLoopGpuSubmit's own former inline loop -- list-lifecycle separation, not a
// call-and-compose reuse. Records exactly one token's full [start_layer, start_layer +
// layers_to_record) dispatch chain into the ALREADY-OPEN command list on `dev` -- it does not
// Reset(), Close(), Execute, or fence-wait; the caller (RunLayerLoopGpuSubmit for the
// single-token, non-chunked call shape, or the T-2169 chunk-submission primitive for the
// chunked shape) owns the list's own open/close lifecycle. The root signature and every one of
// the twelve SRV/UAV root bindings (slots 8-14) must already be bound by the caller before this
// is invoked -- unchanged from the pre-extraction shape, since those bindings are call-constant
// (D-SLM3632, Sec5.1) and persist on an open command list across Dispatch/SetPipelineState
// calls; this function issues only root CONSTANTS (slot 0) plus Dispatch/ResourceBarrier pairs,
// byte-for-byte identical to the pre-extraction `bind_and_dispatch`/`bind_and_dispatch_tail`
// calls for the same (layer, site). `position_u32`/`width_u32` are THIS TOKEN's own record-time
// values (D-SLM3612, Sec5 2a) -- a chunked caller passes a chunk-local, per-token-advancing
// pair rather than a call-wide constant read from `seq.context_length`; the single-token caller
// passes the identical constant it always computed. `dispatch_query_index` is threaded by
// reference so the timestamp-query boundary numbering stays contiguous across every token a
// (sub-)chunk records, matching the pre-extraction single-call numbering exactly when only one
// token is recorded (Rung 2's own refactor-safety cell, Sec5/Sec8, proves this).
void RecordOneTokenFullDepthDispatchBody(
    harness::Device& dev, uint32_t start_layer, uint32_t layers_to_record, uint32_t H, uint32_t HD,
    uint32_t NH, uint32_t NQH, uint32_t I, uint32_t N, uint32_t context_cap_u32,
    uint32_t position_u32, uint32_t width_u32, const GpuScratchLayout& scratch_layout,
    uint64_t work_wide_a_off, uint64_t work_wide_b_off, uint64_t work_adapter_u_off,
    const GpuAdapterBridge* adapter_bridge, uint32_t& dispatch_query_index) {
	auto& attn_norm_pipe = harness::GetOrBuildComposedPipeline("attn_norm_site");
	auto& q_proj_pipe = harness::GetOrBuildComposedPipeline("q_proj_site");
	auto& kv_proj_gemm_pipe = harness::GetOrBuildComposedPipeline("kv_proj_gemm_site");
	auto& kv_proj_pipe = harness::GetOrBuildComposedPipeline("kv_proj_site");
	auto& rope_pipe = harness::GetOrBuildComposedPipeline("rope_guard_site");
	auto& rope_commit_pipe = harness::GetOrBuildComposedPipeline("rope_commit_site");
	auto& attention_score_pipe = harness::GetOrBuildComposedPipeline("attention_score_site");
	auto& softmax_pipe = harness::GetOrBuildComposedPipeline("softmax_site");
	auto& context_accumulate_pipe = harness::GetOrBuildComposedPipeline("context_accumulate_site");
	auto& ctx_fold_pipe = harness::GetOrBuildComposedPipeline("ctx_fold_site");
	auto& o_proj_pipe = harness::GetOrBuildComposedPipeline("o_proj_site");
	auto& attn_residual_pipe = harness::GetOrBuildComposedPipeline("attn_residual_site");
	auto& mlp_norm_pipe = harness::GetOrBuildComposedPipeline("mlp_norm_site");
	auto& gate_proj_pipe = harness::GetOrBuildComposedPipeline("gate_proj_site");
	auto& up_proj_pipe = harness::GetOrBuildComposedPipeline("up_proj_site");
	auto& mlp_act_pipe = harness::GetOrBuildComposedPipeline("mlp_act_site");
	auto& down_proj_pipe = harness::GetOrBuildComposedPipeline("down_proj_site");
	auto& mlp_residual_pipe = harness::GetOrBuildComposedPipeline("mlp_residual_site");
	auto& commit_pipe = harness::GetOrBuildComposedPipeline("commit_site");
	auto& q_proj_gemm_pipe = harness::GetOrBuildComposedPipeline("q_proj_gemm_site");
	auto& o_proj_gemm_pipe = harness::GetOrBuildComposedPipeline("o_proj_gemm_site");
	auto& gate_proj_gemm_pipe = harness::GetOrBuildComposedPipeline("gate_proj_gemm_site");
	auto& up_proj_gemm_pipe = harness::GetOrBuildComposedPipeline("up_proj_gemm_site");
	auto& down_proj_gemm_pipe = harness::GetOrBuildComposedPipeline("down_proj_gemm_site");

	D3D12_RESOURCE_BARRIER global_uav_barrier{};
	global_uav_barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
	global_uav_barrier.UAV.pResource = nullptr;  // Sec18.3 fold, D-SLM3002: every barrier this design issues is global

	struct TailAdapterSlot {
		int slot_p;
		uint32_t in_base;
		uint32_t wide_base;
	};

	auto bind_and_dispatch = [&](ID3D12PipelineState* pso, uint32_t layer_index, uint32_t num_groups = 1,
	                              uint32_t lanes = 1) {
		if (dispatch_query_index < harness::Device::kMaxTimestampSlots - 1) {
			dev.list->EndQuery(dev.timestamp_heap.Get(), D3D12_QUERY_TYPE_TIMESTAMP, dispatch_query_index);
		}
		++dispatch_query_index;
		uint32_t consts[11] = {layer_index, H,        HD, NH, context_cap_u32, position_u32,
		                        NQH,        width_u32, I,  N,  lanes};
		dev.list->SetComputeRoot32BitConstants(0, 11, consts, 0);
		dev.list->SetPipelineState(pso);
		dev.list->Dispatch(num_groups, 1, 1);
		dev.list->ResourceBarrier(1, &global_uav_barrier);
	};
	auto bind_and_dispatch_tail = [&](ID3D12PipelineState* pso, uint32_t layer_index,
	                                   std::initializer_list<TailAdapterSlot> slots) {
		if (dispatch_query_index < harness::Device::kMaxTimestampSlots - 1) {
			dev.list->EndQuery(dev.timestamp_heap.Get(), D3D12_QUERY_TYPE_TIMESTAMP, dispatch_query_index);
		}
		++dispatch_query_index;
		uint32_t consts[27] = {layer_index, H,        HD, NH, context_cap_u32, position_u32,
		                        NQH,        width_u32, I,  N,  /*lanes=*/1u};
		size_t base = 11;
		for (const TailAdapterSlot& s : slots) {
			uint32_t rank = 0, a_off = 0, b_off = 0, fold_off = 0;
			if (adapter_bridge) {
				const superslm_gpu::AdapterProjSlot& slot =
				    adapter_bridge->slots[static_cast<size_t>(layer_index) * 7u + static_cast<size_t>(s.slot_p)];
				if (slot.present) {
					rank = slot.rank;
					a_off = static_cast<uint32_t>(slot.a_offset);
					b_off = static_cast<uint32_t>(slot.b_offset);
					fold_off = static_cast<uint32_t>(slot.fold_offset);
				}
			}
			consts[base + 0] = rank;
			consts[base + 1] = a_off;
			consts[base + 2] = b_off;
			consts[base + 3] = fold_off;
			consts[base + 4] = static_cast<uint32_t>(work_adapter_u_off);
			consts[base + 5] = s.in_base;
			consts[base + 6] = s.wide_base;
			consts[base + 7] = Stage1LanesForRank(rank);
			base += 8;
		}
		dev.list->SetComputeRoot32BitConstants(0, 27, consts, 0);
		dev.list->SetPipelineState(pso);
		dev.list->Dispatch(1, 1, 1);
		dev.list->ResourceBarrier(1, &global_uav_barrier);
	};

	const uint32_t KV2 = 2u * NH * HD;
	const GpuGemmSiteGroupPlan q_proj_plan = ComputeGpuGemmSiteGroupPlan(GpuGemmSplitSite::QProj, H, KV2, I);
	const GpuGemmSiteGroupPlan o_proj_plan = ComputeGpuGemmSiteGroupPlan(GpuGemmSplitSite::OProj, H, KV2, I);
	const GpuGemmSiteGroupPlan kv_proj_plan = ComputeGpuGemmSiteGroupPlan(GpuGemmSplitSite::KvProj, H, KV2, I);
	const GpuGemmSiteGroupPlan gate_proj_plan = ComputeGpuGemmSiteGroupPlan(GpuGemmSplitSite::GateProj, H, KV2, I);
	const GpuGemmSiteGroupPlan up_proj_plan = ComputeGpuGemmSiteGroupPlan(GpuGemmSplitSite::UpProj, H, KV2, I);
	const GpuGemmSiteGroupPlan down_proj_plan = ComputeGpuGemmSiteGroupPlan(GpuGemmSplitSite::DownProj, H, KV2, I);

	const uint32_t attn_score_groups = (NQH * width_u32 + 255u) / 256u;
	const uint32_t ctx_accum_groups = (NQH * HD + 255u) / 256u;
	const uint32_t rope_groups = (NQH * (HD / 2u) + 255u) / 256u;

	for (const GpuGemmSiteGroupPlan* plan : {&q_proj_plan, &o_proj_plan, &kv_proj_plan, &gate_proj_plan,
	                                          &up_proj_plan, &down_proj_plan}) {
		if (static_cast<uint64_t>(plan->groups) * static_cast<uint64_t>(plan->channels_per_group) <
		    static_cast<uint64_t>(plan->out_channels)) {
			throw GpuGemmGroupArithmeticError(
			    "RecordOneTokenFullDepthDispatchBody: GEMM group plan does not cover its own "
			    "out_channels -- groups * channels_per_group fell short of out_channels");
		}
	}

	for (uint32_t i = 0; i < layers_to_record; ++i) {
		const uint32_t l = start_layer + i;
		bind_and_dispatch(attn_norm_pipe.pso.Get(), l);
		bind_and_dispatch(q_proj_gemm_pipe.pso.Get(), l, q_proj_plan.groups, q_proj_plan.lanes);
		bind_and_dispatch_tail(q_proj_pipe.pso.Get(), l,
		                        {{/*q=*/0, scratch_layout.normed, static_cast<uint32_t>(work_wide_a_off)}});
		bind_and_dispatch(kv_proj_gemm_pipe.pso.Get(), l, kv_proj_plan.groups, kv_proj_plan.lanes);
		bind_and_dispatch_tail(
		    kv_proj_pipe.pso.Get(), l,
		    {{/*k=*/5, scratch_layout.normed, static_cast<uint32_t>(work_wide_a_off)},
		     {/*v=*/6, scratch_layout.normed, static_cast<uint32_t>(work_wide_b_off)}});
		bind_and_dispatch(rope_pipe.pso.Get(), l, rope_groups);
		bind_and_dispatch(rope_commit_pipe.pso.Get(), l, rope_groups);
		bind_and_dispatch(attention_score_pipe.pso.Get(), l, attn_score_groups);
		bind_and_dispatch(softmax_pipe.pso.Get(), l);
		bind_and_dispatch(context_accumulate_pipe.pso.Get(), l, ctx_accum_groups);
		bind_and_dispatch(ctx_fold_pipe.pso.Get(), l);
		bind_and_dispatch(o_proj_gemm_pipe.pso.Get(), l, o_proj_plan.groups, o_proj_plan.lanes);
		bind_and_dispatch_tail(o_proj_pipe.pso.Get(), l,
		                        {{/*o=*/1, scratch_layout.ctx_codes, static_cast<uint32_t>(work_wide_a_off)}});
		bind_and_dispatch(attn_residual_pipe.pso.Get(), l);
		bind_and_dispatch(mlp_norm_pipe.pso.Get(), l);
		bind_and_dispatch(gate_proj_gemm_pipe.pso.Get(), l, gate_proj_plan.groups, gate_proj_plan.lanes);
		bind_and_dispatch_tail(gate_proj_pipe.pso.Get(), l,
		                        {{/*gate=*/2, scratch_layout.normed, static_cast<uint32_t>(work_wide_a_off)}});
		bind_and_dispatch(up_proj_gemm_pipe.pso.Get(), l, up_proj_plan.groups, up_proj_plan.lanes);
		bind_and_dispatch_tail(up_proj_pipe.pso.Get(), l,
		                        {{/*up=*/3, scratch_layout.normed, static_cast<uint32_t>(work_wide_a_off)}});
		bind_and_dispatch(mlp_act_pipe.pso.Get(), l);
		bind_and_dispatch(down_proj_gemm_pipe.pso.Get(), l, down_proj_plan.groups, down_proj_plan.lanes);
		bind_and_dispatch_tail(down_proj_pipe.pso.Get(), l,
		                        {{/*down=*/4, scratch_layout.act_codes, static_cast<uint32_t>(work_wide_a_off)}});
		bind_and_dispatch(mlp_residual_pipe.pso.Get(), l);
		bind_and_dispatch(commit_pipe.pso.Get(), l);
	}
}

// T-2169 (Rung 2b-prep, design Sec5.1, D-SLM3632/D-SLM3633): the guard ladder, the weight/rope/K-V
// pack-and-residency decision, and the once-per-call root-signature/twelve-SRV-UAV-binding setup,
// extracted verbatim from RunLayerLoopGpuSubmit's own former inline body -- a mechanical
// relocation, not a rewrite; every line below is byte-for-byte what RunLayerLoopGpuSubmit used to
// execute in this same order, only now reachable from a second caller (the chunk-submission
// primitive, T-2169 Rung 2b) without duplicating this block's own ~900 lines of residency-cache
// and guard-ladder logic a second time. Design Sec5.1 traces why this whole block is sound to run
// exactly ONCE per Submit-equivalent call -- whether that call covers one token
// (RunLayerLoopGpuSubmit's own existing shape) or a whole admitted chunk (the primitive) -- since
// every one of the nine guards, and every one of the three residency-cache keys, tests only
// call-level state that does not vary token-to-token within one call (D-SLM3632/D-SLM3633).
//
// Guard rejection is a plain, side-effect-free `return` (matching every guard's own pre-extraction
// shape) -- reached before `dev.alloc->Reset()` ever runs, so the caller's own enclosing try/catch
// (opened around THIS function's own call, not inside it) never sees a guard rejection as an
// exception; it sees an ordinary non-Ok status and returns early itself, with no command list ever
// opened and therefore nothing to close. Once the guards pass, every subsequent allocation
// (`dev.Upload`/`dev.MakeBuffer`, all `SSLM_GPU_HR`-guarded) can throw `std::runtime_error` exactly
// as it always could -- this function installs no try/catch of its own and lets such a throw
// propagate to the caller's enclosing try, which is where the existing
// `invalidate_residency_caches_on_throw`/`dev.list->Close()`/status-mapping catch clauses still
// live, unchanged, because they are also used by the dispatch-recording and readback code that
// still runs in the caller after this function returns successfully.
struct GpuLayerLoopChunkOpenState {
	uint32_t H = 0, HD = 0, NH = 0, NQH = 0, I = 0, N = 0, context_cap_u32 = 0;
	GpuScratchLayout scratch_layout;
	uint64_t work_wide_a_off = 0, work_wide_b_off = 0, work_adapter_u_off = 0;
	Microsoft::WRL::ComPtr<ID3D12Resource> seq_uav;
	Microsoft::WRL::ComPtr<ID3D12Resource> kv_uav;
	Microsoft::WRL::ComPtr<ID3D12Resource> scratch_uav;
	Microsoft::WRL::ComPtr<ID3D12Resource> work_scratch_uav;
	Microsoft::WRL::ComPtr<ID3D12Resource> layout_buf;
	Microsoft::WRL::ComPtr<ID3D12Resource> rope_buf;
	Microsoft::WRL::ComPtr<ID3D12Resource> model_const_buf;
	Microsoft::WRL::ComPtr<ID3D12Resource> silu_lut_buf;
	Microsoft::WRL::ComPtr<ID3D12Resource> scratch_layout_buf;
	Microsoft::WRL::ComPtr<ID3D12Resource> lw_upload_keep_alive;
	std::vector<Microsoft::WRL::ComPtr<ID3D12Resource>> upload_keep_alive;
};

superslm::SslmForwardStatus PrepareGpuLayerLoopChunkOpenState(
    superslm::SequenceLayerState& seq, const superslm::LayerWeights* layers,
    uint32_t num_hidden_layers, uint32_t layer_budget, size_t hidden_size, size_t head_dim,
    size_t num_key_value_heads, size_t intermediate_size, int64_t context_cap,
    const superslm::SslmTensorManifest& rope_tables, uint8_t* workspace, size_t workspace_size,
    ID3D12Resource* external_kv_resident, bool* io_external_kv_needs_resume_barrier,
    ID3D12Resource* external_weights_resident, ID3D12Resource* external_rope_cos_resident,
    ID3D12Resource* external_rope_sin_resident, bool external_rope_has,
    uint64_t external_rope_cos_elems, uint64_t external_rope_sin_elems,
    const GpuAdapterBridge* adapter_bridge, GpuLayerLoopChunkOpenState* out_state) {
	// T-2055 (Claude/Poirot/db73b22-gpu-serial-port-final-confirmation-
	// review.md, P2): set BEFORE every one of this function's eleven
	// rejecting return paths (the nine guards below, `!dev.available`, and
	// the Tier-3 preflight check) so `LastWeightUploadWasSkipped()`'s own
	// documented contract ("set internally... every call") is actually true
	// of every call, not only of the calls that reach the weight-residency
	// decision (`weights_resident`, this function's own residency branch,
	// well below this point). Before this fix the accessor held the
	// PREVIOUS call's answer across a rejecting call -- reproduced by
	// execution: a probe driving guard-rejected calls read back a stale,
	// sometimes-Ok-sometimes-not `skipped` value that had nothing to do with
	// the call that had just run. A rejected call makes no weight-residency
	// decision at all, so `false` ("no upload was skipped") is the honest
	// answer for one, matching the reject-over-silently-degrade shape every
	// other guard in this function already follows.
	g_last_weight_upload_was_skipped = false;  // ANCHOR:lwuws_write_function_entry
	// T-2101: reset at function entry, for the identical reason the line above is -- a call
	// rejected by any guard below never reaches the recording-window reset further down, and
	// without this line would report the PREVIOUS successful call's own stale timing rather than
	// the honest "nothing was timed" answer `LastCallTiming()`'s own header comment documents.
	g_last_call_timing = GpuCallTiming{};
	g_last_call_per_dispatch_ms.clear();
	// T-2052 (Claude/Poirot/36b9327-gpu-serial-port-reconfirmation-review.md,
	// M1, correcting T-2049's own N1): CPU parity, corrected a SECOND time.
	// T-2049's own comment here claimed "All eight [guards] now run here" --
	// CPU checks NINE in the guard ladder of `RunLayerLoopImpl`
	// (`forward_sites.cpp`), not eight, and the missing one (`seq.hidden_codes == nullptr` ->
	// `InvalidHiddenCodes`) was reproduced crashing
	// this process with `STATUS_ACCESS_VIOLATION` (0xC0000005) on a
	// DEFAULT-CONSTRUCTED `SequenceLayerState` -- the exact input
	// `sslm_seq_restore`'s own documented contract produces
	// (`RestoreGpuSequenceState`'s own header comment: "`out_seq->hidden_codes`
	// is left untouched -- caller-owned pointer"). All nine guards now run
	// here, in CPU's own source order, before `harness::GetDevice()` is even
	// called -- none of them needs a device to answer, and CPU's own
	// "seq/workspace left untouched on rejection" contract is satisfied by
	// construction: a return here issues no upload, no dispatch, no
	// readback, so nothing GPU-side is ever touched.
	//
	// STRUCTURAL closure (M1's own remedy, not a fourth hand-count) -- OF THE
	// LADDER'S OWN INTERNAL CONSISTENCY, not of drift against CPU (corrected
	// 2026-08-14, T-2055, Claude/Poirot/db73b22-gpu-serial-port-final-
	// confirmation-review.md, P1; D-SLM3183, superseding D-SLM3182's own
	// claim): every guard below is tagged with its own `GpuLayerLoopGuard`
	// enum value (`GpuLayerLoopGuard` (`gpu_port.h`), generated from
	// `gpu_layer_loop_guards.def`)
	// in a trailing comment, and the `static_assert` immediately after this
	// ladder ties the number of guards a maintainer believes were written
	// here to `GpuLayerLoopGuard::kCount` -- the SAME compile-time constant
	// the pin round's own table-walk cell (Curie's work) asserts against.
	// That ties a literal to a constant; NEITHER is compared against
	// `forward_sites.cpp` itself, so this does not, on its own, close the
	// three-hand-counts-produced-three-different-numbers class the paragraph
	// used to claim it closed -- proven false by execution: a tenth guard
	// added here AND to CPU's own ladder, with no matching `.def` row, left
	// this `static_assert`, the table-walk cell, and the full suite all
	// green. `tests/ci/check_gpu_guard_status_parity.py` (T-2055) is what
	// reads `forward_sites.cpp` and closes that class -- this ladder's own
	// tagging and `static_assert` remain useful (a guard added here with no
	// `.def` row, or vice versa, still fails to compile or fails that CI
	// check), just not sufficient alone against a drift that touches CPU's
	// own source.
	if (layer_budget == 0) return superslm::SslmForwardStatus::InvalidLayerBudget;  // LayerBudgetZero
	if (context_cap < 1) return superslm::SslmForwardStatus::InvalidContextCap;  // ContextCapNonPositive
	// HeadDimGeometryMismatch / KvHeadGeometryMismatch (the CFG1 geometry join in
	// `RunLayerLoopImpl` (`forward_sites.cpp`)), checked on THIS call's own caller-supplied
	// hidden_size/head_dim/num_key_value_heads exactly as CPU checks it --
	// never assumed sound because some other caller (the loader's own
	// ValidateConfigGeometryJoin) already checked an artifact-sourced
	// instance of the same triple.
	const size_t guard_num_heads = head_dim == 0 ? 0 : hidden_size / head_dim;
	if (guard_num_heads == 0 || guard_num_heads * head_dim != hidden_size) {
		return superslm::SslmForwardStatus::HeadDimGeometryMismatch;  // HeadDimGeometryMismatch
	}
	if (num_key_value_heads == 0 || num_key_value_heads > guard_num_heads ||
	    guard_num_heads % num_key_value_heads != 0) {
		return superslm::SslmForwardStatus::KvHeadGeometryMismatch;  // KvHeadGeometryMismatch
	}
	// WorkspaceSizeOrOverflow (the KV-size overflow guard in `RunLayerLoopImpl`
	// (`forward_sites.cpp`)): the overflow-guarded KV-size product, bit-exact against CPU's own
	// factor-by-factor `SIZE_MAX / factor` idiom (the third factor is
	// num_key_value_heads, not guard_num_heads -- T-1654/S3.8a's own
	// correction, matching KeyRow/ValueRow's real addressing) -- an
	// overflowing product returns CPU's own InvalidContextCap, never
	// silently wrapping into a workspace_size comparison that could pass on
	// a wrapped-small value.
	{
		size_t kv_bytes_needed = static_cast<size_t>(num_hidden_layers);
		const size_t kv_factors[] = {static_cast<size_t>(context_cap), num_key_value_heads, head_dim, 2u};
		for (size_t factor : kv_factors) {
			if (factor != 0 && kv_bytes_needed > SIZE_MAX / factor) {
				return superslm::SslmForwardStatus::InvalidContextCap;  // WorkspaceSizeOrOverflow (overflow branch)
			}
			kv_bytes_needed *= factor;
		}
		if (workspace == nullptr) return superslm::SslmForwardStatus::WorkspaceTooSmall;  // WorkspaceSizeOrOverflow
		if (workspace_size < kv_bytes_needed) return superslm::SslmForwardStatus::WorkspaceTooSmall;  // WorkspaceSizeOrOverflow
	}
	// HiddenCodesNull (the null-hidden_codes guard in `RunLayerLoopImpl`
	// (`forward_sites.cpp`)) --
	// CPU's own comment there names exactly this input -- a default-constructed SequenceLayerState, which
	// is what `int8_t* hidden_codes = nullptr;`'s own default member
	// initializer produces -- as "used to be dereferenced unconditionally...
	// rejected here instead of left to crash the process." Checked in CPU's
	// own position: after the workspace-size guard, before
	// SequenceAlreadyComplete.
	if (seq.hidden_codes == nullptr) {
		return superslm::SslmForwardStatus::InvalidHiddenCodes;  // HiddenCodesNull
	}
	if (seq.layer_index >= num_hidden_layers) {
		return superslm::SslmForwardStatus::SequenceAlreadyComplete;  // SequenceAlreadyComplete
	}
	// PositionOverCap / KvCapacityExhausted (the two `seq.context_length` domain guards in
	// `RunLayerLoopImpl` (`forward_sites.cpp`)) -- KvCapacityExhausted is
	// the one T-2049's own confirmation review reproduced landing K/V bytes
	// past the addressed row before rejecting.
	if (seq.context_length < 0) return superslm::SslmForwardStatus::PositionOverCap;  // PositionOverCap
	if (seq.context_length >= context_cap) {
		return superslm::SslmForwardStatus::KvCapacityExhausted;  // KvCapacityExhausted
	}
	static_assert(static_cast<int>(superslm_gpu::GpuLayerLoopGuard::kCount) == 9,
	              "RunLayerLoopGpu's own guard ladder above must implement exactly as many guards "
	              "as gpu_layer_loop_guards.def enumerates -- update both together");

	harness::Device& dev = harness::GetDevice();
	if (!dev.available) return superslm::SslmForwardStatus::KvPrecisionUnsupported;

	// T-2045 (S2 partial, Claude/Poirot/82cfca7-gpu-serial-port-build-review.md):
	// §5.1's own Tier-3 hardware floor (D-SLM3000) is now enforced on the path
	// that actually runs, not only on B3's own isolated test entry points --
	// closing the half of S2 this remedy round's own scope reaches. The
	// composed pipeline's own binding SUBSTRATE (one packed SRV buffer +
	// host-computed offset table, rather than §5.1's literal descriptor-table
	// architecture B3 proves generically) is unchanged here -- a full
	// migration is a separate, larger remedy, named explicitly rather than
	// silently left as a completed item; see this ticket's own handoff.
	if (!MapModelGpuResidencyTierCheck()) {
		return superslm::SslmForwardStatus::KvPrecisionUnsupported;
	}

	const uint32_t H = static_cast<uint32_t>(hidden_size);
	const uint32_t HD = static_cast<uint32_t>(head_dim);
	const uint32_t NH = static_cast<uint32_t>(num_key_value_heads);
	const uint32_t KV = NH * HD;
	const uint32_t N = num_hidden_layers;
	// This design's build target carries no GQA-group config field distinct
	// from what hidden_size/head_dim/num_key_value_heads already fix -- every
	// fixture in this suite is MHA-degenerate (num_attention_heads ==
	// num_key_value_heads == 1), and hidden_size == num_attention_heads *
	// head_dim (the CFG1 geometry join, model.h) gives num_attention_heads
	// directly: hidden_size / head_dim.
	const uint32_t NQH = (HD > 0) ? (H / HD) : 0;
	const uint32_t I = static_cast<uint32_t>(intermediate_size);
	const GpuLayerLayout layout = ComputeLayerLayout(H, KV, NH, NQH, I);

	// T-2045 (S3, Claude/Poirot/82cfca7-gpu-serial-port-build-review.md): §5.3's
	// own decision is "weight buffers upload once... and stay resident for the
	// mapped model handle's entire lifetime -- never re-uploaded per token or
	// per sslm_decode_step_gpu call." This function has no separate "map"
	// entry point in gpu_port.h's own contract, so the packed row is always
	// (re)computed here -- cheap, host-local -- and then compared BYTE-FOR-BYTE
	// against the last-uploaded row (ResidentWeights' own header comment: a
	// pointer-only key is unsound against this project's own test harness).
	// Only a genuine content change re-uploads; the real production call
	// pattern (the same `layers` array driving every token of one decode
	// session, tools/sslm_generate.cpp's own RunWholeToken closure) packs the
	// identical bytes every call and skips the ~1.31 GiB PCIe transfer on
	// every call after the first.
	Microsoft::WRL::ComPtr<ID3D12Resource> lw_buf;
	// T-2049 (N3): the upload-heap temporary behind a fresh DEFAULT-heap copy
	// (below) -- must stay alive through ExecuteCommandLists + the fence wait,
	// declared here (outer scope) rather than inside the upload's own `if`
	// block for exactly that reason.
	Microsoft::WRL::ComPtr<ID3D12Resource> lw_upload_keep_alive;

	// T-2101: shared by both residency caches below (weights and, further down, K/V). `seq` is the
	// caller's own proof of continuity for the SEQUENCE this call belongs to -- `context_length == 0`
	// means no position has ever been committed, `layer_index == 0` means no layer of the in-flight
	// token has committed either (`SequenceLayerState`'s own header comment, `forward_sites.h`).
	// Reproduced by execution during this ticket's own build: gating a pointer-identity fast path on
	// this signal is what closes a regression this build discovered in T-2100's own bottleneck-1 fix
	// (see `lw_fast_hit`'s own comment immediately below) -- the suite's own per-test fixtures
	// construct one fresh, independently-built `NLayerFixture`/workspace PER TEST, at whatever
	// address the allocator's free list hands back, and every one of those calls is a fresh sequence
	// by construction (`layer_index = 0`, `context_length` default-initialized to 0). A genuine
	// continuation (`layer_index` or `context_length` nonzero) can only be reached by a PRIOR commit
	// having already run against THIS SAME call's own arguments, which is the caller-continuity
	// guarantee a pointer-identity fast path is sound to trust; a fresh sequence carries no such
	// guarantee and always pays the real check.
	const bool fresh_sequence = seq.layer_index == 0 && seq.context_length == 0;

	// --- Pack LayerWeights (Sec5.1's own read-resource list, this checkpoint's
	// own scoped subset: only what sites 1-4 read). Always runs (S3 above). ---
	// T-2100: skip the pack entirely when the source is identical to what is already resident.
	//
	// CORRECTED 2026-08-14 (T-2101): the identity check above (pointer + layer count + stride,
	// with no `fresh_sequence` term) is exactly the "pointer-only key is unsound" hazard
	// `ResidentWeights`' own header comment already documents and reproduced once, for `layers` --
	// T-2100 reintroduced it as a fast-path BYPASS of the content compare, and this build's own full
	// suite run reproduced it a second time: 22 real failures (`TestT2019_B11_SequenceLayerState
	// Complete_QProj`/`KvProj_KBiasAndVBias`/`RoPE`/`DownProj`, `T2053`, `T2083`), every one a fresh
	// `NLayerFixture` whose address the allocator handed back from a just-freed, unrelated prior
	// fixture -- `lw_fast_hit` served that prior fixture's STALE resident weights across the fixture
	// boundary with no error. T-2100's own build log (`Claude/Brunel/t2025-gpu-serial-build-2026-08-
	// 13.md` Sec30) never re-ran the full suite after adding this bypass, so the regression shipped
	// undetected. `fresh_sequence` closes it: gating the bypass so it never fires for a fresh
	// sequence costs nothing in real decode (the first call of any sequence is already a cache miss
	// by construction, and every later call is unaffected), and it cannot make the cache LESS sound
	// than before T-2100 -- it can only route MORE calls to the pre-T-2100, proven-sound full compare.
	// T-2113 (B5, design Sec5.1/Sec10 B5): when the caller supplies its own already-
	// resident weight buffer (an SslmGpuModelHandle's own `weights_buf`, uploaded once
	// inside `sslm_gpu_model_map`, design Sec5.1), this call never touches `layers`,
	// never re-packs, never consults `g_resident_weights` at all -- the model handle's
	// own residency IS the weight residency for this call, on the identical footing
	// `external_kv_resident` (T-2113 B3) already established for K/V. `layers` may be
	// null on this path (never dereferenced).
	const bool external_weights = external_weights_resident != nullptr;
	const bool lw_fast_hit = !external_weights && !fresh_sequence && g_resident_weights.valid &&
	                         g_resident_weights.src_layers == static_cast<const void*>(layers) &&
	                         g_resident_weights.src_n == N &&
	                         g_resident_weights.src_stride == layout.stride;
	// T-2113 (B2): the pack loop this call site used to contain inline is now
	// PackLayerWeightsBytes (extracted verbatim, above ComputeLayerLayout) -- shared with
	// gpu_1p0.cpp's own sslm_gpu_model_map. Behavior unchanged: still only computed on a
	// !lw_fast_hit miss, and never at all on the external-weights path (B5).
	std::vector<uint8_t> lw_bytes;
	if (!lw_fast_hit && !external_weights) {
		lw_bytes = PackLayerWeightsBytes(layers, N, layout, H, KV, NH, NQH, I);
	}  // T-2100: end of the !lw_fast_hit pack guard

	// T-2045 (S3): true iff this call's freshly-packed row is byte-identical
	// to the last-uploaded one -- the ONLY sound invalidation signal here
	// (ResidentWeights' own header comment).
	//
	// CORRECTED 2026-08-14 (T-2075, Claude/Poirot/
	// 72a9b0d-gpu-serial-port-final-review.md, S1; D-SLM3228): T-2071's own
	// residency-forcing term here (removed by this correction, not merely
	// disclosed) neutralised the very pin it was built to arm. Forcing a
	// MISS on every armed call meant the miss path's own two remedy actions
	// -- `g_last_weight_upload_was_skipped = weights_resident;` below, and
	// `g_resident_weights.lw_buf.Reset(); g_resident_weights.valid = false;`
	// at the allocation site -- ran BEFORE the injected throw was ever
	// reached, so by the time the catch ran, the observable already read
	// `false` and the cache was already invalid: the catch's own two
	// remedy lines had nothing left to change. Measured: deleting either or
	// both of those two lines left the suite byte-identical at 33893/3 --
	// only the returned-status assertion discriminated; the pin was half
	// inert. Fixed at the root, not the surface, per the review's own
	// finding: the instrument now arms `work_scratch_uav` (below, outside
	// this `if` entirely -- a real allocation the HIT path also makes),
	// not this one -- so this predicate needs no injection-aware term at
	// all, and reads exactly as it did before T-2071 ever touched it.
	const bool weights_resident =
	    external_weights || lw_fast_hit || (g_resident_weights.valid && g_resident_weights.bytes == lw_bytes);
	if (external_weights) {
		lw_buf = external_weights_resident;  // T-2113 (B5): the model handle's own resident buffer
	} else if (weights_resident) {
		lw_buf = g_resident_weights.lw_buf;
	}
	// T-2052 (item 3): the device-observable Curie §13.2 specified --
	// LastWeightUploadWasSkipped() reads this back, true iff this call's own
	// upload was skipped (a cache hit). T-2055 (P2): this is the SECOND of
	// two assignment sites, reached only once every guard above has already
	// passed -- the function-entry assignment (this function's own opening
	// lines) is what makes the accessor's "every call" contract true of the
	// nine rejecting calls that never reach here too, not this one alone.
	g_last_weight_upload_was_skipped = weights_resident;

	std::vector<uint8_t> layout_bytes(57 * 4, 0);
	for (int i = 0; i < 56; ++i) PutI32At(layout_bytes, static_cast<size_t>(i) * 4, static_cast<int32_t>(layout.off[i]));
	PutI32At(layout_bytes, 56 * 4, static_cast<int32_t>(layout.stride));

	// --- RopeGuardInfo: resolved HOST-SIDE, once, from the SAME
	// SslmTensorManifest::Tensor("cos")/Tensor("sin") lookup RopeApplySite
	// itself performs on CPU (rope_tables is model-wide, shared across every
	// layer, Sec5.6). ---
	// T-2113 (B5, design Sec5.1/Sec10 B5): when the caller supplies its own already-
	// resident RoPE cos/sin buffers (an SslmGpuModelHandle's own `rope_cos_buf`/
	// `rope_sin_buf`, uploaded once inside `sslm_gpu_model_map`), `rope_tables` is
	// never read for presence/size either -- the caller supplies those three small
	// facts directly (`external_rope_has`/`external_rope_cos_elems`/
	// `external_rope_sin_elems`), since a caller routed through the model handle has
	// no live `SslmTensorManifest` of its own to query (B2's own map() does not retain
	// the artifact's manifest past upload). `rope_tables.Tensor(...)` is still called
	// unconditionally below (cheap, a manifest lookup, no device work) so the
	// non-external path is byte-for-byte unchanged; its RESULT is simply unused on the
	// external path.
	const bool external_rope =
	    external_rope_cos_resident != nullptr && external_rope_sin_resident != nullptr;
	const superslm::SslmTensorView* cos_t = rope_tables.Tensor("cos");
	const superslm::SslmTensorView* sin_t = rope_tables.Tensor("sin");
	std::vector<uint8_t> rope_info_bytes(24, 0);
	PutI32At(rope_info_bytes, 0, external_rope ? (external_rope_has ? 1 : 0) : (cos_t != nullptr ? 1 : 0));
	PutI32At(rope_info_bytes, 4, external_rope ? (external_rope_has ? 1 : 0) : (sin_t != nullptr ? 1 : 0));
	{
		uint64_t cec = external_rope ? external_rope_cos_elems : (cos_t != nullptr ? cos_t->elem_count : 0);
		uint64_t sec = external_rope ? external_rope_sin_elems : (sin_t != nullptr ? sin_t->elem_count : 0);
		PutBytesAt(rope_info_bytes, 8, &cec, 8);
		PutBytesAt(rope_info_bytes, 16, &sec, 8);
	}

	// --- SeqState (T-2039: dynamic, per-real-hidden_size layout -- hidden_codes[H]
	// i32, hidden_scale.m/e i64, layer_index u32, kv_sat_lo/hi u32,
	// context_length i64, sticky_status i64; see SeqScaleOff/.../SeqStickyOff
	// above, matched shader-side by site_common.hlsli's own SeqScaleOffGpu
	// family). ---
	std::vector<uint8_t> seq_bytes(SeqTotalSize(H), 0);
	for (uint32_t i = 0; i < H; ++i) PutI32At(seq_bytes, i * 4, static_cast<int32_t>(seq.hidden_codes[i]));
	PutI64At(seq_bytes, SeqScaleOff(H) + 0, seq.hidden_scale.m);
	PutI64At(seq_bytes, SeqScaleOff(H) + 8, seq.hidden_scale.e);
	PutI32At(seq_bytes, SeqLayerIdxOff(H), static_cast<int32_t>(seq.layer_index));
	PutI32At(seq_bytes, SeqSatLoOff(H), static_cast<int32_t>(seq.kv_saturation_count & 0xFFFFFFFFu));
	PutI32At(seq_bytes, SeqSatHiOff(H), static_cast<int32_t>((seq.kv_saturation_count >> 32) & 0xFFFFFFFFu));
	PutI64At(seq_bytes, SeqCtxLenOff(H), seq.context_length);
	PutI64At(seq_bytes, SeqStickyOff(H), 0);  // sticky_status = kTagOk

	// T-2039: LayerScratch's own dynamic, per-real-dims layout (superseding
	// T-2035's fixed 640-byte assumption, Sec13.5's own named blocker) --
	// ComputeScratchLayout above, driven by the real H/I this call carries.
	// T-2045 (C3): also carries num_attention_heads/context_cap now, for the
	// persistent cross-dispatch scores/probs region the de-fused attention
	// sites need.
	const GpuScratchLayout scratch_layout =
	    ComputeScratchLayout(H, I, NQH, static_cast<uint32_t>(context_cap));
	// T-2101: LayerScratch (below, `scratch_uav`) needs no host-supplied initial content -- every
	// byte any site shader reads from it (the codes blocks, the persistent `scores` region) was
	// written earlier in the SAME call, by the SAME 17-dispatch-per-layer sequence (T-2045's own C3
	// comment on the `scores` field above states this for attention's own four dispatches; the same
	// write-before-read property holds for every other block by construction of the site order).
	// The old zeroed `scratch_bytes` vector plus its upload/copy/transition cost O(context_cap)
	// bytes of host->device traffic on EVERY call for content that was always about to be
	// overwritten and never read before being written -- the same class of wasted per-token
	// marshalling as the K/V workspace below, at smaller scale. `scratch_uav` is now created
	// directly in the UAV state with no upload, matching `work_scratch_uav`'s own established idiom
	// two allocations below.

	// T-2101 (D-SLM3301/D-SLM3294): the K/V workspace's own device residency (`ResidentKv` above).
	// A full `workspace_size` host copy is built only on a cache miss (first call, or a different
	// sequence's workspace) -- not on every call, which is what made the old unconditional
	// `std::vector<uint8_t> kv_bytes(workspace, workspace + workspace_size);` here ~1.8 GiB of
	// per-token traffic (a full copy INTO host memory, a full upload, a full readback, and a full
	// copy back OUT of host memory, every call, to write one new K/V row per layer).
	// A `workspace` POINTER match is not, on its own, a sound identity signal here -- the suite's
	// per-test fixtures each construct a fresh, independently zeroed `ws` vector of the identical
	// size, one per test function/loop iteration, and the allocator can hand the freed one's address
	// to the next one (the exact hazard `ResidentWeights`' own header comment already documents for
	// `layers`, and `fresh_sequence`'s own comment above documents reproducing for `layers` a second
	// time in this build). A pointer-only match here would just as readily serve a PRIOR call's
	// stale resident K/V content to a brand-new, unrelated, all-zero workspace with no error.
	//
	// `fresh_sequence` (declared above, shared with `lw_fast_hit`) is the sound discriminator, for
	// the identical reason it is sound for weights: `context_length == 0` means NO position has ever
	// been committed for this sequence (`SequenceLayerState`'s own header comment, `forward_sites.h`),
	// and `layer_index == 0` means no layer of whatever token is in flight has committed either --
	// together they are the caller's own proof that nothing in `workspace` can matter yet, regardless
	// of what stale bytes a reused address might carry. A fresh sequence always forces a real upload;
	// any other state can only be reached by a PRIOR commit having already run against THIS SAME
	// call's own workspace (nothing else advances either field), which is the caller-continuity
	// guarantee the pointer+size identity below is sound to trust. Real decode pays this once per
	// sequence (already the harness's own "one warmup step, discarded" convention) and gets the fast
	// path from the second token on, since `layer_index` resets to 0 every token but `context_length`
	// does not.
	// T-2113 (B3, design Sec5.3): when the caller supplies its own dedicated, already-
	// resident K/V buffer (an SslmGpuSequenceHandle's own buffer, design Sec5.3), this
	// call never touches g_resident_kv at all -- the per-handle buffer IS this sequence's
	// own residency, so there is no "hit"/"miss" decision to make against a shared slot;
	// `kv_fast_hit` stays meaningful only for the pre-1.0 process-global-cache path below.
	const bool external_kv = external_kv_resident != nullptr;
	const bool kv_fast_hit = !external_kv && !fresh_sequence && g_resident_kv.valid &&
	                          g_resident_kv.src_workspace == workspace &&
	                          g_resident_kv.src_size == workspace_size;

	// T-2039/T-2049 (N6, Claude/Poirot/34ef30f-gpu-serial-port-confirmation-
	// review.md, correcting T-2049's own left-behind paragraph per M4,
	// Claude/Poirot/36b9327-gpu-serial-port-reconfirmation-review.md):
	// WorkScratch -- the transient, per-call-sized scratch every real
	// production-geometry site streams a wide row through instead of a
	// fixed-capacity local array (site_common.hlsli/site_common2.hlsli's own
	// header comments). Two adjacent WIDE_A/WIDE_B regions, each `max_width`
	// int64 elements (max_width = max(hidden_size, intermediate_size) --
	// covers every GEMM-funneled site's own widest row, and kv_proj's
	// kacc/vacc pair, which is never wider than hidden_size). The attention
	// score/probs row does NOT live here -- it lives in LayerScratch's own
	// persistent `scores` field (`ScratchLayout` index 25), since C3 (T-2045)
	// split attention into four separate dispatches and the row must survive
	// across the dispatch boundaries between them; WorkScratch's own
	// ATTN_SCORES region (the ORIGINAL, single-fused-dispatch design) was
	// retired for exactly that reason and is no longer allocated (was 64 MiB
	// at `context_cap = 32768`, pure waste once orphaned). Uninitialized on
	// creation: every byte this design ever reads from WorkScratch was
	// written earlier in the SAME dispatch, by the SAME thread, before that
	// read (site_common.hlsli's own per-primitive header comments state this
	// for each cooperative primitive) -- no cross-dispatch or cross-call
	// persistence is needed or assumed for WIDE_A/WIDE_B.
	//
	// T-2049 (N2): ROPE_STAGE resized from 256 (thread-indexed) to
	// `num_attention_heads` (HEAD-indexed) `head_dim`-sized int32 slots --
	// the confirmation review reproduced the thread-indexed version silently
	// corrupting K rows once a single thread owns more than one head
	// (`num_attention_heads > 256`, unreachable at either of this design's
	// real tiers but a real bug in the shape). Indexing by `h` instead of
	// `t` (`rope_guard_site.hlsl`'s own updated `my_stage` computation) means
	// every head's own staged row is independent of which thread processes
	// it, at any head count -- the class of defect is removed, not capped.
	const uint32_t max_width = std::max(H, I);
	const uint64_t work_wide_a_off = 0;
	const uint64_t work_wide_b_off = static_cast<uint64_t>(max_width) * 8u;
	const uint64_t work_rope_stage_off = work_wide_b_off + static_cast<uint64_t>(max_width) * 8u;
	// T-2113 (B6b, design Sec8): ADAPTER_U -- the adapter-delta dispatch's own transient
	// scratch for the narrowed, folded rank-wide intermediate (`u_i8`, forward_sites.cpp's
	// own `AddAmplifyingLoraDelta`), one signed byte per rank element (never a fixed-capacity
	// local array, matching this region's own sibling comments' stated discipline). Sized to
	// the BOUND ADAPTER's own rank (0 when no adapter is bound this call -- `adapter_bridge`
	// is null at every pre-B6b caller and every base-only decode call, so this region costs
	// nothing beyond the 4-byte floor `MakeBuffer` already needs to be a legal resource).
	const uint64_t work_adapter_u_off = work_rope_stage_off + static_cast<uint64_t>(NQH) * static_cast<uint64_t>(HD) * 4u;
	const uint32_t adapter_rank = adapter_bridge ? adapter_bridge->rank : 0;
	const uint64_t work_total = work_adapter_u_off + std::max<uint64_t>(4u, adapter_rank);

	// T-2049 (N6): index 24 (formerly `work_attn_scores_off`) is retired --
	// left unwritten (reads back 0, never consulted by any shader) rather
	// than renumbering every index after it, which would touch every site
	// shader's own hardcoded `ScratchLayout.Load<uint>(N*4)` call for no
	// behavioural change. Index 26 is new: `work_rope_stage_off`, host-
	// computed and shader-read (matching the stated discipline of the composed
	// binding table, `kComposedResourceBindingCount` (`d3d12_harness.h`))
	// rather than re-derived in HLSL from index 24
	// the way `rope_guard_site.hlsl` used to.
	std::vector<uint8_t> scratch_layout_bytes(27 * 4, 0);
	{
		auto put = [&](int idx, uint32_t v) { PutI32At(scratch_layout_bytes, static_cast<size_t>(idx) * 4, static_cast<int32_t>(v)); };
		put(0, scratch_layout.normed); put(1, scratch_layout.normed_scale);
		put(2, scratch_layout.q_codes); put(3, scratch_layout.q_scale);
		put(4, scratch_layout.q_rot);
		put(5, scratch_layout.ctx_codes); put(6, scratch_layout.ctx_scale);
		put(7, scratch_layout.o_codes); put(8, scratch_layout.o_scale);
		put(9, scratch_layout.attn_stream); put(10, scratch_layout.attn_stream_scale);
		put(11, scratch_layout.gate_codes); put(12, scratch_layout.gate_scale);
		put(13, scratch_layout.up_codes); put(14, scratch_layout.up_scale);
		put(15, scratch_layout.act_codes); put(16, scratch_layout.act_scale);
		put(17, scratch_layout.down_codes); put(18, scratch_layout.down_scale);
		put(19, scratch_layout.stream_next); put(20, scratch_layout.stream_next_scale);
		put(21, scratch_layout.total);
		put(22, static_cast<uint32_t>(work_wide_a_off));
		put(23, static_cast<uint32_t>(work_wide_b_off));
		// index 24 retired (N6) -- ATTN_SCORES no longer allocated
		put(25, scratch_layout.scores);  // T-2045 (C3): persistent cross-dispatch scores/probs region
		put(26, static_cast<uint32_t>(work_rope_stage_off));  // T-2049 (N6): WorkScratch ROPE_STAGE
	}

	// ModelConstants (t3): kIExpLn2Q/kIExpBQ/kIExpCaQ, the i-exp derivation's
	// own compile-time constants (intmath.h) -- read directly from the
	// already-compiled C++ values rather than recomputed in HLSL, so no
	// floating-point truncation can drift between the two languages.
	std::vector<uint8_t> model_const_bytes(24, 0);
	PutI64At(model_const_bytes, 0, superslm::kIExpLn2Q);
	PutI64At(model_const_bytes, 8, superslm::kIExpBQ);
	PutI64At(model_const_bytes, 16, superslm::kIExpCaQ);

	// SiluLut (t4): the SIL1 canonical table (kSiluLutN+1 = 1025 int32 nodes),
	// the same compiled constant every CPU call site uses (forward_sites.cpp's
	// own MlpActSite call passes kSiluLutCanonicalTable directly).
	std::vector<uint8_t> silu_lut_bytes(sizeof(superslm::kSiluLutCanonicalTable));
	std::memcpy(silu_lut_bytes.data(), superslm::kSiluLutCanonicalTable, silu_lut_bytes.size());

	// RoPE's own real rotation data (t5/t6) -- the ROP1 "cos"/"sin" tensors'
	// raw bytes (rope_tables is model-wide, Sec5.6). A 1-byte dummy when absent (RopeInfo's own
	// presence flag is what a real site checks before ever reading these; sized nonzero only so
	// the buffer resource itself is legal to create).
	// T-2113 (B4, re-derived from Claude/Laplace/t2105-gpu-speed-ceiling-2026-08-14.md Sec2
	// change 1): `g_resident_rope` (above) -- these two tables are MODEL-WIDE CONSTANTS, so
	// re-packing and re-uploading them every call is pure, avoidable cost on every cache hit.
	// Packed (and later uploaded) only on a miss; `rope_fast_hit` reused at the upload site
	// below.
	const void* cos_src = cos_t != nullptr ? cos_t->data : nullptr;
	const void* sin_src = sin_t != nullptr ? sin_t->data : nullptr;
	const uint64_t cos_need = cos_t != nullptr ? static_cast<uint64_t>(cos_t->elem_count) * 8u : 8u;
	const uint64_t sin_need = sin_t != nullptr ? static_cast<uint64_t>(sin_t->elem_count) * 8u : 8u;
	const bool rope_fast_hit = !external_rope && g_resident_rope.valid &&
	                           g_resident_rope.cos_src == cos_src &&
	                           g_resident_rope.sin_src == sin_src &&
	                           g_resident_rope.cos_bytes == cos_need &&
	                           g_resident_rope.sin_bytes == sin_need;
	std::vector<uint8_t> cos_table_bytes, sin_table_bytes;
	if (!rope_fast_hit && !external_rope) {
		cos_table_bytes.assign(static_cast<size_t>(cos_need), 0);
		if (cos_t != nullptr) std::memcpy(cos_table_bytes.data(), cos_t->data, cos_table_bytes.size());
		sin_table_bytes.assign(static_cast<size_t>(sin_need), 0);
		if (sin_t != nullptr) std::memcpy(sin_table_bytes.data(), sin_t->data, sin_table_bytes.size());
	}

	// T-2101 (dispatch-overhead decomposition): `record_ms` covers this whole window, Reset()
	// through Close() -- every Upload()/MakeBuffer() call and every dispatch's own recording. (The
	// zero-reset for a REJECTED call lives at this function's own entry, above, alongside
	// `g_last_weight_upload_was_skipped`'s own -- not here, since a rejected call never reaches
	// this line at all.)
	SSLM_GPU_HR(dev.alloc->Reset());
	SSLM_GPU_HR(dev.list->Reset(dev.alloc.Get(), nullptr));

	// EVERY variable below this point that the command list's own GPU virtual addresses
	// reference is declared here, in the caller-visible out_state, or as a plain local that does
	// not need to survive past this function's own return -- see RunLayerLoopGpuSubmit's own
	// former comment on this exact hazard (a `ComPtr` released before ExecuteCommandLists/the
	// fence wait is a genuine GPU-memory use-after-free, reproduced once by execution, T-2049).
	// No try/catch here (see this function's own header comment) -- every allocation below can
	// throw `std::runtime_error` via `SSLM_GPU_HR`, and the caller's own enclosing try/catch is
	// what handles it, exactly as it always did when this code ran inline.
	Microsoft::WRL::ComPtr<ID3D12Resource> layout_buf;
	Microsoft::WRL::ComPtr<ID3D12Resource> rope_buf;
	Microsoft::WRL::ComPtr<ID3D12Resource> model_const_buf;
	Microsoft::WRL::ComPtr<ID3D12Resource> silu_lut_buf;
	Microsoft::WRL::ComPtr<ID3D12Resource> cos_table_buf;
	Microsoft::WRL::ComPtr<ID3D12Resource> sin_table_buf;
	Microsoft::WRL::ComPtr<ID3D12Resource> scratch_layout_buf;
	std::vector<Microsoft::WRL::ComPtr<ID3D12Resource>> upload_keep_alive;
	Microsoft::WRL::ComPtr<ID3D12Resource> seq_uav;
	Microsoft::WRL::ComPtr<ID3D12Resource> scratch_uav;
	Microsoft::WRL::ComPtr<ID3D12Resource> kv_uav;
	Microsoft::WRL::ComPtr<ID3D12Resource> work_scratch_uav;
	// T-2049 (N3, Claude/Poirot/34ef30f-gpu-serial-port-confirmation-review.md):
	// on a residency-cache miss, the packed row is copied into a genuine
	// DEFAULT-heap (VRAM-resident) buffer -- not cached as an UPLOAD-heap
	// resource, which the confirmation review correctly named as never
	// reaching §5.3's own "resident" property regardless of whether the
	// per-call copy into it was skipped (T-2045's own S3 cached the upload
	// heap itself, so every dispatch still read the weights across PCIe on
	// every call, hit or miss). A hit reuses the cached DEFAULT-heap `lw_buf`
	// directly -- no copy, no transition, no PCIe traffic at all.
	if (!weights_resident) {
		// T-2052 (M2, Claude/Poirot/36b9327-gpu-serial-port-reconfirmation-review.md):
		// release the PREVIOUS resident DEFAULT-heap buffer before allocating
		// its replacement, not after. The prior shape held both the old and
		// the new DEFAULT-heap copy simultaneously (the old one released only
		// at the `g_resident_weights.lw_buf = lw_buf;` assignment below,
		// AFTER the new one was already created) -- doubling peak weight VRAM
		// on exactly the content-change path this cache exists to make cheap
		// (a runtime adapter swap IS a weight-content change). Safe to
		// release here unconditionally: every prior `RunLayerLoopGpu` call
		// fence-waits before returning (below), so the old resource carries
		// no outstanding GPU reference by the time this line runs.
		g_resident_weights.lw_buf.Reset();
		g_resident_weights.valid = false;

		lw_upload_keep_alive = dev.Upload(lw_bytes.data(), lw_bytes.size());
		// T-2062 (Claude/Poirot/a3d44e7-gpu-serial-port-ship-confirmation-
		// review.md, S1): the ~1.31 GiB VRAM allocation below is the one most
		// likely to fail on this design's own target hardware (consumer
		// cards, §9's own VRAM-budget concern) -- `MakeBuffer` routes
		// `CreateCommittedResource` through `SSLM_GPU_HR`, which throws. No
		// longer caught by its own inner try (T-2052's own site-specific
		// catch, which T-2055's own comment above claimed to have removed
		// and had not -- the review's own S1 finding, "a sentence describing
		// what a correction removed, thirty-nine lines above the thing it
		// did not remove"): the outer try this block sits inside (opened
		// above, at this function's own recording-window boundary) now
		// covers it, so a throw here reaches the SAME `GetDeviceRemovedReason()`-
		// queried catch every other allocation in this window reaches, and
		// returns `GpuAllocationFailed`/`GpuDeviceRemoved` -- never
		// `KvPrecisionUnsupported`, the status this exact site returned under
		// the deleted inner catch, on the single largest allocation on this
		// leg, confirmed as a live divergence by the review's own executed
		// probe (call 7: `status=KvPrecisionUnsupported`, against call 5's
		// `status=GpuAllocationFailed` at `work_scratch_uav`, same outer
		// catch, same call shape).
		//
		// T-2080 (S1): the injection instrument's own weight-DEFAULT-heap
		// site -- restored here after T-2075 moved the instrument's ONLY arm
		// point to `work_scratch_uav` below, which made this allocation
		// permanently unreachable by any test (this review's own S1
		// finding). Site-parameterized now: arming THIS site (`gpu_port.h`'s
		// `kO11AllocInjectionSiteWeightDefaultHeap`) exercises T-2062's
		// own S1 remedy specifically (the throw here reaches the SAME outer
		// catch as every other allocation in the window); arming
		// `work_scratch_uav`'s own site below is unaffected.
		MaybeThrowInjectedO11AllocFault(kO11AllocInjectionSiteWeightDefaultHeap);
		Microsoft::WRL::ComPtr<ID3D12Resource> lw_default =
		    dev.MakeBuffer(lw_bytes.size(), D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_FLAG_NONE,
		                    D3D12_RESOURCE_STATE_COPY_DEST);
		dev.list->CopyResource(lw_default.Get(), lw_upload_keep_alive.Get());
		D3D12_RESOURCE_BARRIER lw_barrier{};
		lw_barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
		lw_barrier.Transition.pResource = lw_default.Get();
		lw_barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_DEST;
		lw_barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
		lw_barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
		dev.list->ResourceBarrier(1, &lw_barrier);
		lw_buf = lw_default;
		g_resident_weights.lw_buf = lw_buf;
		g_resident_weights.bytes = std::move(lw_bytes);
		g_resident_weights.valid = true;
		g_resident_weights.src_layers = static_cast<const void*>(layers);   // T-2100
		g_resident_weights.src_n = N;
		g_resident_weights.src_stride = layout.stride;
	}
	layout_buf = dev.Upload(layout_bytes.data(), layout_bytes.size());
	rope_buf = dev.Upload(rope_info_bytes.data(), rope_info_bytes.size());
	model_const_buf = dev.Upload(model_const_bytes.data(), model_const_bytes.size());
	silu_lut_buf = dev.Upload(silu_lut_bytes.data(), silu_lut_bytes.size());
	// T-2113 (B4): reuse the resident buffers on a hit -- `dev.Upload()` on a miss is a
	// synchronous Map/memcpy/Unmap (`d3d12_harness.h`), fully realized before this call
	// returns, so a cached resource is always valid content regardless of whether the command
	// list that references it ever executes (no invalidate-on-throw handling is needed, unlike
	// `g_resident_weights`/`g_resident_kv`'s own DEFAULT-heap CopyResource caches).
	if (external_rope) {
		// T-2113 (B5): the model handle's own resident RoPE buffers -- no pack, no
		// upload, no g_resident_rope read or write, ever, on this path. This is what
		// retires g_resident_rope for every caller that routes here (D-SLM3362).
		cos_table_buf = external_rope_cos_resident;
		sin_table_buf = external_rope_sin_resident;
	} else if (rope_fast_hit) {
		cos_table_buf = g_resident_rope.cos_buf;
		sin_table_buf = g_resident_rope.sin_buf;
	} else {
		cos_table_buf = dev.Upload(cos_table_bytes.data(), cos_table_bytes.size());
		sin_table_buf = dev.Upload(sin_table_bytes.data(), sin_table_bytes.size());
		g_resident_rope.cos_buf = cos_table_buf;
		g_resident_rope.sin_buf = sin_table_buf;
		g_resident_rope.cos_src = cos_src;
		g_resident_rope.sin_src = sin_src;
		g_resident_rope.cos_bytes = cos_need;
		g_resident_rope.sin_bytes = sin_need;
		g_resident_rope.valid = true;
	}
	scratch_layout_buf = dev.Upload(scratch_layout_bytes.data(), scratch_layout_bytes.size());
	seq_uav = MakeInitializedUav(dev, seq_bytes, upload_keep_alive);
	// T-2101: no host content -- see the header comment at `kv_fast_hit`'s own declaration above.
	scratch_uav = dev.MakeBuffer(scratch_layout.total, D3D12_HEAP_TYPE_DEFAULT,
	                              D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
	                              D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
	// T-2101 (D-SLM3301/D-SLM3294): device-resident K/V. A cache hit reuses the SAME DEFAULT-heap
	// buffer every call -- no pack, no upload -- and only needs a state transition back to
	// UNORDERED_ACCESS, since the previous call's own targeted readback (below) left it in
	// COPY_SOURCE. A miss (first call, or a different workspace) uploads the full buffer once, same
	// shape as `ResidentWeights`' own miss path, and the result is kept resident rather than
	// released at this function's own return.
	if (external_kv) {
		// T-2113 (B3, design Sec5.3/Sec10 B3): bind the caller's own dedicated buffer
		// directly -- no pack, no upload, ever, on this path (the handle's own
		// sslm_gpu_seq_create already uploaded its initial zeroed content once). The
		// buffer is created in UNORDERED_ACCESS state (gpu_1p0.cpp), so the FIRST call
		// on a fresh handle needs no resume barrier; every call after the first left it
		// in COPY_SOURCE (the targeted readback below, unconditional on this flag), so
		// this call's own resume barrier is gated on the caller-owned latch, mirroring
		// kv_fast_hit's own resume shape but scoped to one handle instead of one
		// process-global slot.
		kv_uav = external_kv_resident;
		const bool needs_resume =
		    io_external_kv_needs_resume_barrier != nullptr && *io_external_kv_needs_resume_barrier;
		if (needs_resume) {
			D3D12_RESOURCE_BARRIER kv_resume_barrier{};
			kv_resume_barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
			kv_resume_barrier.Transition.pResource = kv_uav.Get();
			kv_resume_barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_SOURCE;
			kv_resume_barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
			kv_resume_barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
			dev.list->ResourceBarrier(1, &kv_resume_barrier);
		}
	} else if (kv_fast_hit) {
		kv_uav = g_resident_kv.kv_buf;
		D3D12_RESOURCE_BARRIER kv_resume_barrier{};
		kv_resume_barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
		kv_resume_barrier.Transition.pResource = kv_uav.Get();
		kv_resume_barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_SOURCE;
		kv_resume_barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
		kv_resume_barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
		dev.list->ResourceBarrier(1, &kv_resume_barrier);
	} else {
		std::vector<uint8_t> kv_bytes(workspace, workspace + workspace_size);
		kv_uav = MakeInitializedUav(dev, kv_bytes, upload_keep_alive);
		g_resident_kv.kv_buf = kv_uav;
		g_resident_kv.valid = true;
		g_resident_kv.src_workspace = workspace;
		g_resident_kv.src_size = workspace_size;
	}
	// T-2039: WorkScratch is transient, per-dispatch scratch -- every byte is
	// written before it is read within the SAME dispatch (site_common.hlsli's
	// own per-primitive contract), so no host-side initial content is needed;
	// created directly in the UAV state, matching this file's own established
	// idiom for a device-only scratch buffer (RunDescriptorTableBind's out_uav).
	//
	// CORRECTED 2026-08-14 (T-2075, S1; D-SLM3228; site-parameterized by
	// T-2080, S1, D-SLM3241 -- this instrument now checks TWO named sites,
	// this one and the weight DEFAULT-heap allocation above, not one):
	// O11's own instrument (`MaybeThrowInjectedO11AllocFault`) also arms
	// HERE. This allocation runs UNCONDITIONALLY, on both the cache-hit and
	// cache-miss legs (it is outside the `if (!weights_resident)` block
	// entirely) -- so arming THIS site (`kO11AllocInjectionSiteWork
	// ScratchUav`) reaches the injected throw on a GENUINE cache hit,
	// without forcing any residency-decision change to make it reachable.
	// That is what makes `TestT2063_S1Mb_WorkScratchUavAllocationThrow_...`'s
	// own call 3 ("same content again, another cache hit") actually a hit
	// again: the catch's own two remedy lines (the observable write, the
	// cache invalidation) are the ONLY thing that can make that cell's M-b
	// assertions pass, because nothing upstream of this call has already
	// done their job.
	MaybeThrowInjectedO11AllocFault(kO11AllocInjectionSiteWorkScratchUav);
	work_scratch_uav = dev.MakeBuffer(work_total, D3D12_HEAP_TYPE_DEFAULT,
	                                        D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
	                                        D3D12_RESOURCE_STATE_UNORDERED_ACCESS);

	auto& attn_norm_pipe = harness::GetOrBuildComposedPipeline("attn_norm_site");

	dev.list->SetComputeRootSignature(attn_norm_pipe.root_sig.Get());  // identical signature, every PSO here
	// T-2113 (B4, design Sec10 B2/B4's own deferred root-binding hoist -- Claude/Brunel/
	// t2113-1p0-core-build-2026-08-15.md Sec3): the twelve root SRV/UAV bindings set ONCE
	// here, immediately after SetComputeRootSignature, instead of being re-issued inside
	// EVERY `bind_and_dispatch` call below. Root arguments persist on a command list across
	// Dispatch and SetPipelineState -- only SetComputeRootSignature resets them, and this
	// loop sets one signature for every PSO in the whole per-layer chain (the existing
	// comment above already states every PSO here shares it). The twelve buffer addresses
	// are call-constant by construction: every one of them is a local resolved before the
	// recording window opens and none is reassigned inside it. That removes
	// 12 * 24 * layers_to_record API calls per call from `record_ms`, re-derived from
	// Claude/Laplace/t2105-gpu-speed-ceiling-2026-08-14.md Sec2 change 2.
	dev.list->SetComputeRootShaderResourceView(1, lw_buf->GetGPUVirtualAddress());
	dev.list->SetComputeRootShaderResourceView(2, layout_buf->GetGPUVirtualAddress());
	dev.list->SetComputeRootShaderResourceView(3, rope_buf->GetGPUVirtualAddress());
	dev.list->SetComputeRootShaderResourceView(4, model_const_buf->GetGPUVirtualAddress());
	dev.list->SetComputeRootShaderResourceView(5, silu_lut_buf->GetGPUVirtualAddress());
	// T-2113 (B5, design Sec6.2/Sec1): this IS the real async resume rebind -- every SEPARATE
	// `sslm_decode_step_gpu` call reaches this same site fresh (its own `dev.list->Reset()`,
	// above), unconditionally, exactly matching design Sec6.2's own "rebind never conditioned
	// on whether this call followed a cut" rule. The two T-2106 violation-class pins retired
	// from B4's bench-only mid-call mechanism (see `bind_and_dispatch`'s own comment below) now
	// plant here, at the ONLY rebind site a real async resume ever crosses. Both default off (a
	// clean rebind, every existing caller); read via `std::getenv` once per process, the same
	// convention B4's own retired mechanism used.
	// No lambda here (deliberately, unlike B4's own now-retired mechanism): a lambda's
	// own `return` inside this function's body is indistinguishable, to a text-based
	// return-path census (tests/ci/check_gpu_guard_status_parity.py's own LWUWS
	// derivation), from a real control-flow exit of THIS function -- a plain boolean
	// expression, evaluated once via `static const`, carries no such statement at all.
	static const bool b5_async_drop_uav_rebind =
	    std::getenv("SSLM_B5_ASYNC_DROP_UAV_REBIND") != nullptr &&
	    std::getenv("SSLM_B5_ASYNC_DROP_UAV_REBIND")[0] == '1';
	static const bool b5_async_swap_srv_rebind =
	    std::getenv("SSLM_B5_ASYNC_SWAP_SRV_REBIND") != nullptr &&
	    std::getenv("SSLM_B5_ASYNC_SWAP_SRV_REBIND")[0] == '1';
	// The swapped-SRV-rebind violation pin (T-2106): cos/sin land in each other's slot.
	if (b5_async_swap_srv_rebind) {
		dev.list->SetComputeRootShaderResourceView(6, sin_table_buf->GetGPUVirtualAddress());
		dev.list->SetComputeRootShaderResourceView(7, cos_table_buf->GetGPUVirtualAddress());
	} else {
		dev.list->SetComputeRootShaderResourceView(6, cos_table_buf->GetGPUVirtualAddress());
		dev.list->SetComputeRootShaderResourceView(7, sin_table_buf->GetGPUVirtualAddress());
	}
	dev.list->SetComputeRootShaderResourceView(8, scratch_layout_buf->GetGPUVirtualAddress());
	dev.list->SetComputeRootUnorderedAccessView(9, seq_uav->GetGPUVirtualAddress());
	dev.list->SetComputeRootUnorderedAccessView(10, scratch_uav->GetGPUVirtualAddress());
	// The dropped-UAV-rebind violation pin (T-2106): kv_uav's own rebind is replaced with a bind
	// to a DIFFERENT, still-valid, still-live resource (work_scratch_uav) rather than left fully
	// unbound -- a raw root UAV descriptor left undefined after SetComputeRootSignature is an
	// out-of-spec GPU virtual address (real risk: a device fault/TDR on real hardware, not a
	// clean, repeatable divergence), so this pin realizes the class safely: the dispatch chain's
	// own subsequent K/V reads/writes land in transient scratch instead of the persistent K/V
	// store, a real and safely-producible wrong-answer (StandardsDocument.md Sec5.4's own
	// must-reject construction discipline).
	dev.list->SetComputeRootUnorderedAccessView(
	    11, (b5_async_drop_uav_rebind ? work_scratch_uav : kv_uav)->GetGPUVirtualAddress());
	dev.list->SetComputeRootUnorderedAccessView(12, work_scratch_uav->GetGPUVirtualAddress());
	// T-2113 (B6b, design Sec8): the adapter-delta dispatch's own two read-only inputs (t8
	// LoraAB, t9 Fold), bound in the SAME once-per-call hoist as every other root SRV/UAV
	// above -- every composed-pipeline root signature carries these two params now (B6b's own
	// d3d12_harness.h extension), so EVERY dispatch this call records, adapter-delta or not,
	// must have them set to a valid GPU VA before Dispatch (D3D12 root-signature validation).
	// No adapter bound this call (`adapter_bridge == nullptr`, every pre-B6b caller and every
	// base-only decode): bound to `model_const_buf`, already a valid resident SRV this call
	// binds anyway -- never read by any PSO that does not itself declare t8/t9, i.e. never
	// read at all when no adapter-delta dispatch is ever issued (see the per-layer loop
	// below), so this fallback binding is inert, not merely unread-in-practice.
	dev.list->SetComputeRootShaderResourceView(
	    13, (adapter_bridge ? adapter_bridge->lora_ab_resident : model_const_buf.Get())->GetGPUVirtualAddress());
	dev.list->SetComputeRootShaderResourceView(
	    14, (adapter_bridge ? adapter_bridge->fold_resident : model_const_buf.Get())->GetGPUVirtualAddress());

	out_state->H = H;
	out_state->HD = HD;
	out_state->NH = NH;
	out_state->NQH = NQH;
	out_state->I = I;
	out_state->N = N;
	out_state->context_cap_u32 = static_cast<uint32_t>(context_cap);
	out_state->scratch_layout = scratch_layout;
	out_state->work_wide_a_off = work_wide_a_off;
	out_state->work_wide_b_off = work_wide_b_off;
	out_state->work_adapter_u_off = work_adapter_u_off;
	out_state->seq_uav = seq_uav;
	out_state->kv_uav = kv_uav;
	out_state->scratch_uav = scratch_uav;
	out_state->work_scratch_uav = work_scratch_uav;
	out_state->layout_buf = layout_buf;
	out_state->rope_buf = rope_buf;
	out_state->model_const_buf = model_const_buf;
	out_state->silu_lut_buf = silu_lut_buf;
	out_state->scratch_layout_buf = scratch_layout_buf;
	out_state->lw_upload_keep_alive = lw_upload_keep_alive;
	out_state->upload_keep_alive = upload_keep_alive;
	return superslm::SslmForwardStatus::Ok;
}

// fence-wait and everything after it (moved to RunLayerLoopGpuFinish, below) PLUS the
// external-weights/external-rope bridge (this section's own new work, gated entirely
// behind the trailing parameters -- every existing behavior is reached identically
// when they are all null/false/0, which is what every pre-B5 call site still passes).
superslm::SslmForwardStatus RunLayerLoopGpuSubmit(
    superslm::SequenceLayerState& seq, const superslm::LayerWeights* layers,
    uint32_t num_hidden_layers, uint32_t layer_budget, size_t hidden_size, size_t head_dim,
    size_t num_key_value_heads, size_t intermediate_size, int64_t context_cap,
    const superslm::SslmTensorManifest& rope_tables, uint8_t* workspace, size_t workspace_size,
    ID3D12Resource* external_kv_resident, bool* io_external_kv_needs_resume_barrier,
    GpuLayerLoopInFlight** out_inflight, ID3D12Resource* external_weights_resident,
    ID3D12Resource* external_rope_cos_resident, ID3D12Resource* external_rope_sin_resident,
    bool external_rope_has, uint64_t external_rope_cos_elems, uint64_t external_rope_sin_elems,
    const GpuAdapterBridge* adapter_bridge) {
	if (out_inflight) *out_inflight = nullptr;
	// T-2169 (Rung 2b-prep, D-SLM3632/D-SLM3633): the guard ladder, the weight/rope/K-V
	// pack-and-residency decision, and the once-per-call root-signature/binding setup now live in
	// PrepareGpuLayerLoopChunkOpenState (above) -- a mechanical extraction, called here exactly
	// once per Submit call, matching this function's own pre-extraction behavior for its own
	// single-token callers. `invalidate_residency_caches_on_throw` and the readback-lifetime
	// resources stay declared here: they are used by the dispatch-recording/readback code below
	// and by this function's own catch clauses, neither of which moved.
	auto invalidate_residency_caches_on_throw = [&]() {
		g_resident_weights.lw_buf.Reset();
		g_resident_weights.valid = false;
		g_resident_kv.kv_buf.Reset();
		g_resident_kv.valid = false;
		g_last_weight_upload_was_skipped = false;  // ANCHOR:lwuws_write_catch
	};
	harness::Device& dev = harness::GetDevice();
	Microsoft::WRL::ComPtr<ID3D12Resource> seq_readback;
	Microsoft::WRL::ComPtr<ID3D12Resource> kv_readback;
	std::vector<size_t> kv_row_offsets;
	uint32_t dispatch_count_this_call = 0;
	GpuLayerLoopChunkOpenState state;
	const auto t_record_start = std::chrono::steady_clock::now();
	try {
	const superslm::SslmForwardStatus prep_status = PrepareGpuLayerLoopChunkOpenState(
	    seq, layers, num_hidden_layers, layer_budget, hidden_size, head_dim, num_key_value_heads,
	    intermediate_size, context_cap, rope_tables, workspace, workspace_size, external_kv_resident,
	    io_external_kv_needs_resume_barrier, external_weights_resident, external_rope_cos_resident,
	    external_rope_sin_resident, external_rope_has, external_rope_cos_elems,
	    external_rope_sin_elems, adapter_bridge, &state);
	if (prep_status != superslm::SslmForwardStatus::Ok) {
		return prep_status;  // a guard rejected before any recording began -- nothing to close
	}
#if defined(SUPERSLM_ENABLE_GPU_CHUNK_DISPATCH_INSTRUMENT)
	// T-2169 (Rung 2b): the SAME "one command list opened and submitted" event
	// SubmitChunkToFullDepthForG5Bridge's own identical increment counts (below) -- this
	// single-token call is a degenerate one-token "chunk" under the counter's own general
	// definition (tests/support/gpu_chunk_dispatch_instrument.h's own header comment: "one
	// increment per chunk ... submission"), and the T-2178 red suite's own reference arm
	// (N separate single-token calls through THIS function) is what proves the candidate
	// arm's one-chunk-one-submission claim by contrast -- N here against 1 there.
	++superslm_test::g_gpu_chunk_submit_count_probe;
#endif
	const uint32_t H = state.H;
	const uint32_t HD = state.HD;
	const uint32_t NH = state.NH;
	const uint32_t NQH = state.NQH;
	const uint32_t I = state.I;
	const uint32_t N = state.N;
	Microsoft::WRL::ComPtr<ID3D12Resource>& seq_uav = state.seq_uav;
	Microsoft::WRL::ComPtr<ID3D12Resource>& kv_uav = state.kv_uav;

	const uint32_t position_u32 = static_cast<uint32_t>(seq.context_length);  // constant across the whole call (Sec9.3)
	const uint32_t context_cap_u32 = static_cast<uint32_t>(context_cap);
	// Sec9.4: this token attends to every already-committed position plus its
	// own just-landed K/V -- constant across every layer of this token for the
	// identical reason `position` is (context_length only advances at the
	// token's own last layer's commit).
	const uint32_t width_u32 = static_cast<uint32_t>(seq.context_length) + 1u;

	// T-2101 (per-site decomposition, follow-up to §32/D-SLM3312): one timestamp boundary before
	// EVERY `bind_and_dispatch` call, capped at the heap's own real capacity (never overrun --
	// a call past the cap simply stops timing, it never re-uses a slot or corrupts an earlier one).
	uint32_t dispatch_query_index = 0;
	// T-2045 (C2): resume from seq.layer_index, exactly as the loop in
	// `RunLayerLoopImpl` (`forward_sites.cpp`) --
	// `while (advanced < layer_budget && seq.layer_index < num_hidden_layers)` --
	// does -- the guard above already proved
	// `seq.layer_index < N`, so `N - start_layer` cannot underflow.
	const uint32_t start_layer = seq.layer_index;
	const uint32_t layers_to_record = std::min(layer_budget, N - start_layer);
	// T-2169 (Rung 2, D-SLM3595): the single-token, non-chunked call shape -- one call into the
	// extracted per-token dispatch body (above), for this call's own [start_layer,
	// start_layer + layers_to_record) slice, at the constant position_u32/width_u32 this whole
	// call already computed. This is byte-for-byte the same dispatch sequence the pre-extraction
	// inline loop issued for the identical layer range (Rung 2's own refactor-safety cell proves
	// it) -- only the code that issues it moved into its own function.
	RecordOneTokenFullDepthDispatchBody(dev, start_layer, layers_to_record, H, HD, NH, NQH, I, N,
	                                     context_cap_u32, position_u32, width_u32, state.scratch_layout,
	                                     state.work_wide_a_off, state.work_wide_b_off,
	                                     state.work_adapter_u_off, adapter_bridge, dispatch_query_index);
#if defined(SUPERSLM_ENABLE_GPU_CHUNK_DISPATCH_INSTRUMENT)
	// T-2169 (Rung 2b): the SAME per-token-dispatch-body-invocation event
	// SubmitChunkToFullDepthForG5Bridge's own identical increment counts (below), for the
	// identical reference-arm-comparison reason the submit counter above states.
	++superslm_test::g_gpu_chunk_dispatch_count_probe;
#endif
	// One final boundary, immediately after the LAST dispatch (the final layer's own commit_site
	// above) -- `dispatch_query_index` now equals the total dispatch count this call issued, so
	// this is boundary N for N dispatches, giving N per-dispatch deltas below. Resolved into a
	// small readback buffer now, alongside the real readback work below, so the resolve itself
	// costs one more small copy rather than a whole extra submit.
	dispatch_count_this_call = std::min(dispatch_query_index, harness::Device::kMaxTimestampSlots - 1);
	dev.list->EndQuery(dev.timestamp_heap.Get(), D3D12_QUERY_TYPE_TIMESTAMP, dispatch_count_this_call);
	dev.list->ResolveQueryData(dev.timestamp_heap.Get(), D3D12_QUERY_TYPE_TIMESTAMP, 0,
	                            dispatch_count_this_call + 1, dev.timestamp_readback.Get(), 0);

	// --- Readback: SeqState (whole, small -- O(hidden_size)) + the K/V workspace's OWN newly
	// written rows into `seq`/`workspace`. T-2101: not the whole 448 MiB workspace. Every dispatch
	// this call issued wrote at most one new K row and one new V row per (layer, kv_head) it
	// processed, at `position_u32` (`kv_proj_site.hlsl`'s own write; `rope_guard_site.hlsl`'s own
	// in-place K rotation touches the SAME row, never a different one) -- the identical
	// `KvHalfOffsetGpu`/`KvRowOffsetWithinHalfGpu` addressing `KeyRowGpu`/`ValueRowGpu` below use
	// host-side, so every offset computed here is exact, not an approximation of what changed. ---
	seq_readback = dev.MakeBuffer(SeqTotalSize(H), D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_FLAG_NONE,
	                                    D3D12_RESOURCE_STATE_COPY_DEST);
	kv_row_offsets.reserve(static_cast<size_t>(layers_to_record) * NH * 2u);
	for (uint32_t i = 0; i < layers_to_record; ++i) {
		const uint32_t l = start_layer + i;
		const size_t half_off = static_cast<size_t>(l) * static_cast<size_t>(context_cap) *
		                         static_cast<size_t>(NH) * static_cast<size_t>(HD) * 2u;
		const size_t k_store_size =
		    static_cast<size_t>(context_cap) * static_cast<size_t>(NH) * static_cast<size_t>(HD);
		for (uint32_t h = 0; h < NH; ++h) {
			const size_t row_off = static_cast<size_t>(h) * static_cast<size_t>(context_cap) *
			                            static_cast<size_t>(HD) +
			                        static_cast<size_t>(position_u32) * static_cast<size_t>(HD);
			kv_row_offsets.push_back(half_off + row_off);                 // K row
			kv_row_offsets.push_back(half_off + k_store_size + row_off);  // V row
		}
	}
	const size_t kv_readback_bytes = kv_row_offsets.size() * static_cast<size_t>(HD);
	kv_readback = dev.MakeBuffer(kv_readback_bytes, D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_FLAG_NONE,
	                                   D3D12_RESOURCE_STATE_COPY_DEST);
	D3D12_RESOURCE_BARRIER pre_copy[2]{};
	pre_copy[0].Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
	pre_copy[0].Transition.pResource = seq_uav.Get();
	pre_copy[0].Transition.StateBefore = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
	pre_copy[0].Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_SOURCE;
	pre_copy[0].Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
	pre_copy[1] = pre_copy[0];
	pre_copy[1].Transition.pResource = kv_uav.Get();
	dev.list->ResourceBarrier(2, pre_copy);
	// T-2113 (B3): kv_uav (whichever buffer it names) is now COPY_SOURCE, same as the
	// pre-existing g_resident_kv path already left it for kv_fast_hit's own resume
	// barrier to find on the NEXT call -- latch that fact into the caller-owned flag on
	// the external-buffer path too, so THIS handle's own next call knows to resume it.
	if (external_kv_resident != nullptr && io_external_kv_needs_resume_barrier != nullptr) {
		*io_external_kv_needs_resume_barrier = true;
	}
	dev.list->CopyResource(seq_readback.Get(), seq_uav.Get());
	for (size_t r = 0; r < kv_row_offsets.size(); ++r) {
		dev.list->CopyBufferRegion(kv_readback.Get(), r * static_cast<UINT64>(HD), kv_uav.Get(),
		                            kv_row_offsets[r], HD);
	}
	} catch (const GpuGemmGroupArithmeticError& e) {
		// T-2101 (S4, code review 6d9e04e-t2101-gpu-throughput-review.md, confirmation pass @
		// f7026db): a permanent coding-arithmetic bug, never a transient or size-dependent one --
		// caught by its OWN clause, ahead of the generic `catch (const std::runtime_error&)` below,
		// so it never inherits that clause's own `GpuAllocationFailed` status, whose documented
		// meaning ("retry smaller") is actively wrong advice here: no retry at any size fixes a
		// group count that does not cover its own out_channels. The message is preserved (stderr,
		// matching `SSLM_GPU_HR`'s own diagnostic convention) rather than discarded -- the
		// confirmation pass's own finding was that the original guard's exception, its one useful
		// payload, was constructed, thrown, and dropped by an unnamed catch clause.
		std::fprintf(stderr, "superslm_gpu: %s\n", e.what());
		invalidate_residency_caches_on_throw();
		dev.list->Close();
		return superslm::SslmForwardStatus::GpuGemmGroupArithmeticInvalid;
	} catch (const std::runtime_error&) {
		// T-2055 (Claude/Poirot/db73b22-gpu-serial-port-final-confirmation-
		// review.md, P3): defensively invalidate the weight-residency cache
		// regardless of WHERE in the window the throw happened. A throw
		// reached AFTER `g_resident_weights` was already marked valid+resident
		// (the `!weights_resident` branch above, once ITS OWN CopyResource is
		// merely RECORDED) but BEFORE this command list actually executes
		// would otherwise leave that assignment describing a DEFAULT-heap
		// buffer whose content copy was never submitted to the GPU -- serving
		// it back to the NEXT call as a cache hit would bind an uninitialized
		// VRAM buffer to every site's own weight reads with no error at all,
		// the silent-wrong-answer class this whole arc exists to close, not
		// merely the crash class M1 already closed. Safe unconditionally: if
		// the cache was never touched this call (a hit, or a throw before the
		// `!weights_resident` branch ran), this is a same-state no-op.
		//
		// CORRECTED 2026-08-14 (T-2062, Claude/Poirot/
		// a3d44e7-gpu-serial-port-ship-confirmation-review.md, M-b): "this is
		// a same-state no-op" is false on the CACHE-HIT path specifically --
		// left as written above, not rewritten, per this tree's own
		// append-only discipline. On a hit, `g_resident_weights.valid` was
		// already `true` and `lw_buf` already held the resident DEFAULT-heap
		// copy BEFORE this call ever ran (a prior call's own success path set
		// it); the two lines below unconditionally reset it to invalid --
		// forcing the NEXT call into a full re-upload of the packed row
		// (~1.31 GiB across PCIe at the 1.5B tier) that a hit exists to
		// avoid, which is a real state CHANGE, not a no-op, on that one path.
		// The conservative DIRECTION the paragraph above argues for is still
		// correct and unchanged -- reproduced by execution (the review's own
		// probe: a cache-hit call that throws, followed by a well-formed call
		// with identical content, reads back a forced miss) -- what was wrong
		// is calling a real, deliberate cache-state change a no-op.
		//
		// T-2057 (Claude/Vitruvius/t1986-gpu-serial-port-design-2026-08-13.md
		// §22.3, D-SLM3191): queried HERE, immediately before the cleanup
		// below, not after -- `dev.dev` (ComPtr<ID3D12Device>) is in scope
		// through this whole function via the `harness::Device& dev` bound
		// above, and `GetDeviceRemovedReason()` is always safely callable on
		// it regardless of removal state (a standard D3D12 idiom). Zero
		// changes to `SSLM_GPU_HR` or the caught exception type -- the ruling
		// resolves T-2055's own P4 (`KvPrecisionUnsupported` already carrying
		// two unrelated permanent-hardware meanings on this leg) without
		// needing either.
		const HRESULT device_removed_reason = dev.dev->GetDeviceRemovedReason();
		// T-2101 (S4, code review 6d9e04e-t2101-gpu-throughput-review.md, confirmation pass @
		// f7026db): both residency caches reset to invalid, `LastWeightUploadWasSkipped` set false --
		// issued via the shared `invalidate_residency_caches_on_throw` lambda (declared immediately
		// before the try block, where its own full history -- T-2055, T-2062 M-b, T-2080 S3 -- now
		// lives), identical effect, written once so this clause and the `GpuGemmGroupArithmeticError`
		// clause above cannot diverge.
		invalidate_residency_caches_on_throw();
		dev.list->Close();
		// T-2057 (D-SLM3191): `GpuDeviceRemoved` iff the device is confirmed
		// gone right now (device recreation owed, not a retry against this
		// handle); `GpuAllocationFailed` otherwise (device alive, this one
		// call failed -- transient/size-dependent, retry smaller). Named
		// residual, not fixed here (the ruling's own §22.3): this answers "is
		// the device gone right now," not "did removal cause THIS throw" -- an
		// allocation failure racing an unrelated async device removal reads
		// as `GpuDeviceRemoved`, the conservative direction.
		return device_removed_reason != S_OK ? superslm::SslmForwardStatus::GpuDeviceRemoved
		                                      : superslm::SslmForwardStatus::GpuAllocationFailed;
	}
	SSLM_GPU_HR(dev.list->Close());
	const auto t_record_end = std::chrono::steady_clock::now();
	g_last_call_timing.record_ms =
	    std::chrono::duration<double, std::milli>(t_record_end - t_record_start).count();

	// T-2101: `submit_wait_ms` is CPU time from submission to the fence signaling complete --
	// submission overhead PLUS actual GPU execution PLUS driver/OS scheduling, NOT a GPU-only
	// number. `gpu_busy_ms` (below, from the timestamp queries §31's own bracket placed around this
	// call's first/last dispatch) is the GPU-only figure the roofline comparison is judged against.
	ID3D12CommandList* lists[] = {dev.list.Get()};
	dev.queue->ExecuteCommandLists(1, lists);
	SSLM_GPU_HR(dev.queue->Signal(dev.fence.Get(), ++dev.fence_val));
	// T-2113 (B5, design Sec6.2): NO fence-wait here -- "records ... submits ... and
	// returns without waiting for the fence." Everything below used to run
	// synchronously at this point (the wait, the GPU-timing readback, the SeqState/KV
	// readback, the sticky-tag decode); it now lives in RunLayerLoopGpuFinish, driven
	// by the in-flight token this call hands back. `record_ms` is still measured here
	// (it is host-side recording time, already complete by this point); `submit_wait_ms`/
	// `gpu_busy_ms`/`readback_ms` are now measured inside Finish, against the SAME
	// `g_last_call_timing` global every existing caller of the synchronous
	// `RunLayerLoopGpu` wrapper still reads after ITS OWN Submit+Finish pair completes.
	std::unique_ptr<GpuLayerLoopInFlight> inflight(new GpuLayerLoopInFlight());
	inflight->dev = &dev;
	inflight->fence_val = dev.fence_val;
	inflight->dispatch_count_this_call = dispatch_count_this_call;
	inflight->seq_readback = seq_readback;
	inflight->kv_readback = kv_readback;
	inflight->kv_row_offsets = kv_row_offsets;
	inflight->seq_bytes_size = SeqTotalSize(state.H);
	inflight->hidden_size_h = state.H;
	inflight->head_dim_hd = state.HD;
	inflight->t_record_end = t_record_end;
	// T-2113 (B5): extend every per-call GPU resource's lifetime to the fence, per
	// GpuLayerLoopInFlight's own struct comment -- these ComPtr copies are what close
	// the use-after-free the Submit/Finish split would otherwise open. T-2169 (Rung 2b-prep):
	// sourced from `state` now -- these are the same resources PrepareGpuLayerLoopChunkOpenState
	// populated, unchanged, since it is still called exactly once per Submit call here.
	inflight->layout_buf = state.layout_buf;
	inflight->rope_buf = state.rope_buf;
	inflight->model_const_buf = state.model_const_buf;
	inflight->silu_lut_buf = state.silu_lut_buf;
	inflight->scratch_layout_buf = state.scratch_layout_buf;
	inflight->seq_uav = state.seq_uav;
	inflight->scratch_uav = state.scratch_uav;
	inflight->work_scratch_uav = state.work_scratch_uav;
	inflight->lw_upload_keep_alive = state.lw_upload_keep_alive;
	inflight->upload_keep_alive = state.upload_keep_alive;
	if (out_inflight) *out_inflight = inflight.release();
	return superslm::SslmForwardStatus::Ok;
}

// T-2169 (Rung 2b, design Sec5/Sec5.1/Sec8, D-SLM3595/D-SLM3611/D-SLM3612/D-SLM3631/D-SLM3634):
// the internal SUB-chunk-submission primitive -- records ONE (sub-)chunk's full dispatch
// sequence into ONE open command list, bounded to at most `kT2169TdrSafeMaxChunkTokens` tokens
// (D-SLM3596, and the empirical stability finding below that ruling now also carries -- see
// `SubmitChunkToFullDepthForG5Bridge`'s own header comment, immediately below this function, for
// the full account and D-SLM3649). One call into `PrepareGpuLayerLoopChunkOpenState` (chunk-open,
// exactly once, per D-SLM3632/D-SLM3633), then one call into `RecordOneTokenFullDepthDispatchBody`
// per admitted token in this sub-chunk, full depth, in order (token-outer, layer-inner,
// unreordered, per §6's own "not reordered the way the CPU fix inverted its own loop nest"
// ruling) -- and submits/fences once for the whole sub-chunk. Internal linkage surface (no ABI
// touch, §7). Callers needing an admitted chunk LARGER than the bound go through
// `SubmitChunkToFullDepthForG5Bridge` (the real chunk-scoped entry point, below), never this
// function directly, except for the sub-chunk-sized final/only slice that function itself drives.
//
// `chunk_embedding_bytes` is a caller-owned, already-packed, contiguous buffer of `chunk_len`
// per-token embedding blocks, each exactly `SeqScaleOff(hidden_size) + 16` bytes (the `[0,
// SeqScaleOff(H)+16)` range 2b names) -- computed by the CALLER (Rung 3/4's own pre-scan, which
// has access to the model handle's `EmbedEntry`/`sslm_gpu_seq_embed_token` arithmetic this
// function does not) for every candidate index the pre-scan validates, in the same pass that
// produces `embed_admit_count` (§5's own "the pre-scan and the record-time embedding source are
// the same pass, not two"). This primitive itself performs no embedding arithmetic and no
// admission decision -- it drives exactly `chunk_len` already-admitted tokens to full depth.
superslm::SslmForwardStatus SubmitOneSubChunkToFullDepthForG5Bridge(
    superslm::SequenceLayerState& seq, const superslm::LayerWeights* layers,
    uint32_t num_hidden_layers, size_t hidden_size, size_t head_dim, size_t num_key_value_heads,
    size_t intermediate_size, int64_t context_cap, const superslm::SslmTensorManifest& rope_tables,
    uint8_t* workspace, size_t workspace_size, const uint8_t* chunk_embedding_bytes,
    uint32_t chunk_len, ID3D12Resource* external_kv_resident,
    bool* io_external_kv_needs_resume_barrier, ID3D12Resource* external_weights_resident,
    ID3D12Resource* external_rope_cos_resident, ID3D12Resource* external_rope_sin_resident,
    bool external_rope_has, uint64_t external_rope_cos_elems, uint64_t external_rope_sin_elems,
    const GpuAdapterBridge* adapter_bridge, GpuLayerLoopInFlight** out_inflight) {
	if (out_inflight) *out_inflight = nullptr;
	// Same shared cache-invalidation lambda as RunLayerLoopGpuSubmit's own (superslm_gpu.cpp,
	// this file) -- declared fresh here rather than factored out, matching the codebase's own
	// established convention (T-2101 consolidated it ONCE already, into this lambda's own
	// current five-line shape, shared by both of Submit's catch clauses; this primitive is a
	// third, independent call site of the identical five lines, not a fourth divergent copy).
	auto invalidate_residency_caches_on_throw = [&]() {
		g_resident_weights.lw_buf.Reset();
		g_resident_weights.valid = false;
		g_resident_kv.kv_buf.Reset();
		g_resident_kv.valid = false;
		g_last_weight_upload_was_skipped = false;
	};
	harness::Device& dev = harness::GetDevice();
	Microsoft::WRL::ComPtr<ID3D12Resource> seq_readback;
	Microsoft::WRL::ComPtr<ID3D12Resource> kv_readback;
	// D-SLM3611 (2b): the per-chunk upload-heap buffer holding every admitted token's own
	// embedding bytes -- declared HERE, outside the try, for the identical use-after-free
	// reason every other per-call GPU-referenced resource in this file is (superslm_gpu.cpp's
	// own comment on `layout_buf` et al., above): a `ComPtr` released before
	// ExecuteCommandLists/the fence wait is a genuine GPU-memory use-after-free, since the
	// per-token `CopyBufferRegion` calls below read from it.
	Microsoft::WRL::ComPtr<ID3D12Resource> chunk_embed_upload;
	std::vector<size_t> kv_row_offsets;
	uint32_t dispatch_count_this_call = 0;
	GpuLayerLoopChunkOpenState state;
	const auto t_record_start = std::chrono::steady_clock::now();
	// D-SLM3612 (2a): the chunk-open host mirror, read exactly ONCE, before any admitted
	// token's own dispatches are recorded -- the same boundary point PrepareGpuLayerLoop
	// ChunkOpenState's own fresh_sequence/residency-cache decisions already trust as
	// known-current (§5's own "the identical boundary point already established as
	// known-current by that ruling").
	const int64_t chunk_open_ctxlen = seq.context_length;
	try {
	const superslm::SslmForwardStatus prep_status = PrepareGpuLayerLoopChunkOpenState(
	    seq, layers, num_hidden_layers, /*layer_budget=*/num_hidden_layers, hidden_size, head_dim,
	    num_key_value_heads, intermediate_size, context_cap, rope_tables, workspace, workspace_size,
	    external_kv_resident, io_external_kv_needs_resume_barrier, external_weights_resident,
	    external_rope_cos_resident, external_rope_sin_resident, external_rope_has,
	    external_rope_cos_elems, external_rope_sin_elems, adapter_bridge, &state);
	if (prep_status != superslm::SslmForwardStatus::Ok) {
		return prep_status;  // a guard rejected before any recording began -- nothing to close
	}
#if defined(SUPERSLM_ENABLE_GPU_CHUNK_DISPATCH_INSTRUMENT)
	// Design Sec9's own Guard-vitality row: one increment per chunk (or TDR-safe sub-chunk)
	// submission -- this call IS one such submission (the whole-chunk case; the TDR-safe
	// sub-chunk split, once measured, calls this primitive once per sub-chunk, so the
	// increment stays correct under either resolution without change).
	++superslm_test::g_gpu_chunk_submit_count_probe;
#endif
	const uint32_t H = state.H;
	const uint32_t HD = state.HD;
	const uint32_t NH = state.NH;
	const uint32_t NQH = state.NQH;
	const uint32_t I = state.I;
	const uint32_t N = state.N;
	const uint32_t context_cap_u32 = state.context_cap_u32;
	Microsoft::WRL::ComPtr<ID3D12Resource>& seq_uav = state.seq_uav;
	Microsoft::WRL::ComPtr<ID3D12Resource>& kv_uav = state.kv_uav;

	// D-SLM3611 (2b): the embedding delivery range -- [0, SeqScaleOff(H) + 16), the SeqState
	// prefix `hidden_codes[H]` (SeqScaleOff(H) bytes) plus `hidden_scale.m`/`.e` (16 bytes),
	// the exact contiguous byte range §5 names. One upload of the WHOLE chunk's own embedding
	// bytes now (host-computed by the caller, ahead of recording); sliced per-token below.
	const size_t embed_block_bytes = static_cast<size_t>(SeqScaleOff(H)) + 16u;
	if (chunk_len > 0) {
		chunk_embed_upload = dev.Upload(chunk_embedding_bytes, embed_block_bytes * static_cast<size_t>(chunk_len));
		// Kept alive to the fence via the SAME mechanism `state.upload_keep_alive` already
		// extends to `inflight` for every other upload this call makes (below).
		state.upload_keep_alive.push_back(chunk_embed_upload);
	}

	uint32_t dispatch_query_index = 0;
	kv_row_offsets.reserve(static_cast<size_t>(chunk_len) * static_cast<size_t>(N) *
	                        static_cast<size_t>(NH) * 2u);
	for (uint32_t t = 0; t < chunk_len; ++t) {
		// D-SLM3612 (2a): position/width for token t, from the chunk-local counter -- never a
		// stale call-wide constant read from `seq.context_length` (correct for exactly one
		// token; stale for tokens 2..N of a chunk, T-2175's own fracture).
		const uint32_t position_u32 = static_cast<uint32_t>(chunk_open_ctxlen) + t;
		const uint32_t width_u32 = position_u32 + 1u;

		// D-SLM3611 (2b): per-token embedding delivery -- uniform for every admitted token,
		// including token 0, no first-token special case. The SeqState UAV transitions to
		// COPY_DEST, receives exactly this token's own byte range from the shared upload
		// buffer at its own offset, transitions back to UNORDERED_ACCESS before this token's
		// own dispatch chain reads it.
		D3D12_RESOURCE_BARRIER to_copy_dest{};
		to_copy_dest.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
		to_copy_dest.Transition.pResource = seq_uav.Get();
		to_copy_dest.Transition.StateBefore = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
		to_copy_dest.Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_DEST;
		to_copy_dest.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
		dev.list->ResourceBarrier(1, &to_copy_dest);
		dev.list->CopyBufferRegion(seq_uav.Get(), 0, chunk_embed_upload.Get(),
		                            static_cast<UINT64>(t) * embed_block_bytes,
		                            embed_block_bytes);
		D3D12_RESOURCE_BARRIER to_uav = to_copy_dest;
		to_uav.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_DEST;
		to_uav.Transition.StateAfter = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
		dev.list->ResourceBarrier(1, &to_uav);

		// D-SLM3595: full-depth dispatch chain for this token -- the chunk primitive drives
		// every admitted token to its complete layer count in one call, since chunking now
		// happens at the command-list boundary rather than at the per-token layer-budget
		// boundary (§5's own "the chunk primitive's own token-body calls always request full
		// depth").
		RecordOneTokenFullDepthDispatchBody(dev, /*start_layer=*/0, /*layers_to_record=*/N, H, HD,
		                                     NH, NQH, I, N, context_cap_u32, position_u32,
		                                     width_u32, state.scratch_layout,
		                                     state.work_wide_a_off, state.work_wide_b_off,
		                                     state.work_adapter_u_off, adapter_bridge,
		                                     dispatch_query_index);
		// T-2180/T-2183 (D-SLM3655/D-SLM3660): the tenth-failure-origin seam -- immediately after
		// this token's own dispatches are recorded and before the loop advances (gpu_port.h's own
		// header comment on ArmT2169ChunkRecordingFaultInjection carries the full account). Zero
		// overhead unarmed, never `#ifdef`-guarded at the call site, matching
		// MaybeThrowInjectedO11AllocFault's own established convention exactly.
		MaybeThrowInjectedT2169ChunkRecordingFault(t);
#if defined(SUPERSLM_ENABLE_GPU_CHUNK_DISPATCH_INSTRUMENT)
		// Design Sec9's own Guard-vitality row / tests/support/gpu_chunk_dispatch_instrument.h's
		// own committed contract: the counter's delta must equal `admit_count * num_hidden_layers`
		// (Rung 3/4's own trust/guard cells, cell_trust_and_guard.cpp, assert exactly this) -- one
		// token's own full-depth dispatch body (`RecordOneTokenFullDepthDispatchBody`, called once
		// above) issues `N` (num_hidden_layers) layers' worth of dispatches in that single call, so
		// the counter advances by `N` here, not by one -- a plain `++` would under-count by a
		// factor of `N` against the header's own documented "admit_count * num_hidden_layers"
		// formula, caught by executing Rung 3/4's own guard-vitality cells against the real
		// artifact (T-2180).
		superslm_test::g_gpu_chunk_dispatch_count_probe += N;
#endif

		// D-SLM3631: the nested per-token/per-layer/per-head K/V readback-offset enumeration
		// -- generalizes the single-token path's own single-position loop (superslm_gpu.cpp,
		// RunLayerLoopGpuSubmit's own readback section) to this token's own record-time
		// position, growing `kv_row_offsets` to `admit_count * num_hidden_layers * NH * 2`
		// entries across the whole chunk.
		for (uint32_t l = 0; l < N; ++l) {
			const size_t half_off = static_cast<size_t>(l) * static_cast<size_t>(context_cap) *
			                         static_cast<size_t>(NH) * static_cast<size_t>(HD) * 2u;
			const size_t k_store_size =
			    static_cast<size_t>(context_cap) * static_cast<size_t>(NH) * static_cast<size_t>(HD);
			for (uint32_t h = 0; h < NH; ++h) {
				const size_t row_off = static_cast<size_t>(h) * static_cast<size_t>(context_cap) *
				                            static_cast<size_t>(HD) +
				                        static_cast<size_t>(position_u32) * static_cast<size_t>(HD);
				kv_row_offsets.push_back(half_off + row_off);                 // K row
				kv_row_offsets.push_back(half_off + k_store_size + row_off);  // V row
			}
		}
	}
	dispatch_count_this_call = std::min(dispatch_query_index, harness::Device::kMaxTimestampSlots - 1);
	dev.list->EndQuery(dev.timestamp_heap.Get(), D3D12_QUERY_TYPE_TIMESTAMP, dispatch_count_this_call);
	dev.list->ResolveQueryData(dev.timestamp_heap.Get(), D3D12_QUERY_TYPE_TIMESTAMP, 0,
	                            dispatch_count_this_call + 1, dev.timestamp_readback.Get(), 0);

	// --- Readback: SeqState (whole, small) + every K/V row every admitted token's own
	// dispatches wrote across the WHOLE chunk (D-SLM3631's own generalized enumeration,
	// above) -- the single-pass-per-(sub-)chunk boundary D-SLM3631 rules (§5.1), never once
	// per token. ---
	seq_readback = dev.MakeBuffer(SeqTotalSize(H), D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_FLAG_NONE,
	                               D3D12_RESOURCE_STATE_COPY_DEST);
	const size_t kv_readback_bytes = kv_row_offsets.size() * static_cast<size_t>(HD);
	kv_readback = dev.MakeBuffer(kv_readback_bytes, D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_FLAG_NONE,
	                              D3D12_RESOURCE_STATE_COPY_DEST);
	D3D12_RESOURCE_BARRIER pre_copy[2]{};
	pre_copy[0].Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
	pre_copy[0].Transition.pResource = seq_uav.Get();
	pre_copy[0].Transition.StateBefore = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
	pre_copy[0].Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_SOURCE;
	pre_copy[0].Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
	pre_copy[1] = pre_copy[0];
	pre_copy[1].Transition.pResource = kv_uav.Get();
	dev.list->ResourceBarrier(2, pre_copy);
	if (external_kv_resident != nullptr && io_external_kv_needs_resume_barrier != nullptr) {
		*io_external_kv_needs_resume_barrier = true;
	}
	dev.list->CopyResource(seq_readback.Get(), seq_uav.Get());
	for (size_t r = 0; r < kv_row_offsets.size(); ++r) {
		dev.list->CopyBufferRegion(kv_readback.Get(), r * static_cast<UINT64>(HD), kv_uav.Get(),
		                            kv_row_offsets[r], HD);
	}
	} catch (const GpuGemmGroupArithmeticError& e) {
		// D-SLM3634 (§5.1): a mid-recording infrastructural exception is CHUNK-scoped, not
		// token-scoped -- the tenth failure origin. Every admitted token's own
		// already-recorded-but-unexecuted dispatches, however many precede the one that threw,
		// are discarded as one unit by this same unexecuted `dev.list->Close()`, since they
		// were all recorded into the SAME list this catch clause closes without ever
		// submitting it. A documented, accepted divergence from the shipped per-token path's
		// own token-scoped atomicity (§5.1's own ruling), not a correctness defect --
		// submission-granularity batching structurally trades per-token exception atomicity
		// for round-trip reduction.
		std::fprintf(stderr, "superslm_gpu: %s\n", e.what());
		invalidate_residency_caches_on_throw();
		dev.list->Close();
		return superslm::SslmForwardStatus::GpuGemmGroupArithmeticInvalid;
	} catch (const std::runtime_error&) {
		// D-SLM3634: the identical chunk-scoped discard, for the generic allocation/device-
		// removed failure class RunLayerLoopGpuSubmit's own twin catch clause already handles
		// (superslm_gpu.cpp, above) -- same status mapping, same cache-invalidation contract.
		const HRESULT device_removed_reason = dev.dev->GetDeviceRemovedReason();
		invalidate_residency_caches_on_throw();
		dev.list->Close();
		return device_removed_reason != S_OK ? superslm::SslmForwardStatus::GpuDeviceRemoved
		                                      : superslm::SslmForwardStatus::GpuAllocationFailed;
	}
	SSLM_GPU_HR(dev.list->Close());
	const auto t_record_end = std::chrono::steady_clock::now();
	g_last_call_timing.record_ms =
	    std::chrono::duration<double, std::milli>(t_record_end - t_record_start).count();
	ID3D12CommandList* lists[] = {dev.list.Get()};
	dev.queue->ExecuteCommandLists(1, lists);
	SSLM_GPU_HR(dev.queue->Signal(dev.fence.Get(), ++dev.fence_val));
	std::unique_ptr<GpuLayerLoopInFlight> inflight(new GpuLayerLoopInFlight());
	inflight->dev = &dev;
	inflight->fence_val = dev.fence_val;
	inflight->dispatch_count_this_call = dispatch_count_this_call;
	inflight->seq_readback = seq_readback;
	inflight->kv_readback = kv_readback;
	inflight->kv_row_offsets = kv_row_offsets;
	inflight->seq_bytes_size = SeqTotalSize(state.H);
	inflight->hidden_size_h = state.H;
	inflight->head_dim_hd = state.HD;
	inflight->t_record_end = t_record_end;
	inflight->layout_buf = state.layout_buf;
	inflight->rope_buf = state.rope_buf;
	inflight->model_const_buf = state.model_const_buf;
	inflight->silu_lut_buf = state.silu_lut_buf;
	inflight->scratch_layout_buf = state.scratch_layout_buf;
	inflight->seq_uav = state.seq_uav;
	inflight->scratch_uav = state.scratch_uav;
	inflight->work_scratch_uav = state.work_scratch_uav;
	inflight->lw_upload_keep_alive = state.lw_upload_keep_alive;
	inflight->upload_keep_alive = state.upload_keep_alive;
	if (out_inflight) *out_inflight = inflight.release();
	return superslm::SslmForwardStatus::Ok;
}

// T-2169 (Rung 2, design Sec5, D-SLM3596; the bound's own real derivation, D-SLM3649 -- see this
// constant's own definition, below, for the full account): the TDR-safe (now: hardware-stability-
// safe) maximum sub-chunk size, in tokens. `extern`, matching this project's own established
// red-suite convention (tests/support/gpu_chunk_dispatch_instrument.h's sibling counters) --
// declared in every TU under SUPERSLM_ENABLE_GPU_CHUNK_DISPATCH_INSTRUMENT, defined exactly once,
// here.
extern const uint32_t kT2169TdrSafeMaxChunkTokens;

// T-2169 (Rung 2, design Sec5/Sec8): the real chunk-scoped entry point -- splits an admitted
// chunk of arbitrary size into a bounded sequence of sub-chunks, each driven through
// `SubmitOneSubChunkToFullDepthForG5Bridge` (above), each at most `kT2169TdrSafeMaxChunkTokens`
// tokens. Design Sec5's own resolution: "either (a) recording the whole admitted chunk into one
// list when it stays under both ceilings, or (b) splitting into a bounded number of sub-chunks,
// each its own command list/fence-wait pair, still far fewer than one per token."
//
// D-SLM3649 (ruled this rung, folding the TDR-measurement pass's own executed finding): the
// bound this design anticipated as TDR-derived is not what actually governs on the certified
// target hardware measured this pass (RTX 2080S, driver version confirmed via this machine's own
// installed NVIDIA driver at measurement time). The pure TDR arithmetic at the measured dispatch
// rate (~0.023 ms/dispatch, this model's own 672 dispatches/token) admits roughly 256 tokens
// under half this machine's own measured 8-second TDR delay -- far larger than any realistic
// chunk. A SMALLER, previously unanticipated ceiling binds first: driving a `chunk_tokens=8`
// sub-chunk through `SubmitOneSubChunkToFullDepthForG5Bridge` reproducibly crashes the process
// with `STATUS_STACK_OVERFLOW` (`0xC00000FD`); `chunk_tokens<=7` does not, verified repeatedly.
// Root-caused under `cdb` (`k`/`kb`, a fresh-process repro): the crash is a recursive cycle
// **entirely inside `nvwgf2umx.dll`** (NVIDIA's own D3D12 user-mode driver) -- every captured
// frame across the full depth cdb reports is inside that module; ZERO frames from this
// executable, from `d3d12.dll`, or from `dxgi.dll` appear anywhere in the cycle. **This is a
// defect in pre-existing, third-party, shipped code (the GPU vendor's own driver), not in this
// design's own new primitive** -- named loudly, per this ticket's own standing instruction, and
// ruled here rather than silently worked around: there is no file:line in this codebase to fix,
// because the recursion never enters this codebase's own call frames. The only lever this
// project has is never asking the driver to hold open the specific dispatch-recording pattern
// that triggers it -- i.e., the sub-chunk bound itself. `kT2169TdrSafeMaxChunkTokens` is
// therefore set from the SMALLER of the two ceilings (design Sec5's own "the smaller of the two
// ceilings governs," now with a third ceiling in the comparison), with the SAME half-margin
// discipline design Sec5 already applies to the TDR ceiling: **4**, half of the confirmed-crashing
// 8, leaving a full token of margin below the confirmed-safe 7 as well. This is deliberately
// conservative rather than pinned to the exact 7/8 boundary -- the boundary was measured on one
// specific real-artifact configuration (Qwen2.5-1.5B, this context_cap) and a driver-internal
// recursion depth is not guaranteed to trip at the identical token count under a different
// weight/adapter/context_cap combination; halving the confirmed-crashing point, not the
// confirmed-safe one, is the conservative direction.
const uint32_t kT2169TdrSafeMaxChunkTokens = 4;

// T-2169 (Rung 3/4, design Sec5 2b, declared gpu_port.h): the SeqState embedding-byte block
// size for one token, `[0, SeqScaleOff(hidden_size)+16)` -- exposed so a pre-scan caller
// (gpu_1p0.cpp) can size/index its own per-chunk embedding buffer without re-deriving
// `SeqScaleOff`'s own alignment arithmetic (internal linkage in this file, `Align8U32`/
// `SeqScaleOff` above).
uint32_t T2169SeqEmbeddingBlockBytes(uint32_t hidden_size) { return SeqScaleOff(hidden_size) + 16u; }

superslm::SslmForwardStatus SubmitChunkToFullDepthForG5Bridge(
    superslm::SequenceLayerState& seq, const superslm::LayerWeights* layers,
    uint32_t num_hidden_layers, size_t hidden_size, size_t head_dim, size_t num_key_value_heads,
    size_t intermediate_size, int64_t context_cap, const superslm::SslmTensorManifest& rope_tables,
    uint8_t* workspace, size_t workspace_size, const uint8_t* chunk_embedding_bytes,
    uint32_t chunk_len, ID3D12Resource* external_kv_resident,
    bool* io_external_kv_needs_resume_barrier, ID3D12Resource* external_weights_resident,
    ID3D12Resource* external_rope_cos_resident, ID3D12Resource* external_rope_sin_resident,
    bool external_rope_has, uint64_t external_rope_cos_elems, uint64_t external_rope_sin_elems,
    const GpuAdapterBridge* adapter_bridge, GpuLayerLoopInFlight** out_inflight) {
	if (out_inflight) *out_inflight = nullptr;
	if (chunk_len == 0) {
		// Nothing to submit -- no guard ladder has run, so this is not itself a rejection; the
		// caller (Rung 3/4's own admit_count==0 case) is expected to handle a zero-length chunk
		// before ever reaching here. Defensive, not a load-bearing path this design specifies.
		return superslm::SslmForwardStatus::Ok;
	}
	const size_t embed_block_bytes = static_cast<size_t>(SeqScaleOff(static_cast<uint32_t>(hidden_size))) + 16u;
	uint32_t submitted = 0;
	while (submitted < chunk_len) {
		const uint32_t remaining = chunk_len - submitted;
		const uint32_t this_sub_chunk =
		    remaining < kT2169TdrSafeMaxChunkTokens ? remaining : kT2169TdrSafeMaxChunkTokens;
		const bool is_last = (submitted + this_sub_chunk) >= chunk_len;
		GpuLayerLoopInFlight* inflight = nullptr;
		const superslm::SslmForwardStatus submit_status = SubmitOneSubChunkToFullDepthForG5Bridge(
		    seq, layers, num_hidden_layers, hidden_size, head_dim, num_key_value_heads,
		    intermediate_size, context_cap, rope_tables, workspace, workspace_size,
		    chunk_embedding_bytes + static_cast<size_t>(submitted) * embed_block_bytes, this_sub_chunk,
		    external_kv_resident, io_external_kv_needs_resume_barrier, external_weights_resident,
		    external_rope_cos_resident, external_rope_sin_resident, external_rope_has,
		    external_rope_cos_elems, external_rope_sin_elems, adapter_bridge, &inflight);
		if (submit_status != superslm::SslmForwardStatus::Ok) {
			return submit_status;
		}
		if (is_last) {
			// The caller's own async contract (design Sec4.2/Sec4.3): only the FINAL
			// (sub-)chunk's own inflight token is handed back unfinished -- the caller drives
			// its own Finish, exactly as it already does for the single-sub-chunk case.
			if (out_inflight) *out_inflight = inflight;
			return superslm::SslmForwardStatus::Ok;
		}
		// Every non-final sub-chunk is finished SYNCHRONOUSLY, here, before the next sub-chunk's
		// own PrepareGpuLayerLoopChunkOpenState call -- required, not merely convenient: that
		// call reads `seq.context_length`/`workspace` fresh at ITS OWN chunk-open boundary
		// (D-SLM3612's own 2a ruling, generalized across sub-chunks), so the previous
		// sub-chunk's own commit must already be visible to the host before the next one's
		// chunk-open counter is read. A D3D12 command list also cannot be resumed after its own
		// Close()+ExecuteCommandLists() -- Reset() for the next sub-chunk requires the GPU be
		// done with the allocator the current list used, which the fence wait below guarantees.
		int32_t ready = 0;
		const superslm::SslmForwardStatus finish_status =
		    RunLayerLoopGpuFinish(inflight, seq, workspace, /*block=*/1, &ready);
		if (finish_status != superslm::SslmForwardStatus::Ok) {
			return finish_status;
		}
		// D-SLM3649 (fold, found by execution): `commit_site.hlsl`'s own per-layer increment
		// leaves the device's persistent `SeqState.layer_index` field at `num_hidden_layers`
		// after a token's own LAST layer commits -- it is never auto-wrapped back to 0
		// device-side. Every existing single-token caller never notices, because
		// `sslm_gpu_seq_embed_token`'s own HOST-side write (`seq->layer_index = 0;`,
		// gpu_1p0.cpp) resets it before the NEXT token's own call ever re-uploads the SeqState
		// (Sec5.1's own guard-ladder trace: `layer_index==0` is what makes tokens 2..N of a
		// SINGLE chunk vacuously fresh, because nothing within one chunk primitive call ever
		// reads this field back to the host mid-chunk). This wrapper is the one caller that
		// DOES read it back mid-sequence (`RunLayerLoopGpuFinish`, immediately above) and then
		// hands `seq` to a SECOND `PrepareGpuLayerLoopChunkOpenState` call, whose own guard
		// ladder (`seq.layer_index >= num_hidden_layers` -> `SequenceAlreadyComplete`) reads the
		// terminal value this sub-chunk's own last token legitimately left behind and rejects a
		// perfectly healthy continuation. The reset every embed_token call already performs for
		// the single-token path is performed here, once, for the identical reason: this
		// sub-chunk's own last token has fully committed and the sequence is ready for a fresh
		// token, exactly the state embed_token's own reset always signals.
		seq.layer_index = 0;
		submitted += this_sub_chunk;
	}
	return superslm::SslmForwardStatus::Ok;  // unreachable (chunk_len > 0 always returns inside the loop)
}

// T-2113 (B5, design Sec4.2/Sec6.2/Sec10 B5): the FINISH half of the async split --
// declared in gpu_port.h. Byte-for-byte the tail RunLayerLoopGpu's own synchronous body
// always ran after its fence-wait (the GPU-timing readback, the SeqState/KV readback,
// the sticky-tag decode) -- only WHERE the fence-wait itself sits has moved: `block==1`
// waits unconditionally (matching the old unconditional wait exactly); `block==0` polls
// `GetCompletedValue()` and, if not yet signaled, returns immediately with `*out_ready=0`
// and `inflight` untouched -- still owned by the caller, still pollable.
superslm::SslmForwardStatus RunLayerLoopGpuFinish(GpuLayerLoopInFlight* inflight,
                                                    superslm::SequenceLayerState& seq,
                                                    uint8_t* workspace, int32_t block,
                                                    int32_t* out_ready) {
	if (out_ready) *out_ready = 0;
	if (!inflight || !inflight->dev) {
		return superslm::SslmForwardStatus::GpuAllocationFailed;  // caller error: no live token
	}
	harness::Device& dev = *inflight->dev;
	const bool signaled = dev.fence->GetCompletedValue() >= inflight->fence_val;
	if (!signaled && !block) {
		// Pure poll, nothing ready yet -- design Sec4.2: "*out_ready=0, no state change."
		// `inflight` is NOT consumed; the caller polls again later against the SAME token.
		return superslm::SslmForwardStatus::Ok;
	}
	// T-2113 (B5, design Sec9's own DeviceLost row): the pre-split synchronous function
	// left this whole tail (the wait, the readback Maps) OUTSIDE any try/catch, on the
	// pre-1.0 substrate's own footing where an uncaught exception merely terminated a
	// C++ test binary. `RunLayerLoopGpuFinish` is reached from `sslm_gpu_ready`, a
	// C-ABI-shaped, status-returning 1.0 API entry point -- design Sec9's own table
	// promises `DeviceLost` is DELIVERABLE through a defined channel when "a lost device
	// is discovered," never as an uncaught exception escaping across that boundary and
	// crashing the caller's whole process. Reproduced by execution (this section's own
	// violation-pin commissioning, StandardsDocument.md Sec5.4): a REAL device fault
	// (DXGI_ERROR_DEVICE_REMOVED) triggered by the dropped-UAV-rebind pin's own
	// out-of-bounds write, discovered here (at the fence-wait/readback, not during
	// Submit's own recording) with no catch present, terminated the process instead of
	// returning `GpuDeviceRemoved` through the defined channel. Fixed here, the same
	// `GetDeviceRemovedReason()`-queried disposition Submit's own generic catch already
	// uses for the analogous failure discovered during recording.
	//
	// `owned` takes ownership of `inflight` BEFORE the try block -- not after the wait,
	// as an earlier draft had it -- specifically so RAII covers a throw from ANYWHERE
	// in the try, including the wait itself, with no path that could reach the catch
	// below still holding a naked, undeleted `inflight`, and no path where the catch
	// would need to `delete` it manually (which risked a double-free against `owned`'s
	// own destructor, on a throw AFTER `owned` had already taken ownership -- caught by
	// this section's own review before it was ever exercised on real hardware).
	std::unique_ptr<GpuLayerLoopInFlight> owned(inflight);  // consumed, exactly once, from here on
	try {
	if (!signaled) {
		SSLM_GPU_HR(dev.fence->SetEventOnCompletion(inflight->fence_val, dev.fence_event));
		WaitForSingleObject(dev.fence_event, INFINITE);
	}
	// From here down: BYTE-FOR-BYTE the old synchronous tail, reading from `inflight`'s
	// own fields instead of this function's own locals (they were this function's own
	// locals, in the pre-split RunLayerLoopGpu; Submit captured them into `inflight`
	// immediately after recording, before this call ever ran).
	const auto t_submit_wait_end = std::chrono::steady_clock::now();
	g_last_call_timing.submit_wait_ms =
	    std::chrono::duration<double, std::milli>(t_submit_wait_end - owned->t_record_end).count();

	if (dev.timestamp_frequency > 0 && owned->dispatch_count_this_call > 0) {
		std::vector<UINT64> ts(owned->dispatch_count_this_call + 1, 0);
		void* qp = nullptr;
		D3D12_RANGE qrange{0, ts.size() * sizeof(UINT64)};
		if (SUCCEEDED(dev.timestamp_readback->Map(0, &qrange, &qp))) {
			std::memcpy(ts.data(), qp, ts.size() * sizeof(UINT64));
			D3D12_RANGE qnone{0, 0};
			dev.timestamp_readback->Unmap(0, &qnone);
			const double freq = static_cast<double>(dev.timestamp_frequency);
			g_last_call_timing.gpu_busy_ms =
			    static_cast<double>(ts.back() - ts.front()) / freq * 1000.0;
			g_last_call_per_dispatch_ms.resize(owned->dispatch_count_this_call);
			for (uint32_t d = 0; d < owned->dispatch_count_this_call; ++d) {
				g_last_call_per_dispatch_ms[d] =
				    static_cast<double>(ts[d + 1] - ts[d]) / freq * 1000.0;
			}
		}
	}

	const uint32_t H = owned->hidden_size_h;
	const uint32_t HD = owned->head_dim_hd;
	const auto t_readback_start = std::chrono::steady_clock::now();
	std::vector<uint8_t> seq_out(owned->seq_bytes_size);
	{
		void* p = nullptr;
		D3D12_RANGE range{0, seq_out.size()};
		SSLM_GPU_HR(owned->seq_readback->Map(0, &range, &p));
		std::memcpy(seq_out.data(), p, seq_out.size());
		D3D12_RANGE none{0, 0};
		owned->seq_readback->Unmap(0, &none);
	}
	{
		const size_t kv_readback_size = owned->kv_row_offsets.size() * static_cast<size_t>(HD);
		void* p = nullptr;
		D3D12_RANGE range{0, kv_readback_size};
		SSLM_GPU_HR(owned->kv_readback->Map(0, &range, &p));
		const uint8_t* src = static_cast<const uint8_t*>(p);
		for (size_t r = 0; r < owned->kv_row_offsets.size(); ++r) {
			std::memcpy(workspace + owned->kv_row_offsets[r], src + r * static_cast<size_t>(HD), HD);
		}
		D3D12_RANGE none{0, 0};
		owned->kv_readback->Unmap(0, &none);
	}
	g_last_call_timing.readback_ms =
	    std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_readback_start)
	        .count();

	for (uint32_t i = 0; i < H; ++i) {
		int32_t v = 0;
		std::memcpy(&v, seq_out.data() + i * 4, 4);
		seq.hidden_codes[i] = static_cast<int8_t>(v);
	}
	int64_t hs_m, hs_e, lidx32, sat_lo32, sat_hi32, ctxlen, sticky_tag;
	std::memcpy(&hs_m, seq_out.data() + SeqScaleOff(H) + 0, 8);
	std::memcpy(&hs_e, seq_out.data() + SeqScaleOff(H) + 8, 8);
	uint32_t lidx_u, lo_u, hi_u;
	std::memcpy(&lidx_u, seq_out.data() + SeqLayerIdxOff(H), 4);
	std::memcpy(&lo_u, seq_out.data() + SeqSatLoOff(H), 4);
	std::memcpy(&hi_u, seq_out.data() + SeqSatHiOff(H), 4);
	std::memcpy(&ctxlen, seq_out.data() + SeqCtxLenOff(H), 8);
	std::memcpy(&sticky_tag, seq_out.data() + SeqStickyOff(H), 8);
	(void)lidx32; (void)sat_lo32; (void)sat_hi32;
	seq.hidden_scale.m = hs_m;
	seq.hidden_scale.e = hs_e;
	seq.layer_index = lidx_u;
	seq.kv_saturation_count = (static_cast<uint64_t>(hi_u) << 32) | static_cast<uint64_t>(lo_u);
	seq.context_length = ctxlen;

	if (out_ready) *out_ready = 1;
	return DecodeStickyTag(sticky_tag);
	} catch (const std::exception&) {
		// T-2101/T-2057's own disposition (superslm_gpu.cpp, Submit's own catch),
		// applied here for the identical reason: `GpuDeviceRemoved` iff the device is
		// confirmed gone right now, `GpuAllocationFailed` otherwise (a transient/size-
		// dependent failure at the readback, device still alive). `owned` (declared
		// BEFORE this try, above) frees `inflight` via its own destructor during stack
		// unwinding, regardless of where inside the try the throw happened -- nothing
		// here frees it a second time.
		if (out_ready) *out_ready = 1;  // a terminal status, not a "still pending" one
		const HRESULT device_removed_reason = dev.dev->GetDeviceRemovedReason();
		return device_removed_reason != S_OK ? superslm::SslmForwardStatus::GpuDeviceRemoved
		                                      : superslm::SslmForwardStatus::GpuAllocationFailed;
	}
}

// T-2113 (B5): RunLayerLoopGpu (gpu_port.h) is now a two-line wrapper -- Submit
// immediately followed by a BLOCKING Finish. Every instruction either one executes is
// unchanged from the pre-split function; only the function-call boundary between them
// is new, and it is crossed here, in the same order, on the same thread, before this
// wrapper ever returns -- byte-for-byte identical observable behavior for all ~40
// existing callers (none of them pass the B5-only trailing external-weights/external-
// rope parameters, all of which default to "off").
superslm::SslmForwardStatus RunLayerLoopGpu(superslm::SequenceLayerState& seq,
                                             const superslm::LayerWeights* layers,
                                             uint32_t num_hidden_layers, uint32_t layer_budget,
                                             size_t hidden_size, size_t head_dim,
                                             size_t num_key_value_heads, size_t intermediate_size,
                                             int64_t context_cap,
                                             const superslm::SslmTensorManifest& rope_tables,
                                             uint8_t* workspace, size_t workspace_size,
                                             ID3D12Resource* external_kv_resident,
                                             bool* io_external_kv_needs_resume_barrier) {
	GpuLayerLoopInFlight* inflight = nullptr;
	const superslm::SslmForwardStatus submit_status = RunLayerLoopGpuSubmit(
	    seq, layers, num_hidden_layers, layer_budget, hidden_size, head_dim, num_key_value_heads,
	    intermediate_size, context_cap, rope_tables, workspace, workspace_size, external_kv_resident,
	    io_external_kv_needs_resume_barrier, &inflight);
	if (!inflight) {
		return submit_status;  // rejected (a guard, or an exception) before submission
	}
	int32_t ready = 0;
	return RunLayerLoopGpuFinish(inflight, seq, workspace, /*block=*/1, &ready);
}

// T-2052 (item 3): the device-observable N3's own residency cache was
// missing (Claude/Curie/t2019-gpu-serial-red-suite-2026-08-13.md §13.2) --
// see gpu_port.h's own declaration comment for the full contract.
bool LastWeightUploadWasSkipped() { return g_last_weight_upload_was_skipped; }

// T-2101 (S3, code review 6d9e04e-t2101-gpu-throughput-review.md): the ceiling-division primitive
// `ComputeGpuGemmSiteGroupPlan` (below) builds on.
uint32_t ComputeGpuGemmGroupCount(uint32_t out_channels, uint32_t threads_per_group) {
	return (out_channels + threads_per_group - 1u) / threads_per_group;
}

// T-2101 (S3-prime, code review 6d9e04e-t2101-gpu-throughput-review.md, confirmation pass @
// f7026db): the ONE source of every one of the (T-2113 B4: six, KvProj added) split GEMM
// dispatches' own grid size -- `RunLayerLoopGpu`'s own `bind_and_dispatch` calls for these sites
// pass this function's own `.groups`/`.lanes` directly, with no intermediate per-call-site local
// variable that a hand edit could diverge from it. `tests/test_main.cpp`'s own
// `TestT2101_ComputeGpuGemmSiteGroupPlan_RealDimensions` calls this SAME function at the real
// model's own dimensions, no device required -- confirmed by execution this round: mutating this
// function's own body (forcing `plan.groups = 1u` unconditionally) reddens that pinned cell
// directly, at 1536/8960, which the PRIOR round's falsification (mutating a downstream local
// variable the test never read) could not do.
//
// T-2101 (N2, code review 6d9e04e-t2101-gpu-throughput-review.md, second confirmation pass): a
// sixth `GpuGemmSplitSite` enumerator with no matching `case` used to leave `plan.threads_per_group`
// at its default-initialized 0, and the line below then computed
// `ComputeGpuGemmGroupCount(0, 0)` -- `(0 + 0 - 1u) / 0u`, an integer division by zero, a hardware
// fault with no compiler warning (`/W4`, no `/WX`, and MSVC's unhandled-enumerator diagnostics
// C4061/C4062 are off by default) inside the ONE function this whole round's own commission exists
// to make the single source of truth. Closed two ways: an explicit `default:` that fails LOUDLY
// (the same `GpuGemmGroupArithmeticError`/status pair S4 already built, rather than a silent
// fallback or an assert compiled out of a release build), and a post-switch check that
// `threads_per_group != 0` regardless of how it got there -- so a future case that sets
// `out_channels` but leaves `threads_per_group` at 0 by a copy-paste mistake is caught by the SAME
// guard, not only the enumerator-completeness one.
// T-2113 (B4, design Sec3/Sec6.1): re-derived from Claude/Laplace/t2105-gpu-speed-ceiling-
// 2026-08-14.md Sec2's own dispatch-geometry table (the measured-optimal construction the
// section's own 58 tok/s headline was reproduced under). `threads_per_group` is now fixed
// at 256 for every site (the transposed partition always launches a full group; `lanes`
// decides how those 256 threads split across channels, not the group's own launch width),
// and `lanes` is a FIXED, chosen production value per site -- not read from the environment
// on the hot path, unlike T-2105's own experiment-only sweep knob. `kv_out_channels` is read
// only by `KvProj` (its own dispatch covers 2*kv_hidden_size channels, K and V packed into
// one grid, `kv_proj_gemm_site.hlsl`'s own header comment) and ignored by every other site.
GpuGemmSiteGroupPlan ComputeGpuGemmSiteGroupPlan(GpuGemmSplitSite site, uint32_t hidden_size,
                                                  uint32_t kv_out_channels,
                                                  uint32_t intermediate_size) {
	GpuGemmSiteGroupPlan plan;
	switch (site) {
		case GpuGemmSplitSite::QProj:
			plan.out_channels = hidden_size;
			plan.threads_per_group = 256u;
			plan.lanes = 32u;
			break;
		case GpuGemmSplitSite::OProj:
			plan.out_channels = hidden_size;
			plan.threads_per_group = 256u;
			plan.lanes = 32u;
			break;
		case GpuGemmSplitSite::DownProj:
			plan.out_channels = hidden_size;
			plan.threads_per_group = 256u;
			plan.lanes = 64u;  // in_channels=8960 favors more cooperating lanes/channel than the
			                    // other five sites' in_channels=1536 (T-2105 Sec2's own table).
			break;
		case GpuGemmSplitSite::GateProj:
			plan.out_channels = intermediate_size;
			plan.threads_per_group = 256u;
			plan.lanes = 32u;
			break;
		case GpuGemmSplitSite::UpProj:
			plan.out_channels = intermediate_size;
			plan.threads_per_group = 256u;
			plan.lanes = 32u;
			break;
		case GpuGemmSplitSite::KvProj:
			// T-2113 (B4, D-SLM3341): the dispatch covers BOTH halves -- K's own kv_hidden_size
			// channels and V's own -- packed into one grid, so the grid is sized over
			// 2*kv_hidden_size (`kv_out_channels`, already doubled by the caller) while the
			// per-half `out_channels` the shader itself bounds-checks against stays
			// kv_hidden_size (kv_proj_gemm_site.hlsl computes that from g_num_kv_heads*g_head_dim
			// directly, not from this plan).
			plan.out_channels = kv_out_channels;
			plan.threads_per_group = 256u;
			plan.lanes = 32u;
			break;
		default:
			throw GpuGemmGroupArithmeticError(
			    "ComputeGpuGemmSiteGroupPlan: unhandled GpuGemmSplitSite enumerator -- no "
			    "out_channels/threads_per_group case exists for this site");
	}
	if (plan.threads_per_group == 0u) {
		throw GpuGemmGroupArithmeticError(
		    "ComputeGpuGemmSiteGroupPlan: threads_per_group == 0 after the switch -- would divide "
		    "by zero computing the group count");
	}
	if (plan.lanes == 0u || plan.lanes > plan.threads_per_group ||
	    (plan.threads_per_group % plan.lanes) != 0u) {
		throw GpuGemmGroupArithmeticError(
		    "ComputeGpuGemmSiteGroupPlan: lanes must be a positive divisor of threads_per_group -- "
		    "the transposed partition requires an exact channels-per-group split");
	}
	// T-2113 (B4): a group of `threads_per_group` (256) threads now covers
	// `threads_per_group / lanes` OUTPUT CHANNELS, not `threads_per_group` of them -- the
	// remaining `lanes` threads per channel cooperate on that channel's own in_channels-wide
	// reduction. The grid size is therefore the ceiling over channels-per-group, and the
	// coverage guard in `RunLayerLoopGpu` checks that quantity.
	plan.channels_per_group = plan.threads_per_group / plan.lanes;
	plan.groups = ComputeGpuGemmGroupCount(plan.out_channels, plan.channels_per_group);
	return plan;
}

GpuCallTiming LastCallTiming() { return g_last_call_timing; }

std::vector<double> LastCallPerDispatchTimingsMs() { return g_last_call_per_dispatch_ms; }

const int8_t* KeyRowGpu(const uint8_t* workspace, uint32_t layer, int64_t context_cap,
                         size_t num_kv_heads, size_t head_dim, size_t kv_head, int64_t position) {
	// T-2032: the KV cache twin's own addressing, bit-exact against
	// superslm::KeyRow (forward_sites.cpp's S3.7 accessor block) -- real once
	// RunLayerLoopGpu (above) has written it; site_common.hlsli's
	// KvHalfOffsetGpu/KvRowOffsetWithinHalfGpu are the SAME formula, ported.
	const size_t half_off = static_cast<size_t>(layer) * static_cast<size_t>(context_cap) * num_kv_heads *
	                             head_dim * 2u;
	const size_t row_off =
	    kv_head * static_cast<size_t>(context_cap) * head_dim + static_cast<size_t>(position) * head_dim;
	return reinterpret_cast<const int8_t*>(workspace) + half_off + row_off;
}

const int8_t* ValueRowGpu(const uint8_t* workspace, uint32_t layer, int64_t context_cap,
                           size_t num_kv_heads, size_t head_dim, size_t kv_head, int64_t position) {
	const size_t half_off = static_cast<size_t>(layer) * static_cast<size_t>(context_cap) * num_kv_heads *
	                             head_dim * 2u;
	const size_t k_store_size = static_cast<size_t>(context_cap) * num_kv_heads * head_dim;
	const size_t row_off =
	    kv_head * static_cast<size_t>(context_cap) * head_dim + static_cast<size_t>(position) * head_dim;
	return reinterpret_cast<const int8_t*>(workspace) + half_off + k_store_size + row_off;
}

// ===========================================================================
// B7 (Sec5.8/Sec11 B7, D-SLM3069/3070/3072): the dispatch_budget contract,
// whole-layer quanta. Pure host-side policy -- no dispatch is issued to
// answer this question, only planned (gpu_port.h's own header note) -- so
// this is a direct, executed realization of Sec5.8's own formula, matching
// t2019_b7::ExpectedDispatchBudgetPlan (tests/test_main.cpp) exactly: floor
// division by 24 -- T-2113 (B4, design Sec3/Sec6.1) re-derived from the
// 17 = 16 sites + 1 commit figure this constant carried before B4 ported
// the per-layer dispatch chain onto its own production geometry -- capped
// at the layers remaining in the current token, DispatchBudgetTooSmall iff
// the capped result is zero.
// ===========================================================================

SslmGpuStatus PlanDispatchBudgetGpu(uint32_t dispatch_budget, uint32_t num_hidden_layers,
                                     uint32_t current_layer_position, uint32_t* out_layers_to_issue) {
	// T-2114 (M1): kDispatchesPerLayer now lives at namespace scope (gpu_port.h), the one
	// source this function and sslm_decode_step_batch_gpu's own budget arithmetic (gpu_1p0.cpp)
	// both read.
	const uint32_t remaining = num_hidden_layers - current_layer_position;
	uint32_t layers = dispatch_budget / kDispatchesPerLayer;  // floor division, never a ceiling
	if (layers > remaining) layers = remaining;                // token-boundary cap (never spills a token)
	if (out_layers_to_issue) *out_layers_to_issue = layers;
	return (layers == 0) ? SslmGpuStatus::DispatchBudgetTooSmall : SslmGpuStatus::Ok;
}

// ===========================================================================
// Sec5.9 (D-SLM3076-3079): the asynchronous sequence lifecycle, Idle ->
// Submitted -> Completed -> Idle. Every `CallProceedsOrBusy_*` function is
// the POLICY half of its named ABI call: given whether the sequence (or, for
// unmap, the model) currently has Submitted work outstanding, does the call
// proceed or return SSLM_BUSY synchronously -- no device fence or command
// list is needed to answer this, only the state predicate itself (D-SLM3077:
// each of the five is refused for a distinct, source-grounded ordering
// hazard against in-flight device work).
// ===========================================================================

bool CallProceedsOrBusy_DecodeStepGpu(bool sequence_is_submitted) { return !sequence_is_submitted; }
bool CallProceedsOrBusy_SeqSave(bool sequence_is_submitted) { return !sequence_is_submitted; }
bool CallProceedsOrBusy_SeqReset(bool sequence_is_submitted) { return !sequence_is_submitted; }
bool CallProceedsOrBusy_SeqRelease(bool sequence_is_submitted) { return !sequence_is_submitted; }
// Model-wide (D-SLM3079): checked against the SET of sequences created
// against the model handle, not a single targeted one -- the caller already
// reduces "any sequence Submitted" to one bool before calling this, matching
// the model-wide check's own predicate.
bool CallProceedsOrBusy_ModelUnmap(bool any_sequence_submitted) { return !any_sequence_submitted; }

// sslm_gpu_ready's dual role (D-SLM3078): exempt from SSLM_BUSY by
// construction -- the one call legal against a Submitted sequence, since it
// is how a host polls out of that state. Before the fence signals: ordinary
// polling, *ready=0, sequence stays Submitted. After: collapses
// Submitted -> Completed -> Idle in this one call, *ready=1, and hands back
// the aggregated forward status through *out_status. This policy function
// carries no real submitted-work state of its own (no live sequence handle
// is threaded through the test's own call shape, gpu_port.h) -- the status
// it reports on the completed path is Ok, the only value this suite's own
// TestT2019_Sec59_GpuReadyDualRole cell checks the ready flag against; the
// real per-sequence aggregated status is RunLayerLoopGpu's own return value,
// surfaced through the production sslm_gpu_ready wrapper once that ABI
// surface is built (not this red suite's own scope, per gpu_port.h's header).
bool GpuReadySignalsCompletion(bool fence_signaled, int32_t* out_ready,
                                superslm::SslmForwardStatus* out_status) {
	if (!fence_signaled) {
		if (out_ready) *out_ready = 0;
		return true;  // never SSLM_BUSY; ordinary polling, still Submitted
	}
	if (out_ready) *out_ready = 1;
	if (out_status) *out_status = superslm::SslmForwardStatus::Ok;
	return true;  // never SSLM_BUSY; collapses to Idle
}

// ===========================================================================
// B8 (Sec11 B8, D-SLM3034/D-SLM3080): device residency round-trip,
// mechanism-level. SuperSLM_Plan.md Sec12's sslm_seq_save/restore C-ABI
// wrapper is not yet built on any backend (gpu_port.h's own header note), so
// this red suite gates the device-resident round-trip MECHANISM the design
// commits to -- byte-serializing exactly the SequenceLayerState fields the
// design names (Sec10 dim 9: layer_index, kv_saturation_count,
// context_length; hidden_scale carried for completeness though this
// suite's own oracle does not check it) plus the caller's workspace bytes
// (the K/V cache, S3.7's own addressable unit).
//
// T-2114 (C1, Claude/Poirot/50f3d5d-t2113-1p0-gpu-core-build-review.md):
// this function's own header comment used to say hidden_codes' pointee never
// round-trips through this blob, "the caller's own responsibility" -- true
// under the pre-1.0 convention where hidden_codes was a caller-owned
// pointer. T-2113's 1.0 API moved ownership of hidden_codes INTO the
// SslmGpuSequenceHandle (design Sec5.3), and sslm_gpu_seq_restore rebuilds
// its fresh handle through sslm_gpu_seq_create, which zeroes hidden_codes --
// so under the 1.0 contract, "the caller's own responsibility" is nobody's
// responsibility, and a mid-token restore silently resumed decode from an
// all-zero residual stream paired with a restored non-zero hidden_scale,
// returning Ok. This function now carries hidden_codes through the blob,
// gated by a caller-supplied `hidden_codes_size` exactly like `workspace`
// already is (0 = the caller does not want it round-tripped -- the pre-1.0
// callers in test_main.cpp pass 0 and keep their own established contract;
// the 1.0 `sslm_gpu_seq_restore` path (gpu_1p0.cpp) passes the model's real
// hidden_size and gets the full round-trip). The blob format is versioned
// honestly: the magic changed from the pre-fix 'SSLM' (v1) to 'SLM2' (v2,
// T-2114 C1: +hidden_codes_size/bytes) to 'SLM3' (v3, T-2113 P2, design
// Sec4.2/Sec22, D-SLM3415: +model_content_hash) -- an older-format blob is
// rejected cleanly at the first check below, never misread against a newer,
// larger header layout.
// ===========================================================================

namespace {
struct GpuSeqBlobHeader {
	uint32_t magic;  // 'SLM3' little-endian -- v3 format (T-2113 P2: +model_content_hash). Every
	                  // older magic ('SSLM' v1, 'SLM2' v2) is a DIFFERENT value on purpose: an
	                  // older-format blob fails the magic check below rather than being misread
	                  // against this larger header.
	uint32_t layer_index;
	int64_t hidden_scale_m;
	int64_t hidden_scale_e;
	uint64_t kv_saturation_count;
	int64_t context_length;
	uint64_t hidden_codes_size;  // T-2114 C1: byte count of the residual stream that follows
	                              // this header in the blob body, before the workspace bytes.
	uint64_t workspace_size;
	// T-2113 (P2, design Sec4.2/Sec22, D-SLM3415): the SAVING model's own 32-byte
	// `SslmModelView::RawIntegrityHash()` (design Sec5.1) -- appended LAST, the identical
	// append-only precedent T-2114 C1 already set for `hidden_codes_size`/`workspace_size`.
	// The 1.0 `sslm_gpu_seq_restore` entry point compares this against the TARGET model handle's
	// own hash and rejects a mismatch under `SSLM_RESTORE_MODEL_MISMATCH`, closing the identity
	// gap the N1 size-admissibility widening (Sec21) left open: a foreign blob whose derived
	// context_cap happens to be admissible against a same-shape-but-different model used to
	// restore silently.
	std::array<uint8_t, superslm::kIntegrityHashBytes> model_content_hash;
};
constexpr uint32_t kGpuSeqBlobMagic = 0x334D4C53u;  // "SLM3" little-endian u32 (v3: +model_content_hash)
}  // namespace

bool SaveGpuSequenceState(const superslm::SequenceLayerState& seq, size_t hidden_codes_size,
                           const uint8_t* workspace, size_t workspace_size,
                           const std::array<uint8_t, superslm::kIntegrityHashBytes>& model_content_hash,
                           void* out_blob, size_t* out_blob_size) {
	const size_t required = sizeof(GpuSeqBlobHeader) + hidden_codes_size + workspace_size;
	if (!out_blob_size) return false;
	if (!out_blob || *out_blob_size < required) {
		*out_blob_size = required;  // report the size a real call needs, not silently truncate
		return false;
	}
	GpuSeqBlobHeader hdr{};
	hdr.magic = kGpuSeqBlobMagic;
	hdr.layer_index = seq.layer_index;
	hdr.hidden_scale_m = seq.hidden_scale.m;
	hdr.hidden_scale_e = seq.hidden_scale.e;
	hdr.kv_saturation_count = seq.kv_saturation_count;
	hdr.context_length = seq.context_length;
	hdr.hidden_codes_size = static_cast<uint64_t>(hidden_codes_size);
	hdr.workspace_size = static_cast<uint64_t>(workspace_size);
	hdr.model_content_hash = model_content_hash;
	uint8_t* dst = static_cast<uint8_t*>(out_blob);
	std::memcpy(dst, &hdr, sizeof(hdr));
	size_t off = sizeof(hdr);
	if (hidden_codes_size > 0) {
		if (!seq.hidden_codes) return false;
		std::memcpy(dst + off, seq.hidden_codes, hidden_codes_size);
	}
	off += hidden_codes_size;
	if (workspace_size > 0) {
		if (!workspace) return false;
		std::memcpy(dst + off, workspace, workspace_size);
	}
	*out_blob_size = required;
	return true;
}

bool RestoreGpuSequenceState(const void* blob, size_t blob_size, superslm::SequenceLayerState* out_seq,
                              size_t hidden_codes_size, uint8_t* out_workspace, size_t workspace_size) {
	if (!blob || !out_seq || blob_size < sizeof(GpuSeqBlobHeader)) return false;
	GpuSeqBlobHeader hdr{};
	std::memcpy(&hdr, blob, sizeof(hdr));
	if (hdr.magic != kGpuSeqBlobMagic) return false;
	if (hdr.hidden_codes_size != static_cast<uint64_t>(hidden_codes_size)) return false;  // size mismatch, refuse
	if (hdr.workspace_size != static_cast<uint64_t>(workspace_size)) return false;  // size mismatch, refuse
	if (blob_size < sizeof(hdr) + hdr.hidden_codes_size + hdr.workspace_size) return false;
	out_seq->layer_index = hdr.layer_index;
	out_seq->hidden_scale.m = hdr.hidden_scale_m;
	out_seq->hidden_scale.e = hdr.hidden_scale_e;
	out_seq->kv_saturation_count = hdr.kv_saturation_count;
	out_seq->context_length = hdr.context_length;
	// T-2114 (C1): hidden_codes now round-trips through the blob body, immediately after the
	// header and before the workspace bytes -- gated on hidden_codes_size exactly like
	// workspace below is gated on workspace_size. `out_seq->hidden_codes` must already be a
	// valid, correctly-sized pointer when hidden_codes_size > 0 (the 1.0 caller,
	// sslm_gpu_seq_restore, guarantees this: it builds `fresh` through sslm_gpu_seq_create
	// first, which sets `fresh->live_state.hidden_codes = fresh->hidden_codes.data()` before
	// this function is ever called).
	const uint8_t* hidden_codes_src = static_cast<const uint8_t*>(blob) + sizeof(hdr);
	if (hidden_codes_size > 0) {
		if (!out_seq->hidden_codes) return false;
		std::memcpy(out_seq->hidden_codes, hidden_codes_src, hidden_codes_size);
	}
	if (workspace_size > 0) {
		if (!out_workspace) return false;
		const uint8_t* src = hidden_codes_src + hidden_codes_size;

		// T-2045 (S6, Claude/Poirot/82cfca7-gpu-serial-port-build-review.md):
		// this function's own declared contract sites the restore-time device
		// allocation and host-to-device upload HERE (the declaration comment on
		// `RestoreGpuSequenceState` (`gpu_port.h`)) -- a plain host memcpy would pass this gate identically
		// with no GPU present. The workspace bytes now round-trip through a
		// real device: upload heap -> DEFAULT-heap device buffer (the
		// "restored-sequence device allocation" itself) -> readback heap ->
		// `out_workspace`, one real command-list submission, fence-waited
		// before this call returns.
		harness::Device& dev = harness::GetDevice();
		if (!dev.available) return false;
		// T-2114 (S4): fault-injection hook, zero-overhead unarmed (MaybeThrowInjectedO11AllocFault's
		// own header comment) -- lets a test arm kO11AllocInjectionSiteSeqRestore (gpu_port.h) and
		// confirm the exception this throws is caught by sslm_gpu_seq_restore's own try (gpu_1p0.cpp)
		// rather than escaping the status-returning API boundary.
		MaybeThrowInjectedO11AllocFault(superslm_gpu::kO11AllocInjectionSiteSeqRestore);
		SSLM_GPU_HR(dev.alloc->Reset());
		SSLM_GPU_HR(dev.list->Reset(dev.alloc.Get(), nullptr));
		auto upload_buf = dev.Upload(src, workspace_size);
		auto device_buf = dev.MakeBuffer(workspace_size, D3D12_HEAP_TYPE_DEFAULT,
		                                  D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST);
		dev.list->CopyResource(device_buf.Get(), upload_buf.Get());
		D3D12_RESOURCE_BARRIER b{};
		b.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
		b.Transition.pResource = device_buf.Get();
		b.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_DEST;
		b.Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_SOURCE;
		b.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
		dev.list->ResourceBarrier(1, &b);
		auto readback_buf = dev.MakeBuffer(workspace_size, D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_FLAG_NONE,
		                                    D3D12_RESOURCE_STATE_COPY_DEST);
		dev.list->CopyResource(readback_buf.Get(), device_buf.Get());
		SSLM_GPU_HR(dev.list->Close());
		ID3D12CommandList* lists[] = {dev.list.Get()};
		dev.queue->ExecuteCommandLists(1, lists);
		SSLM_GPU_HR(dev.queue->Signal(dev.fence.Get(), ++dev.fence_val));
		if (dev.fence->GetCompletedValue() < dev.fence_val) {
			SSLM_GPU_HR(dev.fence->SetEventOnCompletion(dev.fence_val, dev.fence_event));
			WaitForSingleObject(dev.fence_event, INFINITE);
		}
		void* p = nullptr;
		D3D12_RANGE range{0, static_cast<SIZE_T>(workspace_size)};
		SSLM_GPU_HR(readback_buf->Map(0, &range, &p));
		std::memcpy(out_workspace, p, workspace_size);
		D3D12_RANGE none{0, 0};
		readback_buf->Unmap(0, &none);
	}
	return true;
}

// T-2113 (N1, design Sec4.2/Sec21): header-only peek -- reads `GpuSeqBlobHeader` and returns its
// `workspace_size` without touching the body or the device, so gpu_1p0.cpp's own
// `sslm_gpu_seq_restore` can derive the blob's own `context_cap` (Sec5.1's K/V sizing formula)
// before it allocates anything. Same magic/size checks `RestoreGpuSequenceState` already applies at its own header read,
// duplicated here rather than factored through it because this call must run BEFORE the fresh handle
// (and therefore the `hidden_codes_size`/`out_workspace`/`workspace_size` arguments that function
// requires) exists.
bool PeekGpuSeqBlobWorkspaceSize(const void* blob, size_t blob_size, uint64_t* out_workspace_size) {
	if (!blob || !out_workspace_size || blob_size < sizeof(GpuSeqBlobHeader)) return false;
	GpuSeqBlobHeader hdr{};
	std::memcpy(&hdr, blob, sizeof(hdr));
	if (hdr.magic != kGpuSeqBlobMagic) return false;
	*out_workspace_size = hdr.workspace_size;
	return true;
}

// T-2113 (P2, design Sec4.2/Sec22, routed `Claude/Poirot/50f3d5d-t2113-1p0-gpu-core-build-review.md`
// Sec15, D-SLM3415): the identity twin of `PeekGpuSeqBlobWorkspaceSize` above -- header-only peek,
// reads `GpuSeqBlobHeader.model_content_hash` without touching the body or the device, so
// `sslm_gpu_seq_restore` can compare the blob's own recorded model identity against the TARGET
// model handle's own hash before any device work. Same magic/size checks the other peek/restore
// functions already apply at their own header reads, duplicated here for the identical reason
// `PeekGpuSeqBlobWorkspaceSize`'s own header comment states: this call must run BEFORE the fresh
// handle exists.
bool PeekGpuSeqBlobModelHash(const void* blob, size_t blob_size,
                              std::array<uint8_t, superslm::kIntegrityHashBytes>* out_hash) {
	if (!blob || !out_hash || blob_size < sizeof(GpuSeqBlobHeader)) return false;
	GpuSeqBlobHeader hdr{};
	std::memcpy(&hdr, blob, sizeof(hdr));
	if (hdr.magic != kGpuSeqBlobMagic) return false;
	*out_hash = hdr.model_content_hash;
	return true;
}

// ===========================================================================
// B3 (Sec5.1/Sec11 B3): descriptor-table binding substrate. Real GPU
// dispatch: one shader-visible descriptor heap sized to n_arrays, one RAW
// buffer SRV per array, bound as a single DESCRIPTOR_TABLE root parameter
// (the SM6.2-compatible idiom -- a bound, sized-at-creation-time table with
// an unbounded array declared in the shader, Sec5.1), never N root
// parameters.
// ===========================================================================

namespace {

// Binds `array_pointers[0..n_arrays)` (each `array_element_counts[i]` int8
// elements) through one descriptor table and reads every one back into
// `*out_widened` (one int32 per element, in array order) -- the shared
// mechanism `BindDescriptorTableAndReadback` and
// `DescriptorHeapRegionIsCleanAfterHandleRelease` both drive.
bool RunDescriptorTableBind(const int8_t* const* array_pointers, const size_t* array_element_counts,
                             size_t n_arrays, std::vector<int32_t>* out_widened) {
	harness::Device& dev = GetDevice();
	if (!dev.available || n_arrays == 0) return false;

	std::vector<uint64_t> offsets(n_arrays);
	uint64_t total_elements = 0;
	for (size_t i = 0; i < n_arrays; ++i) {
		offsets[i] = total_elements;
		total_elements += array_element_counts[i];
	}
	if (total_elements == 0) return false;

	// Per-array data buffers (upload heap, GENERIC_READ -- valid directly as
	// an SRV source, no default-heap copy needed for this structural test),
	// each padded to a 4-byte multiple for a RAW buffer view's own NumElements.
	std::vector<Microsoft::WRL::ComPtr<ID3D12Resource>> array_bufs(n_arrays);
	for (size_t i = 0; i < n_arrays; ++i) {
		const size_t padded = ((array_element_counts[i] + 3) / 4) * 4;
		std::vector<uint8_t> bytes(padded, 0);
		std::memcpy(bytes.data(), array_pointers[i], array_element_counts[i]);
		array_bufs[i] = dev.Upload(bytes.data(), padded);
	}

	D3D12_DESCRIPTOR_HEAP_DESC hd{};
	hd.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
	hd.NumDescriptors = static_cast<UINT>(n_arrays);
	hd.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
	Microsoft::WRL::ComPtr<ID3D12DescriptorHeap> heap;
	SSLM_GPU_HR(dev.dev->CreateDescriptorHeap(&hd, IID_PPV_ARGS(&heap)));
	const UINT stride =
	    dev.dev->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
	D3D12_CPU_DESCRIPTOR_HANDLE cpu_start = heap->GetCPUDescriptorHandleForHeapStart();
	for (size_t i = 0; i < n_arrays; ++i) {
		const size_t padded = ((array_element_counts[i] + 3) / 4) * 4;
		D3D12_SHADER_RESOURCE_VIEW_DESC srv{};
		srv.Format = DXGI_FORMAT_R32_TYPELESS;
		srv.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
		srv.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
		srv.Buffer.FirstElement = 0;
		srv.Buffer.NumElements = static_cast<UINT>(padded / 4);
		srv.Buffer.Flags = D3D12_BUFFER_SRV_FLAG_RAW;
		D3D12_CPU_DESCRIPTOR_HANDLE h{cpu_start.ptr + i * stride};
		dev.dev->CreateShaderResourceView(array_bufs[i].Get(), &srv, h);
	}

	std::vector<int64_t> counts64(n_arrays), offsets64(n_arrays);
	for (size_t i = 0; i < n_arrays; ++i) {
		counts64[i] = static_cast<int64_t>(array_element_counts[i]);
		offsets64[i] = static_cast<int64_t>(offsets[i]);
	}
	auto counts_buf = dev.Upload(counts64.data(), counts64.size() * 8);
	auto offsets_buf = dev.Upload(offsets64.data(), offsets64.size() * 8);
	auto out_uav = dev.MakeBuffer(total_elements * 4, D3D12_HEAP_TYPE_DEFAULT,
	                               D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
	                               D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
	auto readback = dev.MakeBuffer(total_elements * 4, D3D12_HEAP_TYPE_READBACK,
	                                D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST);

	static Microsoft::WRL::ComPtr<ID3D12RootSignature> s_root_sig;
	static Microsoft::WRL::ComPtr<ID3D12PipelineState> s_pso;
	if (!s_root_sig) {
		s_root_sig = dev.MakeRootSigDescriptorTable();
		auto cso = harness::ReadFile(harness::ShaderPath("descriptor_table_readback"));
		s_pso = dev.MakePSO(s_root_sig.Get(), cso);
	}

	SSLM_GPU_HR(dev.alloc->Reset());
	SSLM_GPU_HR(dev.list->Reset(dev.alloc.Get(), s_pso.Get()));
	ID3D12DescriptorHeap* heaps[] = {heap.Get()};
	dev.list->SetDescriptorHeaps(1, heaps);
	dev.list->SetComputeRootSignature(s_root_sig.Get());
	const uint32_t n_arrays_u32 = static_cast<uint32_t>(n_arrays);
	dev.list->SetComputeRoot32BitConstants(0, 1, &n_arrays_u32, 0);
	dev.list->SetComputeRootShaderResourceView(1, counts_buf->GetGPUVirtualAddress());
	dev.list->SetComputeRootShaderResourceView(2, offsets_buf->GetGPUVirtualAddress());
	dev.list->SetComputeRootDescriptorTable(3, heap->GetGPUDescriptorHandleForHeapStart());
	dev.list->SetComputeRootUnorderedAccessView(4, out_uav->GetGPUVirtualAddress());
	dev.list->Dispatch(static_cast<UINT>((n_arrays + 63) / 64), 1, 1);
	D3D12_RESOURCE_BARRIER b{};
	b.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
	b.Transition.pResource = out_uav.Get();
	b.Transition.StateBefore = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
	b.Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_SOURCE;
	b.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
	dev.list->ResourceBarrier(1, &b);
	dev.list->CopyResource(readback.Get(), out_uav.Get());
	SSLM_GPU_HR(dev.list->Close());
	ID3D12CommandList* lists[] = {dev.list.Get()};
	dev.queue->ExecuteCommandLists(1, lists);
	SSLM_GPU_HR(dev.queue->Signal(dev.fence.Get(), ++dev.fence_val));
	if (dev.fence->GetCompletedValue() < dev.fence_val) {
		SSLM_GPU_HR(dev.fence->SetEventOnCompletion(dev.fence_val, dev.fence_event));
		WaitForSingleObject(dev.fence_event, INFINITE);
	}

	out_widened->resize(total_elements);
	void* p = nullptr;
	D3D12_RANGE range{0, static_cast<SIZE_T>(total_elements * 4)};
	SSLM_GPU_HR(readback->Map(0, &range, &p));
	std::memcpy(out_widened->data(), p, total_elements * 4);
	D3D12_RANGE none{0, 0};
	readback->Unmap(0, &none);
	return true;
}

}  // namespace

bool BindDescriptorTableAndReadback(const int8_t* const* array_pointers,
                                     const size_t* array_element_counts, size_t n_arrays,
                                     int8_t* out_readback_concat, size_t out_readback_capacity) {
	std::vector<int32_t> widened;
	if (!RunDescriptorTableBind(array_pointers, array_element_counts, n_arrays, &widened)) {
		return false;
	}
	if (widened.size() > out_readback_capacity) return false;
	for (size_t i = 0; i < widened.size(); ++i) {
		out_readback_concat[i] = static_cast<int8_t>(widened[i]);
	}
	return true;
}

bool DescriptorHeapRegionIsCleanAfterHandleRelease() {
	// Sec10 dim 1, Sec14 Fold G1's own sequential-release-then-remap shape,
	// realized on this design's own binding substrate: map a first "handle"
	// (one descriptor-table bind), release every device object it created
	// (ComPtr scope exit below), then map a second, independently-created
	// handle with DIFFERENT content and confirm it reads back only its own
	// pattern -- a real property of this session's device/heap lifecycle,
	// not a tautology (a bug that reused a stale GPU virtual address, or
	// failed to fence-wait before releasing, would show up here).
	const int8_t first_pattern[8] = {1, 2, 3, 4, 5, 6, 7, 8};
	const int8_t* first_ptrs[1] = {first_pattern};
	const size_t first_counts[1] = {8};
	{
		std::vector<int32_t> widened;
		if (!RunDescriptorTableBind(first_ptrs, first_counts, 1, &widened)) return false;
		for (size_t i = 0; i < 8; ++i) {
			if (widened[i] != first_pattern[i]) return false;
		}
	}  // every device object from the first bind releases here

	const int8_t second_pattern[8] = {-1, -2, -3, -4, -5, -6, -7, -8};
	const int8_t* second_ptrs[1] = {second_pattern};
	const size_t second_counts[1] = {8};
	std::vector<int32_t> widened2;
	if (!RunDescriptorTableBind(second_ptrs, second_counts, 1, &widened2)) return false;
	for (size_t i = 0; i < 8; ++i) {
		if (widened2[i] != second_pattern[i]) return false;  // residue from the first bind
	}
	return true;
}

namespace {
bool g_tier_mock_armed = false;
int g_tier_mock_value = 3;
}  // namespace

void ArmResourceBindingTierMock(int tier) {
	g_tier_mock_armed = true;
	g_tier_mock_value = tier;
}
void ClearResourceBindingTierMock() { g_tier_mock_armed = false; }

bool MapModelGpuResidencyTierCheck() {
	// Sec11 B3, D-SLM3082 (Dan's amendment 5): a preflight run before any
	// descriptor-table/UAV-table build or device allocation is attempted --
	// this design's binding architecture requires Tier 3 (D-SLM3000).
	int tier;
	if (g_tier_mock_armed) {
		tier = g_tier_mock_value;
	} else {
		tier = static_cast<int>(GetDevice().QueryResourceBindingTier());
	}
	return tier >= static_cast<int>(D3D12_RESOURCE_BINDING_TIER_3);
}

// ===========================================================================
// B12 (Sec11 B12): deterministic allocation-failure injection. T-2049 (N4,
// Claude/Poirot/34ef30f-gpu-serial-port-confirmation-review.md, correcting
// T-2045's own S1): `N` is honestly the composed pipeline's CURRENT binding
// count -- the confirmation review found T-2045's own "§5.1's own real
// read+write resource-table count" claim wrong: what was counted is
// `MakeRootSigComposed`'s 8 SRVs + 4 UAVs, the root-descriptor substrate
// §5.1 explicitly rejects, not §5.1's own ratified architecture (one SRV
// descriptor table + one UAV descriptor table, D-SLM3001/D-SLM2929) or B12's
// own named-allocand enumeration (the weight-buffer SRVs, the per-layer
// CarriedScale CBV, the descriptor heap, the per-dispatch status word, the
// sequence-level sticky word, the residual-stream buffer and its per-layer
// scratch, the K/V cache, the split-word saturation accumulator) -- a
// different list whose items do not map one-to-one onto what this build
// actually allocates (the descriptor heap and the per-dispatch status words
// do not exist yet; several of the 12 below are not named in that list).
// Deriving a specific integer from that prose without inventing an
// interpretation nobody has ratified would be the same "presented as read
// off the design when it is read off something else" failure this finding
// exists to close, just with a different wrong source. What IS derivable
// exactly, mechanically, and re-derivable automatically if it ever changes:
// `harness::kComposedResourceBindingCount` (d3d12_harness.h), the single
// constant `MakeRootSigComposed` itself is built from. `N` below reads that
// constant directly rather than repeating its value -- not a runtime
// comparison of two independently-maintained numbers (which could still
// drift), but the SAME symbol, so drift is structurally impossible rather
// than merely caught. This value must be re-derived, not reused, the moment
// S2's own migration to §5.1's literal descriptor-table architecture lands.
// ===========================================================================

extern const uint32_t kGpuResidencyAllocationCallCount = harness::Device::kComposedResourceBindingCount;

namespace {
bool g_low_budget_mock_armed = false;
uint64_t g_low_budget_mock_bytes = 0;
bool g_alloc_fail_injection_armed = false;
uint32_t g_alloc_fail_injection_index = 0;
uint32_t g_allocation_calls_attempted = 0;
// T-2045 (S1): a real per-index tracking set -- `MapModelGpuResidencyWithInjection`
// below appends one entry per successful mock allocation and, on an injected
// failure, ERASES every entry it appended (a real cleanup loop), so
// `LiveAllocationCount()` reads the vector's own observed size rather than a
// literal `= 0` assignment standing in for cleanup that did not run. This is
// a mock allocator (Sec21 Fold D-SLM3081's own "not real VRAM exhaustion,
// which is non-deterministic and therefore not a valid gate" ruling) -- no
// real ID3D12Resource is created per index -- but the bookkeeping it
// performs is real: an entry that was never appended cannot be "cleaned up"
// by assignment, only by having never existed or by being erased.
std::vector<uint32_t> g_live_allocations;
}  // namespace

void ArmAllocationFailureInjection(uint32_t index) {
	g_alloc_fail_injection_armed = true;
	g_alloc_fail_injection_index = index;
}
void ArmLowBudgetInjection(uint64_t mocked_budget_bytes) {
	g_low_budget_mock_armed = true;
	g_low_budget_mock_bytes = mocked_budget_bytes;
}
void ClearAllocationInjection() {
	g_alloc_fail_injection_armed = false;
	g_low_budget_mock_armed = false;
}
uint32_t LiveAllocationCount() { return static_cast<uint32_t>(g_live_allocations.size()); }
uint32_t AllocationCallsAttempted() { return g_allocation_calls_attempted; }

bool MapModelGpuResidencyWithInjection(uint64_t required_bytes) {
	// D-SLM3081 item 4's own two-part construction: (1) a mocked-low-budget
	// preflight that fails before ANY of the N allocation calls is attempted,
	// (2) deterministic per-index allocation-failure injection with
	// transactional cleanup (every allocation made before the injected index
	// is released, mock live-count reads 0) -- neither depends on the test
	// device's own real VRAM.
	g_allocation_calls_attempted = 0;
	g_live_allocations.clear();
	if (g_low_budget_mock_armed && required_bytes > g_low_budget_mock_bytes) {
		return false;  // preflight fails cheaply -- 0 allocation calls attempted, by construction above
	}
	const uint32_t N = kGpuResidencyAllocationCallCount;
	for (uint32_t i = 0; i < N; ++i) {
		g_allocation_calls_attempted = i + 1;
		if (g_alloc_fail_injection_armed && i == g_alloc_fail_injection_index) {
			// Transactional cleanup: erase every allocation this call itself
			// appended, one at a time -- an OBSERVED empty vector afterward,
			// not an assumed one.
			while (!g_live_allocations.empty()) g_live_allocations.pop_back();
			return false;
		}
		g_live_allocations.push_back(i);  // this allocation call succeeded; held live until the map completes
	}
	return true;  // every allocation call succeeded (including the N==0 vacuous case)
}

}  // namespace superslm_gpu
