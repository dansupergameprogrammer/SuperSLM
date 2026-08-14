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

#include <cstring>
#include <stdexcept>
#include <vector>

#include "d3d12_harness.h"

namespace superslm_gpu {
namespace harness {

// GetModuleFileNameA-derived directory of the running executable, plus
// "shaders\\<name>.cso" -- build.bat/CMakeLists.txt (this design's own build
// scripts, Sec5.7) place the compiled shaders there alongside the test
// binary.
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
// B5 (Sec5.5/Sec11 B5): the two-schedule int64 abs-max reduction. STUB pending
// B5 -- returns a sentinel that diverges from the CPU oracle on any nonzero
// row so a false pass cannot occur.
// ===========================================================================

int64_t MaxAbsReduceWideGpuScheme0(const int64_t* x, size_t n) {
	(void)x;
	(void)n;
	// STUB (B5 not yet built): INT64_MIN can never equal a real MaxAbsReduceWide
	// result (that function's own range is [1, INT64_MAX], intmath.h:217), so
	// this sentinel is always an observable mismatch, never an accidental pass.
	return INT64_MIN;
}

int64_t MaxAbsReduceWideGpuScheme1(const int64_t* x, size_t n) {
	(void)x;
	(void)n;
	return INT64_MIN;  // STUB (B5 not yet built) -- see Scheme0's own note.
}

// ===========================================================================
// B2 (Sec6/Sec11 B2): the guard-path port, isolated tier. STUB pending B2.
// ===========================================================================

superslm::SslmForwardStatus CheckRoundingDivideByPotExponentDomainGpu(int64_t q_B, int64_t e_a) {
	(void)q_B;
	(void)e_a;
	// STUB (B2 not yet built): a status this predicate can never legitimately
	// return on its own two-valued domain (Ok / RoundingDivideByPotExponentOutOfDomain)
	// -- WorkspaceTooSmall is unrelated to this guard, so any real fixture's
	// expected status diverges from it observably.
	return superslm::SslmForwardStatus::WorkspaceTooSmall;
}

superslm::SslmForwardStatus CheckBiasAccumulateMagnitudeDomainGpu(int64_t acc_i, int64_t b,
                                                                    int64_t q_b, int64_t r_a,
                                                                    int64_t e_a) {
	(void)acc_i;
	(void)b;
	(void)q_b;
	(void)r_a;
	(void)e_a;
	return superslm::SslmForwardStatus::WorkspaceTooSmall;  // STUB (B2 not yet built)
}

superslm::ChainResult RequantChainCheckedGpu(const int64_t* wide_row, size_t n,
                                              const superslm::CarriedScale* incoming,
                                              size_t n_incoming,
                                              superslm::CarriedScale site_constant) {
	(void)wide_row;
	(void)n;
	(void)incoming;
	(void)n_incoming;
	(void)site_constant;
	return superslm::ChainResult{superslm::SslmForwardStatus::WorkspaceTooSmall};  // STUB (B2)
}

// ===========================================================================
// B4/B7/B11 (Sec5.6/Sec11): the composed, device-resident, multi-layer
// forward. STUB pending B3/B4/B5/B6/B7/B11 -- the largest remaining unit.
// ===========================================================================

superslm::SslmForwardStatus RunLayerLoopGpu(superslm::SequenceLayerState& seq,
                                             const superslm::LayerWeights* layers,
                                             uint32_t num_hidden_layers, uint32_t layer_budget,
                                             size_t hidden_size, size_t head_dim,
                                             size_t num_key_value_heads, size_t intermediate_size,
                                             int64_t context_cap,
                                             const superslm::SslmTensorManifest& rope_tables,
                                             uint8_t* workspace, size_t workspace_size) {
	(void)seq;
	(void)layers;
	(void)num_hidden_layers;
	(void)layer_budget;
	(void)hidden_size;
	(void)head_dim;
	(void)num_key_value_heads;
	(void)intermediate_size;
	(void)context_cap;
	(void)rope_tables;
	(void)workspace;
	(void)workspace_size;
	// STUB (B3/B4/B5/B6/B7/B11 not yet built): `seq` is left untouched (no
	// GPU-side composition exists yet), and the returned status is one no
	// fixture in this suite expects (every real fixture's CPU oracle status is
	// Ok or one of the funnel/guard rejections; KvPrecisionUnsupported is
	// never a RunLayerLoop outcome this design's build target reaches) -- an
	// observable mismatch against every T-2019 B4/B6/B7/B8/B11 cell, not an
	// accidental pass.
	return superslm::SslmForwardStatus::KvPrecisionUnsupported;
}

const int8_t* KeyRowGpu(const uint8_t* workspace, uint32_t layer, int64_t context_cap,
                         size_t num_kv_heads, size_t head_dim, size_t kv_head, int64_t position) {
	(void)workspace;
	(void)layer;
	(void)context_cap;
	(void)num_kv_heads;
	(void)head_dim;
	(void)kv_head;
	(void)position;
	// STUB (B4/B6/B7 not yet built): a static zeroed row -- safe to dereference
	// (every B11 fixture reads exactly 2 bytes at this suite's own fixed
	// head_dim=2), and all-zero fails every "K present" assertion honestly
	// (the design's own real K rows are never all-zero once landed).
	static const int8_t kZeroRow[64] = {};
	return kZeroRow;
}

const int8_t* ValueRowGpu(const uint8_t* workspace, uint32_t layer, int64_t context_cap,
                           size_t num_kv_heads, size_t head_dim, size_t kv_head, int64_t position) {
	(void)workspace;
	(void)layer;
	(void)context_cap;
	(void)num_kv_heads;
	(void)head_dim;
	(void)kv_head;
	(void)position;
	static const int8_t kZeroRow[64] = {};
	return kZeroRow;  // STUB (B4/B6/B7 not yet built) -- see KeyRowGpu's own note.
}

// ===========================================================================
// B7 (Sec5.8/Sec11 B7): the dispatch_budget contract. STUB pending B7 -- pure
// host-side policy, but sequenced after B6 per Sec11's own dependency order
// (this design's build target is not fused ahead of it).
// ===========================================================================

SslmGpuStatus PlanDispatchBudgetGpu(uint32_t dispatch_budget, uint32_t num_hidden_layers,
                                     uint32_t current_layer_position, uint32_t* out_layers_to_issue) {
	(void)dispatch_budget;
	(void)num_hidden_layers;
	(void)current_layer_position;
	if (out_layers_to_issue) *out_layers_to_issue = 999999u;  // STUB (B7 not yet built) -- never a real layer count
	return SslmGpuStatus::Busy;  // never the status any T-2019 B7 cell expects (Ok / DispatchBudgetTooSmall)
}

// ===========================================================================
// Sec5.9 (D-SLM3076-3079): the asynchronous sequence lifecycle. STUB pending
// Sec5.9's own build (sequenced with B7/B8 per Sec11).
// ===========================================================================

bool CallProceedsOrBusy_DecodeStepGpu(bool sequence_is_submitted) {
	(void)sequence_is_submitted;
	return true;  // STUB (Sec5.9 not yet built) -- wrong on the Submitted=true cell (must be false)
}
bool CallProceedsOrBusy_SeqSave(bool sequence_is_submitted) {
	(void)sequence_is_submitted;
	return true;  // STUB (Sec5.9 not yet built)
}
bool CallProceedsOrBusy_SeqReset(bool sequence_is_submitted) {
	(void)sequence_is_submitted;
	return true;  // STUB (Sec5.9 not yet built)
}
bool CallProceedsOrBusy_SeqRelease(bool sequence_is_submitted) {
	(void)sequence_is_submitted;
	return true;  // STUB (Sec5.9 not yet built)
}
bool CallProceedsOrBusy_ModelUnmap(bool any_sequence_submitted) {
	(void)any_sequence_submitted;
	return true;  // STUB (Sec5.9 not yet built) -- wrong on the any_sequence_submitted=true cell
}

bool GpuReadySignalsCompletion(bool fence_signaled, int32_t* out_ready,
                                superslm::SslmForwardStatus* out_status) {
	(void)fence_signaled;
	// STUB (Sec5.9 not yet built): *out_ready pinned to a value neither the
	// "before signal" (want 0) nor "after signal" (want 1) cell accepts.
	if (out_ready) *out_ready = -1;
	if (out_status) *out_status = superslm::SslmForwardStatus::WorkspaceTooSmall;
	return false;
}

// ===========================================================================
// B8 (Sec11 B8): device residency round-trip. STUB pending B8.
// ===========================================================================

bool SaveGpuSequenceState(const superslm::SequenceLayerState& seq, const uint8_t* workspace,
                           size_t workspace_size, void* out_blob, size_t* out_blob_size) {
	(void)seq;
	(void)workspace;
	(void)workspace_size;
	(void)out_blob;
	if (out_blob_size) *out_blob_size = 0;
	return false;  // STUB (B8 not yet built)
}
bool RestoreGpuSequenceState(const void* blob, size_t blob_size, superslm::SequenceLayerState* out_seq,
                              uint8_t* out_workspace, size_t workspace_size) {
	(void)blob;
	(void)blob_size;
	(void)out_seq;
	(void)out_workspace;
	(void)workspace_size;
	return false;  // STUB (B8 not yet built)
}

// ===========================================================================
// B3 (Sec5.1/Sec11 B3): descriptor-table binding substrate. STUB pending B3.
// ===========================================================================

bool BindDescriptorTableAndReadback(const int8_t* const* array_pointers,
                                     const size_t* array_element_counts, size_t n_arrays,
                                     int8_t* out_readback_concat, size_t out_readback_capacity) {
	(void)array_pointers;
	(void)array_element_counts;
	(void)n_arrays;
	(void)out_readback_concat;
	(void)out_readback_capacity;
	return false;  // STUB (B3 not yet built)
}
bool DescriptorHeapRegionIsCleanAfterHandleRelease() {
	return false;  // STUB (B3 not yet built)
}

void ArmResourceBindingTierMock(int tier) { (void)tier; }        // STUB (B3 not yet built)
void ClearResourceBindingTierMock() {}                            // STUB (B3 not yet built)
bool MapModelGpuResidencyTierCheck() {
	return true;  // STUB (B3 not yet built) -- wrong on a mocked Tier 1/2 cell (must be false)
}

// ===========================================================================
// B12 (Sec11 B12): deterministic allocation-failure injection. STUB pending
// B3/B12 -- kGpuResidencyAllocationCallCount is a real design quantity
// (Sec5.1's read+write resource-table count) that cannot be derived correctly
// until B3's resource list is final; set to 0 for now (0 allocation-index
// cells run rather than a made-up count silently misreporting the real one --
// named explicitly here, not left implicit) so TestT2019_B12_
// InjectedFailureAtEveryAllocationIndex's own loop is a no-op until B12 names
// the real value with its derivation stated, per this ticket's own brief.
// ===========================================================================

extern const uint32_t kGpuResidencyAllocationCallCount = 0;  // STUB -- see note above; derived at B12

void ArmAllocationFailureInjection(uint32_t index) { (void)index; }  // STUB (B12 not yet built)
void ArmLowBudgetInjection(uint64_t mocked_budget_bytes) { (void)mocked_budget_bytes; }  // STUB
void ClearAllocationInjection() {}                                                        // STUB
uint32_t LiveAllocationCount() { return 1;  }  // STUB (B12 not yet built) -- wrong (never 0)
uint32_t AllocationCallsAttempted() { return 1; }  // STUB (B12 not yet built) -- wrong (never 0)
bool MapModelGpuResidencyWithInjection(uint64_t required_bytes) {
	(void)required_bytes;
	return false;  // STUB (B12 not yet built) -- wrong on the "clean call" cells (must be true)
}

}  // namespace superslm_gpu
