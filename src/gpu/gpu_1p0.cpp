// T-2113 (B1/B2/B3): production implementation of the 1.0 GPU API's context, model,
// and sequence lifecycles (Claude/Vitruvius/t2107-gpu-core-1p0-design-2026-08-14.md
// Sec4.1.1/Sec5.1/Sec5.3, Sec10 B1/B2/B3).
//
// B1's own gate (design Sec10): "existing suite green, unchanged behavior when
// exactly one context is created (a same-process, same-thread equivalence check
// against the substrate's own current single-call behavior)". Concretely: this file
// adds a NEW, additive construction (SslmGpuContext, its own owned harness::Device
// and PSO/root-sig caches) that touches none of the pre-1.0 substrate's own process-
// global statics (`harness::GetDevice()`, `harness::GetOrBuildPipeline`,
// `harness::GetOrBuildComposedPipeline`, all still defined in d3d12_harness.h and
// still used unchanged by `RunLayerLoopGpu` and every other pre-1.0 entry point in
// superslm_gpu.cpp) -- so the existing test suite (tests\test_main.cpp,
// out\superslm_tests.exe) observes zero behavior change. The migration that actually
// removes the pre-1.0 singleton usage from RunLayerLoopGpu's own dispatch-recording
// path is B2/B4's job (design Sec10: B2 folds the T-2105 residency construction onto
// the model handle's own upload path; B4 ports the dispatch geometry onto the new
// handle shapes) -- B1 only builds the object the later sections will route weights,
// PSOs, and dispatches through instead of the singleton, per D-SLM3294's own
// "no process-global device state in the 1.0 backend" requirement (design Sec1).
//
// Symbols DEFINED here: sslm_gpu_context_create, sslm_gpu_context_destroy.
// Every other declaration in include/superslm/gpu_1p0.h stays declared-only until
// its own B-section lands (B2 through B9); this file is the single production TU
// the whole 1.0 API grows in, one B-section at a time -- calling out its own
// section boundary in comments as later sections extend it, mirroring
// src/gpu/superslm_gpu.cpp's own established convention for the pre-1.0 surface.

#include "superslm/gpu_1p0.h"
#include "../forward/parallel_split.h"  // ParallelForHookValid (the hook setters' shared validation)
#include "superslm/gpu_1p0_bench_bridge.h"

// Keep the implementation readable while the installed API exposes scoped status members.
using enum SslmGpuStatus;

// GpuContextConfig's layout is ABI: the suite that links against this file compiles its cells
// against a mirror of the header and asserts the same layout there, so a drift on either side
// fails the compile rather than passing a short struct by value.
static_assert(sizeof(GpuContextConfig) == 16 && offsetof(GpuContextConfig, shader_dir) == 8,
              "GpuContextConfig layout (x64): 16 bytes, shader_dir at offset 8");
// T-2851 (design Sec4.9): GpuResidencyConfig kept its size, offset and alignment when `int reserved`
// became `uint32_t flags`. Design Sec4.9 puts the same assertion on the t2112 suite's mirror side
// (fixture_common.h), which is the test suite's to carry.
static_assert(sizeof(GpuResidencyConfig) == 4 && offsetof(GpuResidencyConfig, flags) == 0,
              "GpuResidencyConfig layout: 4 bytes, flags at offset 0");

#include <algorithm>
#include <array>
#include <cassert>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <map>
#include <memory>
#include <mutex>
#include <new>
#include <stdexcept>
#include <string>
#include <vector>

#include "d3d12_harness.h"
#include "../kv_pages.h"           // paged KV (G3): the page allocator, one module for both backends
#include "gpu_kv_pool_geometry.h"  // paged KV (G3): the GPU pool's host-only arithmetic
#include "superslm/adapter_marshal.h"  // T-2113 B6: LoadAdapterArtifact (relocated, tools/sslm_adapter_loader.h)
#include "superslm/gpu_1p0_g5_bridge.h"  // G5-5 (T-2132): the build-seat-owned schema/parity bridge
#include "superslm/gpu_port.h"       // T-2113 B2: GpuLayerLayout/ComputeLayerLayout/PackLayerWeightsBytes
#include "superslm/layer_marshal.h"  // T-2113 B2: MarshalLayer/LayerBacking (relocated, tools/sslm_marshal.h)
#include "superslm/model.h"          // T-2113 B2: the REAL superslm::SslmModelView this TU needs by value
#include "../spx1_format.h"          // precached prefixes: the SPX1 writer and reader both backends use
#include "superslm/schema_masks.h"   // G5-5 (T-2132): SchemaMasksTable -- the SAME SCM1 reader the CPU
                                      // ABI (sslm_abi.cpp) already uses, reused here, not re-derived.

#if defined(SUPERSLM_ENABLE_GPU_CHUNK_DISPATCH_INSTRUMENT)
namespace superslm_test {
extern std::atomic<int64_t> g_gpu_ready_poll_count_probe;
}
#endif

// The real SslmGpuContext this handle type opaquely names to every 1.0 API caller.
// Owns everything B1's own gate requires be OWNED rather than reached through a
// process-global name: its own harness::Device (never harness::GetDevice()'s
// static), and its own PSO/root-signature caches, keyed exactly the way the
// pre-1.0 singleton caches were (by shader base name) so the later B-sections that
// port dispatch-recording code onto this object can reuse that code's own lookup
// shape unchanged -- only the cache's OWNER moves.
// T-2113 (B8, design Sec5.4/Sec10 B8): thread-safety contract, stated here because this is
// where every thread-shared object this design names actually lives.
//
//   - The PSO/root-signature caches above and `device` are immutable-after-populate for the
//     RESIDENCY half of this object's life (map/create time) and are never touched by the
//     decode dispatch path at all (see the next bullet) -- reading them concurrently is safe.
//   - `SslmGpuModelHandle`/`SslmGpuAdapterHandle`: read-only for the whole of their lifetime
//     after `map` returns (design Sec5.4) -- safe for any number of threads to read (bind,
//     dispatch against, or call `sslm_gpu_seq_embed_token` through) concurrently, by
//     construction, since nothing ever writes to resident weight/adapter/embed buffers again.
//   - `SslmGpuSequenceHandle`: owns state private to itself. Two threads each driving a
//     DIFFERENT sequence handle through `sslm_decode_step_gpu`/`sslm_gpu_ready`/
//     `sslm_gpu_seq_embed_token` touch disjoint per-sequence memory -- this is "thread-safe
//     execution over disjoint sequences" (D-SLM3294), true by construction once the handle's
//     own ownership split holds. Two threads driving the SAME sequence handle concurrently is
//     a caller error this design does not guard against structurally (proven, not assumed: the
//     T-2112 suite's own dim3 M2 cell reproduces a real D3D12 device fault doing exactly this,
//     `Claude/Brunel/t2113-1p0-core-build-2026-08-15.md` Sec11.4) -- named as a residual, never
//     silently treated as safe.
//   - **The one thing this design deliberately does NOT make safe without caller help**: the
//     actual GPU submission every `sslm_decode_step_gpu`/`_batch_gpu`/`sslm_gpu_ready(block=1)`
//     call performs. Design Sec5.4: "the command queue and descriptor-heap allocator are not
//     safe to submit-from or allocate-from concurrently without external synchronization; this
//     design does not build an internal queue-level lock (Sec13's own deferral: 'speculative
//     concurrency infrastructure the product does not yet need') -- a caller driving two
//     threads' decode calls through one context's queue concurrently must serialize the submit
//     call itself." **Grounded, not merely quoted from the design**: as of B5-B7, the decode
//     dispatch path (`RunLayerLoopGpuSubmit`/`RunLayerLoopGpuFinish`, superslm_gpu.cpp) does not
//     even route through THIS context's own `device` above -- it still calls the pre-1.0
//     substrate's process-WIDE `harness::GetDevice()` singleton (`SubmitOneSequenceDecode`'s own
//     `(void)ctx;` -- ctx is not read for dispatch purposes at all, only for residency lookups
//     upstream of it). So the serialization requirement is, today, PROCESS-wide, not merely
//     per-context: any two threads anywhere in the process that call a decode-touching function
//     concurrently, on any context, must be externally serialized at that call boundary, or the
//     shared allocator's `Reset()` races exactly the way the undrained-batch hang (D-SLM3384,
//     `Claude/Brunel/t2113-1p0-core-build-2026-08-15.md` Sec12.2) already proved by execution --
//     that hang is this same hazard's sequential-but-undrained shape; two genuinely concurrent
//     threads racing the identical `Reset()` is the stronger, race-condition form of it. B8's own
//     bench tool (`tools/t2113_b8_thread_smoke.cpp`) is built to this documented discipline: a
//     mutex external to this API wraps exactly the decode-submit-and-drain call pair, nothing
//     more -- proving the design's own "disjoint sequences, externally serialized only at the
//     queue-submit boundary" claim holds, without this file adding the internal lock the design
//     explicitly declines to build (Sec13's own deferral row, dated, routed: "if evidence arrives
//     that a caller needs concurrent submission through one shared queue... add the lock... as
//     its own scoped follow-on, not retrofitted here"). Migrating the decode dispatch path onto
//     this context's OWN `device` (making the requirement genuinely per-context rather than
//     process-wide) is out of B8's own scope -- named here, dated, the same class of routed item
//     `g_resident_rope`'s own retirement note already established (Sec6.3 of the build log).

// T-2243 review finding 2 (D-SLM4113): a process-global in-flight submission count, scoped to
// what the guarded resource actually is. The comment block above already establishes that the
// decode dispatch path does not route through any per-context `device` -- it calls the pre-1.0
// substrate's process-WIDE `harness::GetDevice()` singleton, with ONE command allocator and ONE
// command list shared by every context and every model in the process. `sslm_gpu_seq_restore`'s
// own guard against that singleton's `dev.alloc->Reset()`/`dev.list->Reset(...)` therefore cannot
// be scoped to a single model's `submitted_sequences` (a sibling in flight on a DIFFERENT model,
// or on a different context, reaches the identical Reset() against an executing allocator) -- it
// has to be scoped to the resource, which is process-global. Incremented/decremented at exactly
// the four sites that already maintain the per-model/per-adapter counts this mirrors:
// SubmitOneSequenceDecode's submit, sslm_gpu_ready's drain, and SubmittedWindowScopeGuard's
// construct/destruct pair (the G5 bridge's own submit/drain).
//
// Paged-KV plan §4.4 (rev 11), step G3: this count, the pooled count beside it, and each model's
// and adapter's `submitted_sequences` are private, written only by the counting accessor
// SetGpuSeqState (below) and read through these readers. g_process_wide_submitted_sequences is
// now GpuSubmittedCounts::Process(); the pooled count (Submitted POOLED handles) is
// GpuSubmittedCounts::Pooled(), the guard's pooled scope.
struct SslmGpuSequenceHandle;
struct SslmGpuAdapterHandle;
void SetGpuSeqState(SslmGpuSequenceHandle* seq, superslm_gpu::SslmSequenceGpuState next,
                    const SslmGpuAdapterHandle* adapter);
class GpuSubmittedCounts {
public:
	// Every Submitted handle in the process (the guard's "any" scope).
	static int64_t Process() noexcept { return process_.load(); }
	// Every Submitted pooled handle in the process (the guard's "pooled" scope).
	static int64_t Pooled() noexcept { return pooled_.load(); }

private:
	friend void SetGpuSeqState(SslmGpuSequenceHandle*, superslm_gpu::SslmSequenceGpuState,
	                           const SslmGpuAdapterHandle*);
	static inline std::atomic<int64_t> process_{0};
	static inline std::atomic<int64_t> pooled_{0};
};

// Paged-KV plan §4.4 (rev 6, rev 8, rev 9), step G3: the entry guard every "guarded" row of §4.4's
// derived table runs, in its `…Impl` body, after the call-level refusals that change nothing and
// before its first state change (*Rule*, *Status precedence*). Its terms:
//   - the scope's count: kAny, every Submitted handle in the process (every guarded arc verb, and
//     every existing entry point called on a pooled holder); kPooled, every Submitted pooled handle
//     (every existing entry point called on a dedicated sequence, a model or an adapter);
//   - the ledger term, for a row whose graph reaches the singleton (Allocator "singleton" or
//     "both"): the singleton's ledger read through harness::SingletonLedgerIfConstructed(), which
//     never constructs the device; stranded is checked first and returns SSLM_DEVICE_LOST
//     (terminal, decision 73), unobserved returns SSLM_BUSY.
// Returns SSLM_OK when the call may proceed, having recorded for the boundary guard check that a
// guard ran in this outermost call (inflight::NoteEntrySubmitGuardRan); otherwise the status the
// entry returns, changing nothing.
enum class GuardScope { kAny, kPooled };
enum class GuardDevice { kPerContextOnly, kReachesSingleton };
SslmGpuStatus EntrySubmitGuard(GuardScope scope, GuardDevice device) noexcept {
	if (device == GuardDevice::kReachesSingleton) {
		if (const superslm_gpu::harness::SubmissionLedger* ledger =
		        superslm_gpu::harness::SingletonLedgerIfConstructed()) {
			switch (ledger->ReadGuardTerm()) {
				case superslm_gpu::harness::SubmissionLedger::GuardTerm::Stranded:
					return SSLM_DEVICE_LOST;
				case superslm_gpu::harness::SubmissionLedger::GuardTerm::Unobserved:
					return SSLM_BUSY;
				case superslm_gpu::harness::SubmissionLedger::GuardTerm::Observed:
					break;
			}
		}
	}
	const int64_t in_scope =
	    scope == GuardScope::kAny ? GpuSubmittedCounts::Process() : GpuSubmittedCounts::Pooled();
	if (in_scope > 0) return SSLM_BUSY;
	superslm_gpu::inflight::NoteEntrySubmitGuardRan();
	return SSLM_OK;
}
// The scope an existing entry point uses for its holder: kAny on a pooled holder, kPooled on a
// dedicated one (§4.4, *Scope*).
template <class Handle>
GuardScope ScopeForHolder(const Handle* seq) noexcept {
	return seq->pool != nullptr ? GuardScope::kAny : GuardScope::kPooled;
}

namespace {
// The live-handle counts (SslmGpuContext::live_handles, SslmGpuModelHandle::live_sequences and
// live_pools) are changed by verbs the header lets run concurrently on distinct handles (a budgeted
// create beside a submitter, two pooled releases), so each is atomic (1.13 contract strike S3). A
// decrement never takes a count below zero, as the plain `if (n > 0) n -= 1` it replaces did not.
void DecrementLiveCount(std::atomic<int64_t>& count) noexcept {
	int64_t n = count.load(std::memory_order_acquire);
	while (n > 0 && !count.compare_exchange_weak(n, n - 1, std::memory_order_acq_rel, std::memory_order_acquire)) {
	}
}
}  // namespace

struct SslmGpuContext {
	superslm_gpu::harness::Device device;

	// Mirrors harness::GetOrBuildPipeline's cache shape (one-SRV-one-UAV B1
	// primitive-battery pipelines), now owned per-context instead of a function-
	// local static. Unused until a later B-section routes a call through it;
	// present now so B1 delivers the object the design's own gate names ("PSO/
	// root-sig cache moved off harness::GetDevice()'s static onto the context
	// object") as a real, populatable member, not a placeholder added later.
	std::map<std::string, superslm_gpu::harness::CachedPipeline> simple_pipeline_cache;

	// Mirrors harness::GetOrBuildComposedPipeline's cache shape (the 14-shader
	// composed per-layer-dispatch pipeline family) plus its own single shared
	// root signature, owned per-context for the identical reason.
	std::map<std::string, superslm_gpu::harness::CachedPipeline> composed_pipeline_cache;
	Microsoft::WRL::ComPtr<ID3D12RootSignature> composed_root_sig;

	// ContextHasLiveHandles guard (design Sec4.1.1, Sec9): incremented by every
	// model/adapter/sequence handle this context issues (B2/B3/B6, not yet wired
	// -- no such handle type is constructible yet), decremented on release.
	// destroy() refuses while this is nonzero. Atomic: see DecrementLiveCount.
	std::atomic<int64_t> live_handles{0};

	// Set once destroy() has freed this object's own device/GPU state, so a
	// caller who somehow retains a dangling pointer past a successful destroy
	// cannot be silently handed a half-torn-down object by a second call through
	// the same pointer (destroy is not re-entrant against its own prior success;
	// the object itself is deleted at the end of a successful destroy, so this
	// flag is defensive documentation of that invariant rather than a runtime
	// branch anything reads today).
	bool destroyed = false;

	// T-2851 (A): the host parallel-for hook (sslm_gpu_context_set_host_parallel_for), copied in;
	// `run == NULL` means none. Read only by SslmGpuSeqFinishTokenForG5BridgeImpl's host logits
	// step (LogitsSiteParallel), for a model without a device-resident head.
	sslm_parallel_for host_parallel_for{};
};

// T-2851 (B, design Sec4.6 item 2): everything the device logits step reuses, owned by one mapped
// model (SslmGpuModelHandle::device_logits; null when the model was mapped without
// SSLM_GPU_RESIDENCY_HEAD_ON_DEVICE). Created by CreateDeviceLogitsBuffers and passed explicitly
// to RunDeviceLogits, so the finish reaches every resource it reuses through its own model, and
// two mapped models never share scratch. The context supplies the device, queue and command list.
// Roman numerals follow the design's buffer list; (i), the head's upload-heap staging buffer, is
// local to CreateDeviceLogitsBuffers and released once its copy has completed.
struct DeviceLogitsBuffers {
	uint32_t V = 0, H = 0;
	Microsoft::WRL::ComPtr<ID3D12Resource> head;      // (ii) default heap, V x H B (padded to 4)
	Microsoft::WRL::ComPtr<ID3D12Resource> x_upload;  // (iii) upload heap, H x 4 B, mapped
	int32_t* x_upload_ptr = nullptr;
	Microsoft::WRL::ComPtr<ID3D12Resource> x;         // (iv) default heap, H x 4 B
	Microsoft::WRL::ComPtr<ID3D12Resource> out;       // (v) default heap, V x 8 B, UAV
	Microsoft::WRL::ComPtr<ID3D12Resource> readback;  // (vi) readback heap, V x 8 B, mapped
	const int64_t* readback_ptr = nullptr;
	Microsoft::WRL::ComPtr<ID3D12RootSignature> root;
	Microsoft::WRL::ComPtr<ID3D12PipelineState> pso;
	uint32_t groups_x = 0, groups_y = 0;  // the logits_site dispatch grid

	DeviceLogitsBuffers() = default;
	DeviceLogitsBuffers(const DeviceLogitsBuffers&) = delete;
	DeviceLogitsBuffers& operator=(const DeviceLogitsBuffers&) = delete;
	~DeviceLogitsBuffers() {
		// The two persistent mappings end with the bundle (sslm_gpu_model_unmap's release).
		if (x_upload && x_upload_ptr) x_upload->Unmap(0, nullptr);
		if (readback && readback_ptr) {
			D3D12_RANGE none{0, 0};
			readback->Unmap(0, &none);
		}
	}
};

// T-2113 (B2, design Sec5.1/Sec10 B2): the real SslmGpuModelHandle. Owns the model's own
// weight residency and the T-2105 RoPE cos/sin table residency (Claude/Laplace/
// t2105-gpu-speed-ceiling-2026-08-14.md Sec2 change 1, re-derived here per the design's
// own "re-derived, never cherry-picked" instruction, Sec1 constraints paragraph) --
// uploaded ONCE, inside map(), into DEFAULT-heap buffers this handle owns for its whole
// lifetime (design Sec5.1: "weight buffers upload once... and stay resident for the mapped
// model handle's entire lifetime"). Keyed by the artifact's own content hash
// (SslmModelView::RawIntegrityHash(), design Sec5.1) rather than by the caller's host
// pointer -- this is the structural fix D-SLM3311 names (Sec2.1): identity is "which
// handle," never "which address the allocator happened to hand back." No process-global
// weight/RoPE cache exists anywhere in this file (D-SLM3294) -- every sslm_gpu_model_map
// call uploads its OWN dedicated residency, independent of any other live handle, matching
// design Sec5.1's own "multiple SslmGpuModelHandles may be live at once... nothing in this
// design imposes a one-model-at-a-time limit."
//
// The T-2105 "root-binding-hoist" half of this section's own delivery line (design Sec10
// B2) is NOT built here: hoisting the twelve per-dispatch SetComputeRoot* calls out of a
// dispatch-recording loop has nothing to hoist out of until B4 builds that loop -- this is
// named explicitly, not silently dropped, per StandardsDocument.md Sec5.6 (a deferral is
// surfaced loudly). B4's own gate (design Sec10) is where the composed dispatch chain --
// and therefore the rebind site this handle's own resident buffers get bound at -- first
// exists; see the B2 section of Claude/Brunel/t2113-1p0-core-build-2026-08-15.md.
struct SslmGpuModelHandle {
	SslmGpuContext* ctx = nullptr;
	std::array<uint8_t, superslm::kIntegrityHashBytes> content_hash{};

	// Resident, DEFAULT-heap, read-only for this handle's whole lifetime (design Sec5.4:
	// "read-only for the whole of their lifetime after map returns... safe for any number
	// of threads to read concurrently, by construction").
	Microsoft::WRL::ComPtr<ID3D12Resource> weights_buf;
	Microsoft::WRL::ComPtr<ID3D12Resource> rope_cos_buf;
	Microsoft::WRL::ComPtr<ID3D12Resource> rope_sin_buf;
	bool has_rope_tables = false;  // true iff the artifact's own rope_tables.Tensor("cos"/"sin") existed
	// T-2113 (B5): the RoPE presence/element-count metadata `RunLayerLoopGpuSubmit`'s own
	// external-rope bridge needs at every decode call -- this handle keeps no live
	// SslmTensorManifest past map() returning (map()'s own locals, above, go out of scope),
	// so these three small facts are captured here, once, instead of re-deriving them from a
	// manifest sslm_decode_step_gpu has no access to.
	uint64_t rope_cos_elem_count = 0;
	uint64_t rope_sin_elem_count = 0;

	// The layout/dimension set this handle's own weights_buf was packed at (superslm_gpu::
	// GpuLayerLayout, T-2113 B2's own shared PackLayerWeightsBytes) -- B4 reads these to
	// bind the composed dispatch chain's own per-layer stride/offsets against this handle's
	// buffer without re-deriving them from the model view a second time.
	superslm_gpu::GpuLayerLayout layout{};
	uint32_t num_hidden_layers = 0;
	uint32_t hidden_size = 0;
	uint32_t head_dim = 0;
	uint32_t num_key_value_heads = 0;
	uint32_t num_attention_heads = 0;
	uint32_t intermediate_size = 0;
	int64_t context_cap = 0;
	// Paged-KV plan §3.1, step G3: the artifact's kv_block_size, the page geometry's B when it
	// divides the cap (superslm_gpu::kv_pool::ComputeGpuPageGeometry).
	uint32_t kv_block_size = 0;
	bool has_qk_norm = false;
	uint32_t dispatches_per_layer = superslm_gpu::kLegacyDispatchesPerLayer;

	// ModelHasLiveSequences guard (design Sec9, Sec4.2's own model_unmap row): T-2113 (B3)
	// incremented by sslm_gpu_seq_create, decremented on release, below.
	std::atomic<int64_t> live_sequences{0};  // atomic: see DecrementLiveCount
	// Paged KV (G3): GPU page pools created against this model (sslm_gpu_kv_pool_create, destroy).
	// A pool keeps a `model` pointer for the budgeted create's model check, so unmap refuses while
	// one lives (SSLM_MODEL_HAS_LIVE_SEQUENCES; the plan names no status for it, noted at G3).
	std::atomic<int64_t> live_pools{0};  // atomic: see DecrementLiveCount

	// T-2113 (design Sec4.2/Sec9, routed `Claude/Curie/t2112-1p0-red-suite-2026-08-15.md` Sec7.2
	// item 2, D-SLM3417): design Sec9's own Busy row states `sslm_gpu_model_unmap` returns Busy
	// "against a model with any Submitted sequence bound... which takes precedence" over
	// ModelHasLiveSequences' own "any state, not only Submitted" caveat -- `live_sequences` alone
	// cannot distinguish a Submitted-bound model from an Idle-bound one. Incremented in
	// SubmitOneSequenceDecode (below) the instant a bound sequence transitions Idle -> Submitted;
	// decremented in sslm_gpu_ready the instant that SAME sequence collapses back to Idle. Every
	// sequence that ever transitions to Submitted is, by construction, still bound to exactly the
	// model it was created against (a sequence's own `model` pointer never changes across its
	// lifetime) -- so this count is always a count of THIS model's own Submitted sequences.
	//
	// T-2124 (D-SLM3446 S2, Claude/Poirot/435f730-t2124-adapter-uaf-review.md): `std::atomic`, not
	// a plain `int64_t` -- design Sec5.4 blesses two threads each driving a DIFFERENT sequence
	// handle concurrently, and Sec6.2's own Submit/Finish split is built precisely so the submit
	// (writer: SubmitOneSequenceDecode) and the drain (writer: sslm_gpu_ready) can happen on
	// different threads. A plain `+= 1`/`-= 1` is a non-atomic read-modify-write; two sequences
	// sharing one model, driven from two threads, can race it -- a lost increment reopens the
	// model-handle analogue of this ticket's own adapter UAF class (the guard below reads 0 while
	// a sequence is genuinely Submitted), a lost decrement makes the model permanently un-unmappable.
	// `seq_cst` (the type's own default for every operator used below: `+=`, `-=`, `> 0`, `load()`)
	// is kept deliberately rather than relaxed to `acquire`/`release`: this counter gates a
	// correctness precondition (whether it is safe to free device memory the GPU may still read),
	// not a performance-sensitive hot path -- one Submit/Finish pair per decode step, not per
	// dispatch -- so there is no measured cost this file's own conventions (Sec12, performance
	// claims anchored to execution) would accept trading away the simplest-to-reason-about ordering
	// for. A weaker order would need its own argued happens-before edge to the buffer-free this
	// counter gates; none is on record, so none is assumed.
	// From G3 (plan §4.4, rev 11): private, written only by SetGpuSeqState; read through
	// submitted_sequences().
	int64_t submitted_sequences() const noexcept { return submitted_sequences_.load(); }

	// T-2243 (M2, design Sec9/plan Sec6.1, D-SLM3965): count of currently-mapped adapters
	// against this model -- incremented in sslm_gpu_adapter_map, decremented in
	// sslm_gpu_adapter_unmap. `sslm_gpu_model_unmap` rejects (SSLM_MODEL_HAS_LIVE_ADAPTERS)
	// while this is nonzero -- an adapter handle retains a raw `model` pointer
	// (SslmGpuAdapterHandle::model, below) that is never dereferenced today but must not be
	// left dangling, the same reasoning `live_sequences` already applies to a bound sequence's
	// own `model` pointer. Plain `int64_t`, not atomic: written only from sslm_gpu_adapter_map/
	// _unmap, both ordinary single-threaded map/unmap calls -- never the async Submit/Finish
	// split that motivates `submitted_sequences`' own atomicity above.
	int64_t live_adapters = 0;

	// T-2113 (B3.5, design Sec5.1's own amendment / Sec5.3a): host-side metadata
	// sslm_gpu_seq_embed_token needs for its whole lifetime -- NOT new GPU residency (the
	// design's own explicit "not new residency" clause, Sec5.3a): the embedding matrix
	// itself has no dispatch site and is never bound to a command list, so it stays a
	// plain host-side copy, taken once at map() time (mirroring the packed weight bytes'
	// own "upload once, inside map" discipline, but never uploaded to a device buffer
	// since nothing ever reads it from a shader). Copied rather than pointed at `base`
	// (the SslmModelView passed to sslm_gpu_model_map) because that view's own lifetime is
	// caller-owned and this handle must serve sslm_gpu_seq_embed_token for its OWN whole
	// lifetime, which can outlive the caller's view -- the identical reasoning that
	// already governs why LayerWeights/lw_bytes above are marshaled and packed at map()
	// time rather than read from `base` lazily per call.
	std::vector<int8_t> embed_weights;
	superslm::CarriedScale embed_site_constant{};
	int32_t vocab_size = 0;
	// Precached prefixes: the artifact's kv_precision, the SPX1 header field a restore checks.
	uint32_t kv_precision = 0;

	// G5-5 (T-2132, Brunel): mask-page residency on device (design Sec6 G5-5's own "Builds"
	// list, VRAM, alongside weights/scales per the existing GPU-tenancy model, Sec8.4) --
	// uploaded, read-only, once here, the same UploadResidentBufferSync three-step shape
	// weights_buf/rope_cos_buf/rope_sin_buf already use above. Empty/null when this model's
	// artifact carries no SchemaMasks section (an unconstrained-only artifact, design Sec13.1) --
	// the same "absence loads exactly as before this section type existed" disposition the CPU
	// ABI's own sslm_model_map already applies (src/sslm_abi.cpp).
	Microsoft::WRL::ComPtr<ID3D12Resource> mask_pages_buf;
	// Host-owned copy of the SchemaMasks section's own bytes -- `schemas`'s own SchemaEntry
	// pointers point INTO this vector (schema_masks.h's own "entries point into section_data,
	// the caller must keep the bytes alive" contract), so this handle owns them for its whole
	// lifetime, exactly the "copied rather than pointed at `base`" reasoning this file's own
	// embed_weights capture (B3.5, above) already documents for the identical caller-view-
	// lifetime hazard.
	std::vector<uint8_t> schema_section_bytes;
	// The SAME SchemaMasksTable reader the CPU ABI parses this model's own SchemaMasks section
	// with (schema_masks.h) -- reused, not re-derived, so a schema's mask pages/CSR transitions
	// mean the identical bits on both paths. Count()==0 when this model carries no schema.
	superslm::SchemaMasksTable schemas;
	// G5-5's own final_norm+logits capture (design's "no new arithmetic" claim: the GPU-1.0
	// substrate's job is the layer loop only, design Sec4/Sec12's own established scope -- the
	// finish-token bridge below needs these three host-resident facts to run the SAME
	// RmsNormSite/LogitsSite calls sslm_decode_step's own finishing block already runs, on this
	// model's own GPU-derived hidden state). Mirrors embed_weights/embed_site_constant's own
	// capture immediately above -- read from `*base` once, at map time, never re-read from the
	// caller's view afterward (the identical caller-view-lifetime reasoning).
	std::vector<int32_t> final_norm_gain;
	superslm::CarriedScale final_norm_site_constant{};
	// T-2851 (design Sec4.6a): a host copy of `lm_head`, taken only for an UNTIED artifact mapped
	// without SSLM_GPU_RESIDENCY_HEAD_ON_DEVICE. Empty for a tied artifact, whose head is the
	// embedding `embed_weights` already holds, and empty when the head lives on the device.
	std::vector<int8_t> head_weights;
	// The host head the finish reads, set once at map: `embed_weights.data()` when tied,
	// `head_weights.data()` when untied and flag clear, null when the head is on the device (the
	// host logits path does not run for such a model). Lifetime: the handle's.
	const int8_t* head_rows = nullptr;
	// T-2851 (B): the device logits bundle, present only for a model mapped with
	// SSLM_GPU_RESIDENCY_HEAD_ON_DEVICE. Released with the handle.
	std::unique_ptr<DeviceLogitsBuffers> device_logits;

	// T-2114 (S2, Claude/Poirot/50f3d5d-t2113-1p0-gpu-core-build-review.md): set at the START
	// of unmap(), before `delete this`, matching `SslmGpuContext::destroyed`'s own established
	// disposition (this file, above) -- defensive documentation of "this object's own release
	// has begun," not a runtime branch anything reads. It CANNOT be, structurally: every call
	// site that used to test `->destroyed` on this handle read it AFTER `delete`, which is a
	// use-after-free (undefined behavior, not a check) on a released handle, and is always
	// false -- vacuously -- on a live one, since the flag is set only the line before delete.
	// The twelve call sites this used to gate (across model/adapter/sequence handles) had their
	// own `->destroyed` reads removed in the same fix round: a caller that passes an already-
	// released handle to any 1.0 entry point again invokes undefined behavior, exactly as
	// passing a freed pointer to any C API does -- this was already true before S2 (the guard
	// never protected against it), S2 only removes the code that pretended otherwise.
	bool destroyed = false;

private:
	friend void SetGpuSeqState(SslmGpuSequenceHandle*, superslm_gpu::SslmSequenceGpuState,
	                           const SslmGpuAdapterHandle*);
	std::atomic<int64_t> submitted_sequences_{0};
};

// T-2113 (B6, design Sec5.2/Sec10 B6): the real SslmGpuAdapterHandle. Owns the adapter's own
// device residency, independent of any sequence (design Sec5.2: "resident and shared read-only
// by every sequence that later binds this handle"). Mirrors the CPU-side `LoadAdapterArtifact`/
// `AdapterHandle` shape (include/superslm/adapter_marshal.h, relocated from tools/
// sslm_adapter_loader.h this section, D-SLM3368) exactly, on the device side, per design Sec5.2's
// own instruction. `model` is retained (not merely its content hash) so `sslm_decode_step_gpu`'s
// own AdapterModelMismatch check (design Sec5.2: "a pointer-equality check against the handle
// each side already carries, not a re-hash") is a real pointer compare, never a re-hash.
//
// Residency shape this section builds: every (layer, projection) pair the adapter's own
// DeltaFoldScales section covers gets its lora_A/lora_B bytes and DeltaFoldScales/UFoldScales
// (identity, mult, shift) triples packed into three DEFAULT-heap resident buffers (mirroring the
// model handle's own three-buffer split, weights/rope_cos/rope_sin) -- uploaded ONCE, here, never
// per decode call. `AdapterProjSlot` records each covered slot's own byte offset into those
// buffers plus its rank, indexed [layer][projection] host-side, so a later section's dispatch-
// recording code can bind the right sub-range without re-deriving the packing.
//
// NOT built this section, named per StandardsDocument.md Sec5.6 (a deferral surfaced loudly, not
// silently folded into "adapter support"): nothing in the dispatch-recording path
// (RunLayerLoopGpuSubmit, src/gpu/superslm_gpu.cpp) reads these buffers yet -- no GEMM site's own
// HLSL applies the delta this handle makes resident. `sslm_decode_step_gpu` accepts a bound
// adapter, validates it, and records the identical base-only dispatch chain regardless. See
// Claude/Brunel/t2113-1p0-core-build-2026-08-15.md Sec9 for the full scope statement.
// T-2113 (B6b): AdapterProjSlot now lives in gpu_port.h (superslm_gpu::AdapterProjSlot)
// so RunLayerLoopGpuSubmit's own dispatch-recording code (superslm_gpu.cpp) can read it
// through GpuAdapterBridge without depending on this TU's opaque handle types -- the
// same relocation shape D-SLM3351/D-SLM3368 already used for the marshal headers.
// Aliased here so every existing reference in this file (`AdapterProjSlot`, unqualified)
// keeps compiling unchanged.
using AdapterProjSlot = superslm_gpu::AdapterProjSlot;

struct SslmGpuAdapterHandle {
	SslmGpuContext* ctx = nullptr;
	SslmGpuModelHandle* model = nullptr;  // AdapterModelMismatch check, design Sec5.2: pointer-equality

	Microsoft::WRL::ComPtr<ID3D12Resource> lora_ab_buf;  // every covered slot's lora_A+lora_B bytes
	Microsoft::WRL::ComPtr<ID3D12Resource> fold_buf;     // every covered slot's DFS+UFS (identity,mult,shift) triples

	// [layer][7 projections: q,o,gate,up,down,k,v] -- AdapterProjSlot::present false for a
	// (layer, projection) this adapter does not cover, the identical NULL-adapter-per-projection
	// shape LayerAdapterProjection::a_weight==nullptr already documents on the CPU path
	// (forward_sites.h).
	std::vector<std::array<AdapterProjSlot, 7>> slots;
	uint32_t rank = 0;

	// T-2124 (D-SLM3446 P0-3): count of currently-Submitted sequences whose in-flight decode call
	// bound this adapter -- the adapter-handle analogue of SslmGpuModelHandle::submitted_sequences
	// above (identical Busy-precedence shape, D-SLM3417). `mutable`: written through the const
	// SslmGpuAdapterHandle* every decode-step call receives (design Sec5.2's own "checked at every
	// call" contract keeps that pointer const at the public boundary), unlike the model handle's
	// own field, which SubmitOneSequenceDecode already holds through a non-const `model` local.
	// See sslm_gpu_adapter_unmap's own header comment for why this exists: the command list a
	// decode-step call records reads this handle's own lora_ab_buf/fold_buf via root descriptors
	// and returns before the fence signals, so deleting this handle while any count above zero is
	// outstanding would free a buffer the GPU may still be reading from.
	//
	// `mutable` is a DELIBERATE weakening of the published `const SslmGpuAdapterHandle*` contract
	// every public decode signature (include/superslm/gpu_1p0.h) presents to a caller -- not merely
	// a consequence of that pointer's constness. A caller reasoning from the header's `const` sees a
	// handle safe to share read-only across threads (design Sec5.4's own claim); this field is the
	// one piece of state that call actually writes through that same const pointer. Named here
	// explicitly (T-2124 S2, Claude/Poirot/435f730-t2124-adapter-uaf-review.md) because the prior
	// wording of this comment described only the mechanism and not the contract it weakens, which is
	// what let the concurrency consequence (S2, below) go unexamined at authoring time. `std::atomic`
	// (not a plain `int64_t`): see SslmGpuModelHandle::submitted_sequences' own comment, above, for
	// the race this closes and the memory-order justification -- identical reasoning, restated there
	// rather than duplicated here since this field mirrors that one's shape exactly.
	// From G3 (plan §4.4, rev 11): private, written only by SetGpuSeqState; read through
	// submitted_sequences().
	int64_t submitted_sequences() const noexcept { return submitted_sequences_.load(); }

	// T-2243 (S2, design plan Sec6.1, D-SLM3965): count of sequences currently HOLDING a bind
	// to this adapter (via sslm_gpu_seq_bind_adapter), whether or not a call is in flight this
	// instant -- a different lifetime from `submitted_sequences` above, which counts only calls
	// genuinely in flight. `sslm_gpu_adapter_unmap` rejects (SSLM_ADAPTER_HAS_BOUND_SEQUENCES)
	// while this is nonzero: an idle-but-bound sequence's NEXT call would read a `bound_adapter`
	// pointer that has since been freed, the gap this counter closes. `mutable`/`std::atomic`
	// for the identical reason `submitted_sequences` above is: written through the `const
	// SslmGpuAdapterHandle*` every bind call receives.
	mutable std::atomic<int64_t> bound_sequences{0};

	// T-2114 (S2): see SslmGpuModelHandle's own comment on its `destroyed` field, above --
	// identical disposition here.
	bool destroyed = false;

private:
	friend void SetGpuSeqState(SslmGpuSequenceHandle*, superslm_gpu::SslmSequenceGpuState,
	                           const SslmGpuAdapterHandle*);
	mutable std::atomic<int64_t> submitted_sequences_{0};
};

struct SslmGpuSequenceHandle;
// Paged-KV plan §4.4 (rev 11, rev 15.2), step G3: the counting accessor, the one writer of a
// sequence handle's `state`, of the counts that a move into or out of Submitted raises or lowers,
// and of a pool's `submitted` holder pointer. Defined below the sequence handle.
void SetGpuSeqState(SslmGpuSequenceHandle* seq, superslm_gpu::SslmSequenceGpuState next,
                    const SslmGpuAdapterHandle* adapter);

// Paged-KV plan (rev 16.1) §4 items 3 and 4, step G3: the GPU page pool. One DEFAULT-heap UAV of
// `page_count * page_bytes` bytes, zero-filled once at create (AllocateZeroFillBuffers and
// ZeroFillResidentUavBufferSync, below); a host mirror of the same size, kept complete by every
// pooled submission's readback (G24) and the `base` of the `kv_pages` allocator -- which
// poison-fills a freed dirty page with 0xCD and never zero-fills, so it addresses the mirror and
// never device memory; the pool-level resume latch every pooled sequence passes (item 4); and the
// one Submitted pooled holder, a holder pointer written only by SetGpuSeqState (rev 15.2).
struct SslmGpuKvPoolHandle {
	SslmGpuContext* ctx = nullptr;
	SslmGpuModelHandle* model = nullptr;
	superslm_gpu::kv_pool::GpuPageGeometry geo{};
	uint32_t page_count = 0;
	size_t pool_bytes = 0;
	Microsoft::WRL::ComPtr<ID3D12Resource> pool_buf;
	std::vector<uint8_t> host_mirror;
	superslm::kv_pages::PagePool* pages = nullptr;
	// Item 4: true once a pooled submission's readback left `pool_buf` in COPY_SOURCE; read and
	// consumed by the next pooled submission's resume barrier, with 1.8.0's restore-on-failure.
	bool kv_needs_resume_barrier = false;
	// Sequences drawn from this pool and not yet released; destroy refuses while nonzero. Atomic, as
	// the CPU twin's is (review-g23 nit: the arc verbs change it beside other threads' calls).
	std::atomic<int64_t> live_refs{0};

	SslmGpuKvPoolHandle() = default;
	SslmGpuKvPoolHandle(const SslmGpuKvPoolHandle&) = delete;
	SslmGpuKvPoolHandle& operator=(const SslmGpuKvPoolHandle&) = delete;
	~SslmGpuKvPoolHandle() {
		if (pages) superslm::kv_pages::PoolDestroy(pages);
	}
	// The pooled handle of this pool that is Submitted, or null. No guard reads it (§4 item 4).
	const SslmGpuSequenceHandle* submitted() const { return submitted_.load(std::memory_order_acquire); }

private:
	friend void SetGpuSeqState(SslmGpuSequenceHandle*, superslm_gpu::SslmSequenceGpuState,
	                           const SslmGpuAdapterHandle*);
	// Atomic (review-g23 F3): a poller's lowering clears it while a sibling's submission may set it.
	std::atomic<SslmGpuSequenceHandle*> submitted_{nullptr};
};

// T-2113 (B3, design Sec5.3/Sec10 B3): the real SslmGpuSequenceHandle. Owns a
// **dedicated** DEFAULT-heap K/V buffer, sized to this handle's own `context_cap` at
// create() time -- this is the direct replacement for `g_resident_kv`'s single
// pointer-keyed slot (superslm_gpu.cpp): every sequence gets its own buffer, so two
// concurrently-live sequences never share, alias, or evict each other's K/V state, and
// nothing about a second sequence's existence can invalidate the first's cache the way
// a colliding freed address could under the substrate's own keying (D-SLM3311, design
// Sec2.1/Sec5.3). Also owns the host-mirrored SequenceLayerState fields the design
// names (Sec5.3: "hidden_codes, hidden_scale, layer_index, kv_saturation_count,
// context_length") -- a caller decodes through this handle without maintaining a
// separate SequenceLayerState of its own. `kv_needs_resume_barrier` is this handle's
// own half of the RunLayerLoopGpu bridge (gpu_port.h, T-2113 B3's own additive
// trailing-parameter pair): true once a call has left this handle's own kv_buf in
// COPY_SOURCE state (the targeted per-call readback always does this), read and
// consumed by the NEXT call through this handle.
//
// From G3 (plan §4.4, rev 10, rev 11, rev 12.1): non-copyable and non-assignable; `state` is
// private (an atomic, read through state()) and written only by SetGpuSeqState; the handle is the
// walk's on-handle binding (inflight::InFlightHandleBinding: SubmittedNow() is a plain atomic read
// of the state), and its `in_flight` token is constructed with that binding.
struct SslmGpuSequenceHandle final : public superslm_gpu::inflight::InFlightHandleBinding {
	SslmGpuSequenceHandle() = default;
	SslmGpuSequenceHandle(const SslmGpuSequenceHandle&) = delete;
	SslmGpuSequenceHandle& operator=(const SslmGpuSequenceHandle&) = delete;
	~SslmGpuSequenceHandle() = default;

	superslm_gpu::SslmSequenceGpuState state() const noexcept { return state_.load(); }
	bool SubmittedNow() const noexcept override {
		return state_.load() == superslm_gpu::SslmSequenceGpuState::Submitted;
	}

	SslmGpuContext* ctx = nullptr;
	SslmGpuModelHandle* model = nullptr;

	Microsoft::WRL::ComPtr<ID3D12Resource> kv_buf;
	bool kv_needs_resume_barrier = false;
	size_t kv_bytes = 0;

	int64_t context_cap = 0;

	// Host-mirrored SequenceLayerState surface (design Sec5.3) -- `hidden_codes` is
	// owned here (a std::vector, sized to the model's own hidden_size) rather than
	// caller-supplied, since the 1.0 API's own sequence handle is the thing that owns
	// this state now, not a struct the caller constructs by hand (design Sec4.2's own
	// "no SequenceLayerState a test constructs by hand" framing, generalized to every
	// caller of the 1.0 surface).
	std::vector<int8_t> hidden_codes;
	superslm::CarriedScale hidden_scale{};
	uint32_t layer_index = 0;
	uint64_t kv_saturation_count = 0;
	int64_t context_length = 0;

	// T-2113 (B5, design Sec4.2/Sec6.2/Sec10 B5): the async decode lifecycle's own state.
	// `live_state.hidden_codes` is set ONCE, in sslm_gpu_seq_create, to alias
	// `hidden_codes.data()` directly -- `RunLayerLoopGpuSubmit`/`Finish` read/write through
	// this pointer, so the handle's own `hidden_codes` vector is always the live content, no
	// separate copy. The four scalar fields are synced FROM this handle's own
	// hidden_scale/layer_index/kv_saturation_count/context_length immediately before every
	// sslm_decode_step_gpu call, and copied back INTO them once sslm_gpu_ready's own Finish
	// call completes (design Sec4.2: "collapses Submitted -> Completed -> Idle").
	superslm::SequenceLayerState live_state{};
	// The host mirror `RunLayerLoopGpuSubmit`'s external-K/V readback scatters each call's
	// own newly-written rows into (design Sec5.3's own "still gets the identical CPU-oracle-
	// comparable host mirror" contract) -- sized identically to `kv_bytes` (the same buffer
	// `sslm_gpu_seq_create` already sizes `kv_buf` to), allocated once, accumulated across
	// every decode step this handle ever runs, exactly like the pre-1.0 substrate's own
	// per-session `workspace` vector (tools/t2100_gpu_throughput.cpp's own `ws`).
	std::vector<uint8_t> host_kv_mirror;
	// The in-flight token between a Submitted sslm_decode_step_gpu call and the
	// sslm_gpu_ready call that finishes it -- empty whenever this handle is Idle/Completed. From
	// G3 a GpuInFlightToken bound to this handle (plan §4.4, *Three states*): the walk at the
	// outermost call's return passes its record while this handle is Submitted.
	superslm_gpu::GpuInFlightToken in_flight{this};

	// T-2243 (S2, design plan Sec6.1): the adapter this sequence is currently bound to, across
	// calls -- set by `sslm_gpu_seq_bind_adapter`, distinct from `in_flight_adapter` below
	// (which answers "which adapter did THIS in-flight call use" and is cleared the instant
	// that call's own window closes). The two G5 bridge choke points
	// (`DriveGpuSeqToFullDepthForG5Bridge`, `SubmitAdmittedChunkForG5Bridge`) read this field
	// and pass it into the existing per-call `adapter_or_null` machinery, which populates
	// `in_flight_adapter` for that call exactly as it does today for a direct caller of
	// `sslm_decode_step_gpu`. Null on a freshly created or freshly restored handle (a
	// save/restore blob carries no adapter identity); preserved across `sslm_gpu_seq_reset`
	// (the same "caller's own standing configuration survives reset" precedent
	// `bound_schema_index` below already sets).
	const SslmGpuAdapterHandle* bound_adapter = nullptr;

	// G5-5 (T-2132, Brunel): the GPU-1.0 twin of `sslm_seq_s::bound_schema`/`dfa_walk_state`
	// (src/sslm_abi.cpp) -- a sibling scalar pair on this wrapper handle, NEVER folded into
	// `live_state`/`SequenceLayerState` above, mirroring the CPU ABI's own established shape
	// exactly (SequenceLayerState is `forward_sites.h`'s own pinned "closed unit," §13 dim 9 --
	// design Sec6 G5-5's own "the sequence's own state, resident wherever the sequence's other
	// GPU-side state already lives" is satisfied by this handle, not by widening the shared
	// layer-loop struct every RunLayerLoop call already guards by exact field count). `-1` ==
	// no schema bound (this bridge's own int32_t index convention,
	// gpu_1p0_g5_bridge.h); `dfa_walk_state` stays kSslmGpuDfaWalkStateUnused until a schema
	// binds, matching `kDfaWalkStateUnused`'s identical CPU-side semantics.
	int32_t bound_schema_index = -1;
	uint32_t dfa_walk_state = kSslmGpuDfaWalkStateUnused;
	// Runtime-only bind authority. Create/reset set it; restore and every non-malformed generation
	// call clear it. It is deliberately absent from the save format.
	bool bind_eligible = false;

	// G5-5 session 3 fix (T-2132, Brunel, Claude/Brunel/t2132-g5-build-2026-08-16.md session 3):
	// the GPU-1.0 twin of `sslm_seq_s::ready_for_logits` (src/sslm_abi.cpp) -- session 3's own
	// diagnosis found the parity harness's own prompt-to-decode transition re-embedding the
	// prompt's last token (no GPU-side shortcut existed), committing a duplicate KV row and
	// running `context_length` +1 from decode step 0 onward, silently, until it flipped an
	// argmax pick 19 steps later. This field, set by `SslmGpuSeqPrefillPromptForG5Bridge`/
	// `SslmGpuSeqPrefillSchemaContentForG5Bridge` on success and consumed (cleared) by
	// `SslmGpuSeqDecodeStepForG5Bridge`, is the real fix: it gives the bridge -- not each
	// caller's own ad hoc composition -- the authority to skip re-embedding a token whose final
	// residual a prefill call already computed, mirroring `sslm_decode_step`'s own
	// `ready_for_logits` branch (src/sslm_abi.cpp:1708) exactly. `sslm_gpu_seq_reset` clears it,
	// mirroring `sslm_seq_reset`'s own identical clear (src/sslm_abi.cpp:1488).
	bool ready_for_logits = false;

	// The prefill snapshot (sslm_gpu_seq_read_prefill_final_hidden, gpu_1p0.h): a copy of the
	// completed last-position residual of this sequence's most recent prefill call that reached
	// its admission pre-scan and returned SSLM_OK. `hidden_codes` above is the LIVE residual that
	// embed, the decode calls, finish and the failure exits of either prefill twin all write, and
	// `ready_for_logits` is the decode wrapper's shortcut flag, which several of those leave set;
	// neither is a record of a prefill. This pair is. It has exactly these writers, and no other
	// code in this file writes it:
	//   - sslm_gpu_seq_create (and so sslm_gpu_seq_restore): sized to hidden_size, invalid;
	//   - sslm_gpu_seq_reset: invalid;
	//   - either prefill twin, immediately before its RunChunkAdmissionPreScan call: invalid;
	//   - either prefill twin's kNone exit, as its last statement before `return SSLM_OK`: copies
	//     `hidden_codes`, valid.
	// No scale is stored: the read's only computation is RmsNormSite, which does not read its
	// incoming scale (forward_sites.h), so a stored scale would be state nothing consumes.
	std::vector<int8_t> prefill_hidden_codes;
	bool prefill_hidden_valid = false;

	// Paged-KV plan §4 items 5, 6b and 7, step G3: a POOLED (budgeted) sequence's own state, all
	// null or zero on a dedicated sequence (sslm_gpu_seq_create, unchanged). `holder` is its
	// kv_pages page table and reserve over `pool`; `table_buf` its per-sequence DEFAULT-heap table
	// buffer (ceil(cap / B) * 4 bytes, resting in COMMON), uploaded at the head of the next
	// submission while `table_dirty` (a page was mapped, or the table emptied, since the last one).
	// `limit` = min(origin + budget, cap) (§3.4); origin is 0 until G4's prefixes.
	SslmGpuKvPoolHandle* pool = nullptr;
	superslm::kv_pages::PageHolder* holder = nullptr;
	Microsoft::WRL::ComPtr<ID3D12Resource> table_buf;
	bool table_dirty = false;
	int32_t budget = 0;
	int64_t origin = 0;
	int64_t limit = 0;
	// Test seams (sslm_gpu_test_only_arm_view_fault / _last_ladder_status): a one-shot fault on the
	// view the next pooled submission hands the ladder, and the ladder's status for the last
	// submission (Ok when it was not refused before recording).
	uint32_t armed_view_fault = 0;
	superslm::SslmForwardStatus last_ladder_status = superslm::SslmForwardStatus::Ok;

	// T-2114 (S2): see SslmGpuModelHandle's own comment on its `destroyed` field
	// (gpu_1p0.cpp) -- identical disposition here.
	bool destroyed = false;

	// The same-sequence decode claim (t2112 dim3 M2, the 1.13 box finding; widened by the 1.13
	// contract strike's B1): taken with one atomic exchange by every decode entry point on this
	// handle -- sslm_decode_step_gpu, each batch slot, SslmGpuSeqDecodeStepForG5Bridge and
	// SslmGpuSeqFinishTokenForG5Bridge (SeqDecodeClaim, below) -- before it reads `state`, and held
	// until the call returns, by which time a submission that succeeded has moved the handle to
	// Submitted. A second thread's decode or finish of the SAME handle that finds it held returns
	// SSLM_BUSY and touches nothing. Without it the Idle check, EntrySubmitGuard and
	// BeginRecording()'s ledger check are three unlocked reads, so two threads on one handle could
	// all pass them and record into the singleton's one command list at once (heap corruption under
	// the D3D12 debug layer, a winevulkan abort on the software GPU, or a silent Ok/Ok). It never
	// blocks and never serialises different handles: P1's precondition (§4.4) still binds those.
	std::atomic<bool> decode_claimed{false};

private:
	friend void SetGpuSeqState(SslmGpuSequenceHandle*, superslm_gpu::SslmSequenceGpuState,
	                           const SslmGpuAdapterHandle*);
	// Private, so only the accessor can write them (review-g23 nit).
	//
	// T-2124 (D-SLM3446 P0-3): the adapter bound at this handle's own currently-in-flight
	// submission, null when no adapter was bound or nothing is in flight -- set in
	// SubmitOneSequenceDecode alongside `in_flight` above, cleared and used to decrement the
	// adapter's own `submitted_sequences` count in sslm_gpu_ready the instant this sequence
	// collapses back to Idle. An adapter is a per-CALL argument (sslm_decode_step_gpu's own
	// `adapter_or_null`), never state a sequence handle otherwise retains, so this is the one
	// place a submitted call's own adapter binding needs remembering between Submit and Finish.
	const SslmGpuAdapterHandle* in_flight_adapter = nullptr;
	// SetGpuSeqState's record of which counts its last move to Submitted raised, so the move out
	// lowers exactly those (§4.4, *Three states*). Written only by the accessor.
	bool raised_process = false;
	bool raised_pooled = false;
	SslmGpuModelHandle* raised_model = nullptr;
	const SslmGpuAdapterHandle* raised_adapter = nullptr;
	std::atomic<superslm_gpu::SslmSequenceGpuState> state_{superslm_gpu::SslmSequenceGpuState::Idle};
};


// The counting accessor (§4.4 rev 11; rev 15.2 for the pool pointer). A move to Submitted raises
// the process count, the handle's model's count, the pooled count and the pool's holder pointer
// for a pooled handle, and the adapter's count for a non-null `adapter` (recorded as the handle's
// `in_flight_adapter`), and records on the handle which it raised; a move out of Submitted lowers
// exactly those and clears `in_flight_adapter` and the pool pointer. A move between two
// non-Submitted states, or Submitted to Submitted, changes no count. A move out of Submitted
// re-registers the handle's `in_flight` record under the current outermost call (§4.4 rev 12.1,
// PR11-B1), so the walk at that call's return sees the handle no longer Submitted and drains and
// reports a record the caller left in it.
void SetGpuSeqState(SslmGpuSequenceHandle* seq, superslm_gpu::SslmSequenceGpuState next,
                    const SslmGpuAdapterHandle* adapter) {
	using superslm_gpu::SslmSequenceGpuState;
	const bool was_submitted = seq->state_.load() == SslmSequenceGpuState::Submitted;
	const bool now_submitted = next == SslmSequenceGpuState::Submitted;
	seq->state_.store(next);
	if (!was_submitted && now_submitted) {
		GpuSubmittedCounts::process_ += 1;
		seq->raised_process = true;
		if (seq->model != nullptr) {
			seq->model->submitted_sequences_ += 1;
			seq->raised_model = seq->model;
		}
		if (seq->pool != nullptr) {
			GpuSubmittedCounts::pooled_ += 1;
			seq->raised_pooled = true;
			// The process scope admits at most one Submitted pooled handle (§4 item 4), so the
			// pool records no other holder here (a G3 code-review item).
			assert(seq->pool->submitted_.load(std::memory_order_acquire) == nullptr);
			seq->pool->submitted_.store(seq, std::memory_order_release);
		}
		seq->in_flight_adapter = adapter;
		if (adapter != nullptr) {
			adapter->submitted_sequences_ += 1;
			seq->raised_adapter = adapter;
		}
	} else if (was_submitted && !now_submitted) {
		// The pool's holder pointer is cleared before any count drops (review-g23 F3): the counts
		// admit the next pooled submission, so a sibling the lowered counts admit never finds the
		// pointer still held.
		if (seq->raised_pooled && seq->pool != nullptr) {
			SslmGpuSequenceHandle* expected = seq;
			seq->pool->submitted_.compare_exchange_strong(expected, nullptr, std::memory_order_acq_rel);
		}
		if (seq->raised_process && GpuSubmittedCounts::process_ > 0) {
			GpuSubmittedCounts::process_ -= 1;
		}
		seq->raised_process = false;
		if (seq->raised_model != nullptr && seq->raised_model->submitted_sequences_ > 0) {
			seq->raised_model->submitted_sequences_ -= 1;
		}
		seq->raised_model = nullptr;
		if (seq->raised_pooled) {
			if (GpuSubmittedCounts::pooled_ > 0) GpuSubmittedCounts::pooled_ -= 1;
		}
		seq->raised_pooled = false;
		if (seq->raised_adapter != nullptr && seq->raised_adapter->submitted_sequences_ > 0) {
			seq->raised_adapter->submitted_sequences_ -= 1;
		}
		seq->raised_adapter = nullptr;
		seq->in_flight_adapter = nullptr;
		seq->in_flight.ReregisterOnLowering();
	}
}

#if defined(SUPERSLM_GPU_ALLOC_FAULT_INJECTION)
// Test seam (test builds only; tests/paged-kv-gpu/pkvg_same_handle_race.cpp): a hook run on the
// claiming thread each time a decode call takes a handle's same-sequence claim, while it holds it
// and before it reads or changes anything else on the handle. A race cell pauses the winner there
// and makes its losing calls inside the window, so the overlap is certain rather than likely.
using SslmGpuSeqClaimHookForTestFn = void (*)(SslmGpuSequenceHandle* seq, void* user);
namespace {
std::atomic<SslmGpuSeqClaimHookForTestFn> g_seq_claim_hook_for_test{nullptr};
std::atomic<void*> g_seq_claim_hook_user_for_test{nullptr};
}  // namespace
void SslmGpuSeqClaimHookForTest(SslmGpuSeqClaimHookForTestFn hook, void* user) noexcept {
	g_seq_claim_hook_user_for_test.store(user, std::memory_order_release);
	g_seq_claim_hook_for_test.store(hook, std::memory_order_release);
}
namespace {
void RunSeqClaimHookForTest(SslmGpuSequenceHandle* seq) {
	if (const SslmGpuSeqClaimHookForTestFn hook = g_seq_claim_hook_for_test.load(std::memory_order_acquire)) {
		hook(seq, g_seq_claim_hook_user_for_test.load(std::memory_order_acquire));
	}
}
}  // namespace
#endif

namespace {

// The handle whose same-sequence claim this thread holds, null when none: a decode entry point that
// calls another one on the same handle (the bridge decode drives sslm_decode_step_gpu and calls the
// finish) re-enters its own claim instead of refusing itself.
thread_local SslmGpuSequenceHandle* t_claimed_seq = nullptr;

// The same-sequence decode claim's scope (SslmGpuSequenceHandle::decode_claimed): held() is false
// when another thread's decode of the same handle holds it, and the caller then returns SSLM_BUSY
// (the single calls) or writes it to the slot (the batch) without touching the handle. Taken by
// every decode entry point -- sslm_decode_step_gpu, each batch slot, SslmGpuSeqDecodeStepForG5Bridge
// and SslmGpuSeqFinishTokenForG5Bridge -- before its first read of the handle's state, and held
// until that call returns, on every path (an exception unwinds through the destructor). A nested
// claim on the same thread and handle is held without a second exchange and released by the
// outermost one. The claim never outlives a call: a thread that submits (sslm_decode_step_gpu),
// drains (sslm_gpu_ready, which takes no claim) and then finishes takes it afresh at each decode
// call, and the Submitted state, not the claim, refuses other decode calls between the submit and
// the drain. A second sslm_gpu_ready on the same handle is the caller's to serialize (header).
class SeqDecodeClaim {
public:
	explicit SeqDecodeClaim(SslmGpuSequenceHandle* seq) noexcept
	    : seq_(seq),
	      nested_(t_claimed_seq == seq),
	      held_(nested_ || !seq->decode_claimed.exchange(true, std::memory_order_acq_rel)) {
		if (held_ && !nested_) {
			prev_ = t_claimed_seq;
			t_claimed_seq = seq_;
#if defined(SUPERSLM_GPU_ALLOC_FAULT_INJECTION)
			RunSeqClaimHookForTest(seq_);
#endif
		}
	}
	~SeqDecodeClaim() {
		if (held_ && !nested_) {
			t_claimed_seq = prev_;
			seq_->decode_claimed.store(false, std::memory_order_release);
		}
	}
	SeqDecodeClaim(const SeqDecodeClaim&) = delete;
	SeqDecodeClaim& operator=(const SeqDecodeClaim&) = delete;
	bool held() const noexcept { return held_; }

private:
	SslmGpuSequenceHandle* const seq_;
	const bool nested_;
	const bool held_;
	SslmGpuSequenceHandle* prev_ = nullptr;
};

// Paged KV (G3): the host buffer a sequence's K/V view addresses -- the pool's mirror for a pooled
// sequence, its own dedicated mirror otherwise -- and its size.
uint8_t* KvHostBase(SslmGpuSequenceHandle* seq) {
	return seq->pool != nullptr ? seq->pool->host_mirror.data() : seq->host_kv_mirror.data();
}
size_t KvHostBytes(const SslmGpuSequenceHandle* seq) {
	return seq->pool != nullptr ? seq->pool->host_mirror.size() : seq->host_kv_mirror.size();
}

// Paged KV (G3, §3.6's hot-path map): maps a pooled holder's pages through row `rows - 1` (clamped
// to the cap), from its own reserve, marking the table dirty when it maps one. No pool mutex, no
// allocation. SSLM_GPU_KV_POOL_EXHAUSTED for an empty reserve below the limit (unreachable by §3.4).
SslmGpuStatus MapPooledPagesThrough(SslmGpuSequenceHandle* seq, int64_t rows) {
	if (seq->pool == nullptr || seq->holder == nullptr) return SSLM_OK;
	if (rows > seq->context_cap) rows = seq->context_cap;
	const uint32_t need = superslm_gpu::kv_pool::PagesToCover(rows, seq->pool->geo.page_positions);
	while (superslm::kv_pages::HolderMapped(seq->holder) < need) {
		const superslm::kv_pages::PageStatus ps = superslm::kv_pages::HolderMapNext(seq->holder);
		if (ps != superslm::kv_pages::PageStatus::kOk) {
			return ps == superslm::kv_pages::PageStatus::kExhausted ? SSLM_GPU_KV_POOL_EXHAUSTED
			                                                        : SSLM_SEQUENCE_KV_BUFFER_MISMATCH;
		}
		seq->table_dirty = true;
	}
	return SSLM_OK;
}

// T-2113 (B2): synchronous single-buffer resident upload -- UPLOAD-heap temporary -> a
// fresh DEFAULT-heap copy, one command list, one submit, one fence wait. This is the same
// three-step shape RunLayerLoopGpu's own weight-residency miss path uses
// (src/gpu/superslm_gpu.cpp, the `!weights_resident` branch: Upload, MakeBuffer DEFAULT +
// CopyResource + barrier), performed here as its own self-contained call because
// sslm_gpu_model_map has no shared decode-step command list to append to -- it is not part
// of any `sslm_decode_step_gpu` recording window (design Sec5.1: "upload happens once,
// inside map, before the handle is returned to the caller"). Throws on any D3D12 failure,
// exactly like the substrate's own upload path: through SSLM_GPU_HR (GpuAllocationError for
// E_OUTOFMEMORY), or GpuSubmissionStrandedError for a list or submission it cannot confirm
// recovered. The caller (sslm_gpu_model_map and its siblings) catches this and classifies it
// (ClassifyHandleFault, 1.8.0): SSLM_GPU_ALLOCATION_FAILED for an allocation failure on a live
// device with the submission clean, SSLM_DEVICE_LOST otherwise -- design Sec5.1's original
// disposition ("on an upload/allocation failure, *out_model=nullptr and the call returns
// DeviceLost") for everything but memory, which it now reports as memory.
Microsoft::WRL::ComPtr<ID3D12Resource> UploadResidentBufferSyncTo(
    superslm_gpu::harness::Device& dev, const void* data, size_t bytes,
    D3D12_RESOURCE_FLAGS resource_flags, D3D12_RESOURCE_STATES final_state,
    const std::source_location at = std::source_location::current()) {
	// T-2851 (design Sec4.6 item 4, R16): phase A allocates both buffers BEFORE the command list
	// is reset, so an allocation failure throws with the list still Closed -- the state every call
	// on this context leaves it in -- and the next upload on the same context can reset it. (The
	// previous order reset first; a failed allocation then left the list recording and every later
	// upload on the context failed.) Since 1.8.0 the caller classifies the throw
	// (ClassifyHandleFault): an allocation failure on a live device is SSLM_GPU_ALLOCATION_FAILED.
	Microsoft::WRL::ComPtr<ID3D12Resource> upload = dev.Upload(data, bytes, at);
	Microsoft::WRL::ComPtr<ID3D12Resource> resident =
	    dev.MakeBuffer(bytes, D3D12_HEAP_TYPE_DEFAULT, resource_flags, D3D12_RESOURCE_STATE_COPY_DEST, at);
	// Phase B: reset, record, Close (confirmed, CloseListConfirmed), execute, Signal, wait. The
	// Signal has no retry here, so a failed Signal or wait is a stranded submission (1.8.0). From
	// G3 the reset, the submission and the wait are the ledger's (plan §4.4): BeginRecording(),
	// Submit(no retry), the submission's Wait().
	dev.BeginRecording();
	superslm_gpu::harness::RecordCopyResource(dev.list.Get(), resident.Get(), upload.Get());
	superslm_gpu::harness::RecordTransition(dev.list.Get(), resident.Get(), D3D12_RESOURCE_STATE_COPY_DEST, final_state);
	dev.CloseListOrThrow();
	dev.SubmitAndWaitKeepingAlive(/*retry=*/false, upload, resident);  // kept alive if it strands (F1)
	return resident;
}

// T-2113 (B2): synchronous single-buffer resident upload -- UPLOAD-heap temporary -> a
// fresh DEFAULT-heap copy, one command list, one submit, one fence wait. This is the same
// three-step shape RunLayerLoopGpu's own weight-residency miss path uses
// (src/gpu/superslm_gpu.cpp, the `!weights_resident` branch: Upload, MakeBuffer DEFAULT +
// CopyResource + barrier), performed here as its own self-contained call because
// sslm_gpu_model_map has no shared decode-step command list to append to -- it is not part
// of any `sslm_decode_step_gpu` recording window (design Sec5.1: "upload happens once,
// inside map, before the handle is returned to the caller"). Throws on any D3D12 failure,
// exactly like the substrate's own upload path: through SSLM_GPU_HR (GpuAllocationError for
// E_OUTOFMEMORY), or GpuSubmissionStrandedError for a list or submission it cannot confirm
// recovered. The caller (sslm_gpu_model_map and its siblings) catches this and classifies it
// (ClassifyHandleFault, 1.8.0): SSLM_GPU_ALLOCATION_FAILED for an allocation failure on a live
// device with the submission clean, SSLM_DEVICE_LOST otherwise -- design Sec5.1's original
// disposition ("on an upload/allocation failure, *out_model=nullptr and the call returns
// DeviceLost") for everything but memory, which it now reports as memory. Read-only for its
// whole lifetime (SRV state), matching `weights_buf`/`rope_cos_buf`/`rope_sin_buf`'s own
// contract (design Sec5.4).
Microsoft::WRL::ComPtr<ID3D12Resource> UploadResidentBufferSync(
    superslm_gpu::harness::Device& dev, const void* data, size_t bytes,
    const std::source_location at = std::source_location::current()) {
	return UploadResidentBufferSyncTo(dev, data, bytes, D3D12_RESOURCE_FLAG_NONE,
	                                   D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, at);
}

// T-2113 (B3, design Sec5.3): the K/V-buffer analogue of UploadResidentBufferSync above,
// differing in exactly the two respects a UAV a dispatch chain WRITES to (not merely
// reads) requires: the D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS resource flag, and a
// final state of UNORDERED_ACCESS rather than NON_PIXEL_SHADER_RESOURCE -- matching
// `MakeInitializedUav`'s own target state (superslm_gpu.cpp, internal linkage, not
// reachable from this TU), which is exactly the state RunLayerLoopGpu's own external-
// buffer path (gpu_port.h, T-2113 B3's own trailing-parameter pair) requires a freshly
// created sequence handle's kv_buf to already be in before its first call (no resume
// barrier needed on that first call, design Sec5.3/Sec10 B3).
Microsoft::WRL::ComPtr<ID3D12Resource> UploadResidentUavBufferSync(
    superslm_gpu::harness::Device& dev, const void* data, size_t bytes,
    const std::source_location at = std::source_location::current()) {
	return UploadResidentBufferSyncTo(dev, data, bytes, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
	                                   D3D12_RESOURCE_STATE_UNORDERED_ACCESS, at);
}

// ---------------------------------------------------------------------------------------------
// Paged-KV plan (rev 16.1) §4 item 3 (rev 15, rev 15.1), step G3: the pool's zero-fill, beside the
// upload primitive above and in its two phases, split into two functions so the boundary between
// them is the call, not a marker.
// ---------------------------------------------------------------------------------------------

// What phase A allocates and phase B uses: the UPLOAD staging buffer (min(bytes, 64 MiB), zeroed
// in place) and the pool's DEFAULT buffer (created in COPY_DEST).
struct ZeroFillBuffers {
	Microsoft::WRL::ComPtr<ID3D12Resource> staging;
	Microsoft::WRL::ComPtr<ID3D12Resource> resident;
	uint64_t bytes = 0;
};

// Phase A. Records and resets nothing, so the list stays Closed whatever throws here (no rule-0
// cause exists in it): one UPLOAD staging buffer of min(bytes, 64 MiB) -- never a full-size one,
// and no host zero source -- mapped and zeroed in place, then the pool's DEFAULT buffer. Both
// buffers go through Device::MakeBuffer, so through TryMakeBuffer and SSLM_GPU_HR, which logs a
// failing HRESULT with its file and line. The caller maps every throw from here as one block.
ZeroFillBuffers AllocateZeroFillBuffers(superslm_gpu::harness::Device& dev, uint64_t bytes) {
	ZeroFillBuffers b;
	b.bytes = bytes;
	const uint64_t staging_bytes = superslm_gpu::kv_pool::ZeroFillStagingBytes(bytes);
	b.staging = dev.MakeBuffer(staging_bytes, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_FLAG_NONE,
	                           D3D12_RESOURCE_STATE_GENERIC_READ);
	void* mapped = nullptr;
	D3D12_RANGE none{0, 0};
	SSLM_GPU_HR(b.staging->Map(0, &none, &mapped));
	std::memset(mapped, 0, static_cast<size_t>(staging_bytes));
	b.staging->Unmap(0, nullptr);
	b.resident = dev.MakeBuffer(bytes, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
	                            D3D12_RESOURCE_STATE_COPY_DEST);
	return b;
}

// Test builds only: marks phase B's run for the zero-fill Close seam
// (ArmGpuPoolZeroFillCloseFaultForTest). A no-op in product builds.
class PoolZeroFillSeamScope {
public:
	PoolZeroFillSeamScope() {
#if defined(SUPERSLM_GPU_ALLOC_FAULT_INJECTION)
		superslm_gpu::harness::GpuTestSeamState& seam = superslm_gpu::harness::TestSeamState();
		std::lock_guard<std::mutex> lock(seam.mutex);
		seam.pool_zero_fill_scope += 1;
#endif
	}
	~PoolZeroFillSeamScope() {
#if defined(SUPERSLM_GPU_ALLOC_FAULT_INJECTION)
		superslm_gpu::harness::GpuTestSeamState& seam = superslm_gpu::harness::TestSeamState();
		std::lock_guard<std::mutex> lock(seam.mutex);
		seam.pool_zero_fill_scope -= 1;
#endif
	}
	PoolZeroFillSeamScope(const PoolZeroFillSeamScope&) = delete;
	PoolZeroFillSeamScope& operator=(const PoolZeroFillSeamScope&) = delete;
};

// Phase B: one reset and one submission on the per-context device, waited before it returns.
// Records ceil(bytes / 64 MiB) CopyBufferRegion calls from the staging buffer, each chunk computed
// in place, then one barrier to UNORDERED_ACCESS (the state a pooled submission's first bind
// expects, as UploadResidentUavBufferSync leaves a dedicated buffer). INVARIANT (§4 item 3, rev
// 15.1): nothing between the list's Reset and CloseListOrThrow allocates or throws -- the window
// holds only the copies and the barrier, so a throw the caller classifies always comes from a
// reset (list still Closed), from CloseListOrThrow (which has confirmed the list's state) or from
// the submission tail. Throws as UploadResidentBufferSyncTo does.
void ZeroFillResidentUavBufferSync(superslm_gpu::harness::Device& dev, const ZeroFillBuffers& b) {
	PoolZeroFillSeamScope seam_scope;
	// The ledger's reset (G3, plan §4.4): BeginRecording() runs PoolSubmitAssert before it.
	dev.BeginRecording();
	// ---- no allocation and no throw from here to CloseListOrThrow ----
	const uint64_t chunks = superslm_gpu::kv_pool::ZeroFillChunkCount(b.bytes);
	for (uint64_t i = 0; i < chunks; ++i) {
		uint64_t offset = 0, len = 0;
		superslm_gpu::kv_pool::ZeroFillChunk(b.bytes, i, &offset, &len);
		superslm_gpu::harness::RecordCopyBufferRegion(dev.list.Get(), b.resident.Get(), offset, b.staging.Get(), 0, len);
	}
	superslm_gpu::harness::RecordTransition(dev.list.Get(), b.resident.Get(), D3D12_RESOURCE_STATE_COPY_DEST,
	                                        D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
	dev.CloseListOrThrow();
	// ---- end of the recording window ----
	// The ledger's submission and wait (G3): Submit(no retry), as before, then its Wait(); both
	// buffers kept alive if either strands (review-g23 F1).
	dev.SubmitAndWaitKeepingAlive(/*retry=*/false, b.staging, b.resident);
}

// ---------------------------------------------------------------------------------------------
// T-2851 (B): the device-resident head and its logits step (design Sec4.6).
// ---------------------------------------------------------------------------------------------

// Test builds only: marks the calls that run inside CreateDeviceLogitsBuffers (the allocation
// seam's in-bundle record) and inside either device logits routine (the phase-B Close seam's
// scope). A no-op in product builds.
class DeviceLogitsSeamScope {
public:
	explicit DeviceLogitsSeamScope(bool bundle_creation) : bundle_(bundle_creation) {
#if defined(SUPERSLM_GPU_ALLOC_FAULT_INJECTION)
		superslm_gpu::harness::GpuTestSeamState& seam = superslm_gpu::harness::TestSeamState();
		std::lock_guard<std::mutex> lock(seam.mutex);
		seam.device_logits_scope += 1;
		if (bundle_) seam.bundle_scope += 1;
#endif
	}
	~DeviceLogitsSeamScope() {
#if defined(SUPERSLM_GPU_ALLOC_FAULT_INJECTION)
		superslm_gpu::harness::GpuTestSeamState& seam = superslm_gpu::harness::TestSeamState();
		std::lock_guard<std::mutex> lock(seam.mutex);
		seam.device_logits_scope -= 1;
		if (bundle_) seam.bundle_scope -= 1;
#endif
	}
	DeviceLogitsSeamScope(const DeviceLogitsSeamScope&) = delete;
	DeviceLogitsSeamScope& operator=(const DeviceLogitsSeamScope&) = delete;

private:
	bool bundle_;
};

// Whether the context's device reports itself removed: the fault classifier's rule-1 input
// (harness::DeviceReportedRemoved, whose test-build seam, ArmGpuMapDeviceRemovedQueryInjection,
// makes the next query report removed). A device that was never set up is never read as removed.
bool MapDeviceReportedRemoved(superslm_gpu::harness::Device& dev) {
	return superslm_gpu::harness::DeviceReportedRemoved(dev);
}

// The public status for the exception currently being handled, at a site that creates or uploads
// into a handle (model map, adapter map, sequence create and restore, the device logits bundle).
// Call only from inside a catch handler. `dev` is the device the failed work ran on, or null for a
// host-only step. Rules 0 and 1 are SSLM_DEVICE_LOST, rule 2 SSLM_GPU_ALLOCATION_FAILED, and rule 3
// SSLM_DEVICE_LOST, these sites' non-allocation disposition.
SslmGpuStatus ClassifyHandleFault(superslm_gpu::harness::Device* dev) {
	using superslm_gpu::harness::GpuFaultKind;
	using superslm_gpu::harness::GpuFaultRule;
	const GpuFaultKind kind = superslm_gpu::harness::ClassifyInFlightException();
	const bool stranded = kind == GpuFaultKind::Stranded;
	const bool removed = !stranded && dev != nullptr && MapDeviceReportedRemoved(*dev);
	const GpuFaultRule rule = superslm_gpu::harness::ClassifyGpuFault(
	    stranded, removed, kind == GpuFaultKind::Allocation);
	return rule == GpuFaultRule::Allocation ? SSLM_GPU_ALLOCATION_FAILED : SSLM_DEVICE_LOST;
}

// Phase B's submission tail for the device logits routines: execute, then Signal. A failed Signal
// is retried once at a FRESH fence value, never the value the failed call attempted, as the chunk
// path does (gpu_1p0.h, SSLM_DEVICE_LOST cause (b)); if the retry succeeds the work is waited out
// before the throw, so no buffer is released while the GPU may still use it. A retry that was
// waited out throws the first Signal's typed HRESULT failure (rule 2 for E_OUTOFMEMORY, rule 3
// otherwise); a failed retry, or a wait that cannot be confirmed, throws
// GpuSubmissionStrandedError (rule 0). From G3 all of it is the ledger's (plan §4.4, *The ledger on
// the fault paths*): Submit() makes the fresh-value retry and, recovered, drains it before it
// rethrows the first failure; the submission's own Wait() is the success path's wait.
// `keep...` are the call's resources, kept alive if the submission strands (review-g23 F1).
template <class... T>
void ExecuteSignalAndWait(superslm_gpu::harness::Device& dev, const Microsoft::WRL::ComPtr<T>&... keep) {
	dev.SubmitAndWaitKeepingAlive(/*retry=*/true, keep...);
}

void TransitionBuffer(ID3D12GraphicsCommandList* list, ID3D12Resource* r, D3D12_RESOURCE_STATES before,
                D3D12_RESOURCE_STATES after, const std::source_location at = std::source_location::current()) {
	superslm_gpu::harness::RecordTransition(list, r, before, after, at);
}

// logits_site.hlsl's root signature: four root constants (hidden, vocab, groups_x, unused) at
// b0, the head at t0, the int32 x row at t1, the int64 output row at u0.
Microsoft::WRL::ComPtr<ID3D12RootSignature> MakeLogitsSiteRootSignature(ID3D12Device* device) {
	D3D12_ROOT_PARAMETER ps[4]{};
	ps[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
	ps[0].Constants.Num32BitValues = 4;
	ps[0].Constants.ShaderRegister = 0;
	ps[0].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
	ps[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_SRV;
	ps[1].Descriptor.ShaderRegister = 0;
	ps[2].ParameterType = D3D12_ROOT_PARAMETER_TYPE_SRV;
	ps[2].Descriptor.ShaderRegister = 1;
	ps[3].ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV;
	ps[3].Descriptor.ShaderRegister = 0;
	D3D12_ROOT_SIGNATURE_DESC rs{};
	rs.NumParameters = 4;
	rs.pParameters = ps;
	Microsoft::WRL::ComPtr<ID3DBlob> blob, err;
	if (FAILED(D3D12SerializeRootSignature(&rs, D3D_ROOT_SIGNATURE_VERSION_1, &blob, &err))) {
		if (err) std::fprintf(stderr, "%s\n", static_cast<const char*>(err->GetBufferPointer()));
		throw std::runtime_error("logits_site root signature serialization failed");
	}
	Microsoft::WRL::ComPtr<ID3D12RootSignature> root;
	SSLM_GPU_HR(device->CreateRootSignature(0, blob->GetBufferPointer(), blob->GetBufferSize(),
	                                        IID_PPV_ARGS(&root)));
	return root;
}

constexpr uint32_t kLogitsSiteRowsPerGroup = 8;  // 256 threads / 32 lanes per row (logits_site.hlsl)

// Creates the bundle for one model's head table (`head_rows`, vocab V x hidden H, row-major), in
// two phases so a failure never leaves the context's command list recording (design Sec4.6 item 4,
// R16):
// - Phase A, no command list touched: TryMakeBuffer for (i)-(vi) in order, each failure
//   classified at its own site; the head copied into (i); (iii) and (vi) persistently mapped; the
//   logits_site root signature and pipeline built and stored in the bundle.
// - Phase B: reset, record the head copy (ii) <- (i) and its transition, Close with one retry,
//   execute, Signal with a fresh-value retry, wait. (i) is released when this function returns.
// Returns SSLM_GPU_ALLOCATION_FAILED for E_OUTOFMEMORY on a device that is not removed, from any
// step (1.8.0), and SSLM_GPU_SHADER_BINARY_STALE for a stale logits_site.cso. Every other failure
// is SSLM_DEVICE_LOST (a missing .cso among them, as for every other shader load). On any failure
// `out` holds no usable bundle and the list is Closed, except in the documented terminal case
// where the list cannot be confirmed Closed (CloseListConfirmed), which is SSLM_DEVICE_LOST.
SslmGpuStatus CreateDeviceLogitsBuffers(SslmGpuContext* ctx, const int8_t* head_rows, uint32_t V,
                                        uint32_t H, DeviceLogitsBuffers* out) {
	DeviceLogitsSeamScope scope(/*bundle_creation=*/true);
	superslm_gpu::harness::Device& dev = ctx->device;
	try {
		out->V = V;
		out->H = H;
		// The shader reads the head a dword at a time, so the resident table is padded to a
		// multiple of four bytes (zero padding, never read into a row's sum).
		const UINT64 head_table_bytes = static_cast<UINT64>(V) * H;
		const UINT64 head_bytes = std::max<UINT64>(4, (head_table_bytes + 3) & ~UINT64{3});
		const UINT64 x_bytes = static_cast<UINT64>(H) * 4;
		const UINT64 row_bytes = static_cast<UINT64>(V) * 8;
		Microsoft::WRL::ComPtr<ID3D12Resource> staging;  // (i)
		struct Allocation {
			UINT64 bytes;
			D3D12_HEAP_TYPE heap;
			D3D12_RESOURCE_FLAGS flags;
			D3D12_RESOURCE_STATES state;
			Microsoft::WRL::ComPtr<ID3D12Resource>* target;
		};
		const Allocation allocations[] = {
		    {head_bytes, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_FLAG_NONE,
		     D3D12_RESOURCE_STATE_GENERIC_READ, &staging},
		    {head_bytes, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_FLAG_NONE,
		     D3D12_RESOURCE_STATE_COPY_DEST, &out->head},
		    {x_bytes, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_FLAG_NONE,
		     D3D12_RESOURCE_STATE_GENERIC_READ, &out->x_upload},
		    {x_bytes, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_FLAG_NONE,
		     D3D12_RESOURCE_STATE_COPY_DEST, &out->x},
		    {row_bytes, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
		     D3D12_RESOURCE_STATE_UNORDERED_ACCESS, &out->out},
		    {row_bytes, D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_FLAG_NONE,
		     D3D12_RESOURCE_STATE_COPY_DEST, &out->readback},
		};
		for (const Allocation& a : allocations) {
			const HRESULT hr = dev.TryMakeBuffer(a.bytes, a.heap, a.flags, a.state, a.target);
			if (FAILED(hr)) {
				if (hr == E_OUTOFMEMORY && !MapDeviceReportedRemoved(dev)) {
					return SSLM_GPU_ALLOCATION_FAILED;
				}
				std::fprintf(stderr, "superslm_gpu: device logits buffer allocation failed 0x%08lx\n",
				             static_cast<unsigned long>(hr));
				return SSLM_DEVICE_LOST;
			}
		}

		void* mapped = nullptr;
		D3D12_RANGE no_read{0, 0};
		SSLM_GPU_HR(staging->Map(0, &no_read, &mapped));
		std::memcpy(mapped, head_rows, static_cast<size_t>(head_table_bytes));
		std::memset(static_cast<uint8_t*>(mapped) + head_table_bytes, 0,
		            static_cast<size_t>(head_bytes - head_table_bytes));
		staging->Unmap(0, nullptr);
		SSLM_GPU_HR(out->x_upload->Map(0, &no_read, &mapped));
		out->x_upload_ptr = static_cast<int32_t*>(mapped);
		SSLM_GPU_HR(out->readback->Map(0, nullptr, &mapped));
		out->readback_ptr = static_cast<const int64_t*>(mapped);

		std::string cso_path;
		try {
			cso_path = superslm_gpu::harness::ShaderPath("logits_site");
		} catch (const std::logic_error&) {
			// ShaderPath's only logic_error is its stale-binary refusal (GpuShaderBinaryStaleError,
			// superslm_gpu.cpp), whose diagnostic it has already printed.
			return SSLM_GPU_SHADER_BINARY_STALE;
		}
		const std::vector<uint8_t> cso = superslm_gpu::harness::ReadFile(cso_path);
		out->root = MakeLogitsSiteRootSignature(dev.dev.Get());
		out->pso = dev.MakePSO(out->root.Get(), cso);
		const uint32_t groups = (V + kLogitsSiteRowsPerGroup - 1) / kLogitsSiteRowsPerGroup;
		out->groups_x = std::min<uint32_t>(groups, 65535u);
		out->groups_y = (groups + out->groups_x - 1) / out->groups_x;

		// Phase B. The ledger's reset (G3, plan §4.4).
		dev.BeginRecording();
		superslm_gpu::harness::RecordCopyResource(dev.list.Get(), out->head.Get(), staging.Get());
		TransitionBuffer(dev.list.Get(), out->head.Get(), D3D12_RESOURCE_STATE_COPY_DEST,
		           D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
		dev.CloseListOrThrow();
		ExecuteSignalAndWait(dev, staging, out->head, out->x_upload, out->x, out->out, out->readback);
		return SSLM_OK;
	} catch (const std::bad_alloc&) {
		throw;
	} catch (const std::exception&) {
		// 1.8.0 (TE-426): the fault classifier. Phase B's Close (CloseListOrThrow) and Signal
		// (ExecuteSignalAndWait) carry the stranded input; E_OUTOFMEMORY from any D3D12 call on a
		// live device is SSLM_GPU_ALLOCATION_FAILED, and every other failure SSLM_DEVICE_LOST.
		return ClassifyHandleFault(&dev);
	} catch (...) {
		return ClassifyHandleFault(&dev);  // 1.8.0 (TE-435): every other exception type, as above
	}
}

#if defined(SUPERSLM_GPU_ALLOC_FAULT_INJECTION)
void ApplyReadbackOverrideSeam(int64_t* wide_out, uint32_t V) {
	superslm_gpu::harness::GpuTestSeamState& seam = superslm_gpu::harness::TestSeamState();
	std::lock_guard<std::mutex> lock(seam.mutex);
	if (!seam.readback_override_armed) return;
	seam.readback_override_armed = false;  // single-shot
	if (seam.readback_override_row >= 0 && static_cast<uint32_t>(seam.readback_override_row) < V) {
		wide_out[seam.readback_override_row] = seam.readback_override_value;
	}
}
#endif

// One device logits dispatch through the bundle: the exact int64 row for `x_codes` (H codes) into
// `wide_out` (V elements). Allocates nothing and maps nothing: the codes are written through the
// persistent (iii) mapping before the list is reset, and the row is copied out of the persistent
// (vi) mapping after the fence wait. Its only fallible steps are Reset, Close and Signal, with
// the phase-B dispositions, classified (1.8.0): E_OUTOFMEMORY on a live device with the submission
// clean returns SSLM_GPU_ALLOCATION_FAILED; every other failure SSLM_DEVICE_LOST.
SslmGpuStatus RunDeviceLogits(SslmGpuContext* ctx, const DeviceLogitsBuffers& b,
                              const int8_t* x_codes, int64_t* wide_out) {
	DeviceLogitsSeamScope scope(/*bundle_creation=*/false);
	superslm_gpu::harness::Device& dev = ctx->device;
	try {
		for (uint32_t k = 0; k < b.H; ++k) b.x_upload_ptr[k] = x_codes[k];
		dev.BeginRecording();  // the ledger's reset (G3, plan §4.4); the PSO is set right after it
		dev.list->SetPipelineState(b.pso.Get());
		ID3D12GraphicsCommandList* list = dev.list.Get();
		superslm_gpu::harness::RecordCopyBufferRegion(list, b.x.Get(), 0, b.x_upload.Get(), 0, static_cast<UINT64>(b.H) * 4);
		TransitionBuffer(list, b.x.Get(), D3D12_RESOURCE_STATE_COPY_DEST,
		           D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
		list->SetComputeRootSignature(b.root.Get());
		const uint32_t constants[4] = {b.H, b.V, b.groups_x, 0u};
		list->SetComputeRoot32BitConstants(0, 4, constants, 0);
		superslm_gpu::harness::BindRootSrv(list, 1, b.head.Get());
		superslm_gpu::harness::BindRootSrv(list, 2, b.x.Get());
		superslm_gpu::harness::BindRootUav(list, 3, b.out.Get());
		superslm_gpu::harness::RecordDispatch(list, b.groups_x, b.groups_y, 1);
		TransitionBuffer(list, b.out.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
		           D3D12_RESOURCE_STATE_COPY_SOURCE);
		superslm_gpu::harness::RecordCopyBufferRegion(list, b.readback.Get(), 0, b.out.Get(), 0, static_cast<UINT64>(b.V) * 8);
		// Back to the resting states the next dispatch starts from.
		TransitionBuffer(list, b.out.Get(), D3D12_RESOURCE_STATE_COPY_SOURCE,
		           D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
		TransitionBuffer(list, b.x.Get(), D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
		           D3D12_RESOURCE_STATE_COPY_DEST);
		dev.CloseListOrThrow();
		ExecuteSignalAndWait(dev, b.head, b.x_upload, b.x, b.out, b.readback);
		std::memcpy(wide_out, b.readback_ptr, static_cast<size_t>(b.V) * 8);
#if defined(SUPERSLM_GPU_ALLOC_FAULT_INJECTION)
		ApplyReadbackOverrideSeam(wide_out, b.V);
#endif
		return SSLM_OK;
	} catch (const std::bad_alloc&) {
		throw;
	} catch (const std::exception&) {
		// 1.8.0 (TE-426): the fault classifier, as in CreateDeviceLogitsBuffers.
		return ClassifyHandleFault(&dev);
	} catch (...) {
		return ClassifyHandleFault(&dev);  // 1.8.0 (TE-435): every other exception type, as above
	}
}

}  // namespace

// Design Sec5.1: "uploads the model's weights, fold-scale tables, and RoPE cos/sin tables
// ... into DEFAULT-heap buffers owned by the returned handle, keyed by the artifact's own
// content hash." Marshals `*base` into a real LayerWeights[] (superslm_marshal::MarshalLayer,
// T-2113 B2's own relocation of tools/sslm_marshal.h -> include/superslm/layer_marshal.h --
// the first production call site for this logic), packs it via the shared
// PackLayerWeightsBytes (T-2113 B2, extracted from RunLayerLoopGpu's own miss path so both
// the pre-1.0 substrate and this handle's upload path compute identical bytes from one
// implementation), and uploads weights + RoPE cos/sin tables as three independent resident
// buffers -- no process-global cache anywhere in this path (D-SLM3294).
SslmGpuStatus sslm_gpu_model_mapImpl(SslmGpuContext* ctx, const SslmModelView* base,
                                  GpuResidencyConfig cfg, SslmGpuModelHandle** out_model) {
	if (!out_model) {
		return SSLM_DEVICE_LOST;  // same "no live object to report through" reasoning as
		                          // sslm_gpu_context_create's own null-out-parameter case.
	}
	*out_model = nullptr;
	if (!ctx || !base) {
		return SSLM_DEVICE_LOST;
	}
	// T-2851 (design Sec4.6): a bit this release does not define is refused before any work.
	if ((cfg.flags & ~SSLM_GPU_RESIDENCY_HEAD_ON_DEVICE) != 0) {
		return SSLM_GPU_RESIDENCY_FLAGS_INVALID;
	}
	// EntrySubmitGuard (§4.4, rev 8 placement): the map uploads on the context's own device only
	// (no ledger term); an existing entry point on a model, so its scope is a pooled sequence
	// Submitted. After the flags refusal, before the marshal: the marshal and layout refusals
	// follow it (§4.4, *Status precedence*).
	if (const SslmGpuStatus g = EntrySubmitGuard(GuardScope::kPooled, GuardDevice::kPerContextOnly);
	    g != SSLM_OK) {
		return g;
	}
	const bool head_on_device = (cfg.flags & SSLM_GPU_RESIDENCY_HEAD_ON_DEVICE) != 0;

	const uint32_t num_heads = base->config.num_attention_heads;
	const uint32_t num_kv_heads = base->config.num_key_value_heads;
	const uint32_t num_hidden_layers = base->config.num_hidden_layers;
	const uint32_t H = static_cast<uint32_t>(base->config.hidden_size);
	const uint32_t HD = static_cast<uint32_t>(base->config.head_dim);
	const uint32_t KV = num_kv_heads * HD;
	const uint32_t I = static_cast<uint32_t>(base->config.intermediate_size);
	const uint32_t NQH = num_heads;
	// T-2432 (Track A step 5/7): q_width = num_attention_heads * head_dim, threaded into the
	// layout/pack calls below so this handle's own resident weight buffer is sized and packed
	// at Q's own real (possibly non-square) width.
	const uint32_t QWIDTH = NQH * HD;

	// Marshal every layer's LayerWeights (superslm_marshal::MarshalLayer, the identical
	// call tools/t2100_gpu_throughput.cpp's own T-2100 harness makes before calling
	// RunLayerLoopGpu). A malformed artifact (a missing required tensor) fails here; Sec9's
	// own table assigns model-map failure only DeviceLost (the upload/allocation
	// disposition) -- there is no separate status for "the artifact's own content could not
	// be marshaled," so a marshal failure takes that same disposition rather than this
	// build inventing a status the design's own Sec9 table does not name.
	std::vector<superslm_marshal::LayerBacking> backings(num_hidden_layers);
	std::vector<superslm::LayerWeights> layers(num_hidden_layers);
	for (uint32_t l = 0; l < num_hidden_layers; ++l) {
		std::string marshal_err;
		if (!superslm_marshal::MarshalLayer(*base, l, num_heads, num_kv_heads, backings[l], layers[l],
		                                     &marshal_err)) {
			// T-2570 (M5, Claude/Poirot/cd12179-t2569-trackb-confirmation.md): MarshalLayer
			// names the missing or malformed manifest key on every rejecting path
			// (layer_marshal.h); this is the RECEIVED path -- an artifact that fails to
			// marshal today, unlike the throwing contract violations the try/catch below
			// guards against, which no in-tree producer can reach. Preserve the diagnostic
			// to stderr, matching the same-function catch clause's own convention 28 lines
			// below, rather than discarding it and returning a bare SSLM_DEVICE_LOST.
			std::fprintf(stderr, "sslm_gpu_model_map: layer %u: %s\n", l, marshal_err.c_str());
			return SSLM_DEVICE_LOST;
		}
	}
	const bool has_qk_norm = std::any_of(layers.begin(), layers.end(), [](const auto& layer) {
		return layer.q_norm_gain != nullptr || layer.k_norm_gain != nullptr;
	});

	const superslm_gpu::GpuLayerLayout layout =
	    superslm_gpu::ComputeLayerLayout(H, KV, num_kv_heads, NQH, I, QWIDTH);
	// T-2568 (S1, Claude/Poirot/66626ef-t2567-trackb-confirmation.md): PackLayerWeightsBytes
	// throws `GpuLayerWeightsContractError` (superslm_gpu.cpp, std::logic_error-derived) when a
	// layer's k_norm_gain or iexp_softmax_khead_m/e pointer contract is violated. RunLayerLoopGpu
	// (this file's own sibling call site, superslm_gpu.cpp) contains that throw inside its own
	// try/catch and converts it to a status; this call site did not -- an uncaught C++ exception
	// escaping THIS function past the C ABI boundary above it, contradicting this header's own
	// second sentence ("every fallible call returns an SslmGpuStatus"). Contained here, the same
	// way: caught, the field's name preserved to stderr rather than discarded, and refused as a
	// status -- `SSLM_DEVICE_LOST`, the SAME disposition every other marshal failure in this
	// function already uses (the missing-tensor/malformed-artifact checks above), since design
	// Sec9 assigns no more specific 1.0 status to "the artifact's own content could not be
	// marshaled" than that one. Unreachable through any artifact today -- MarshalLayer
	// (layer_marshal.h) is the only in-tree producer of `layers` and already rejects an artifact
	// whose k_norm_gain is present without its landing pair, or whose composition_constants
	// lacks a softmax_khead entry, before this call is ever reached (the identical reachability
	// argument M4's own remedy rested on for RunLayerLoopGpu's twin path) -- but an uncaught
	// exception crossing a status-return ABI boundary is a defect independent of today's
	// reachability, not merely a defect while it happens to be unreachable.
	std::vector<uint8_t> lw_bytes;
	try {
		lw_bytes = superslm_gpu::PackLayerWeightsBytes(layers.data(), num_hidden_layers, layout, H,
		                                               KV, num_kv_heads, NQH, I, QWIDTH);
	} catch (const std::bad_alloc&) {
		throw;
	} catch (const std::exception& e) {
		std::fprintf(stderr, "sslm_gpu_model_map: %s\n", e.what());
		// 1.8.0 (TE-426): a host step, no device. std::length_error is an allocation failure,
		// SSLM_GPU_ALLOCATION_FAILED; the layer-weights contract error stays SSLM_DEVICE_LOST.
		return ClassifyHandleFault(nullptr);
	} catch (...) {
		return ClassifyHandleFault(nullptr);  // 1.8.0 (TE-435): every other exception type, as above
	}

	// T-2105's own RoPE cos/sin residency construction (Claude/Laplace/
	// t2105-gpu-speed-ceiling-2026-08-14.md Sec2 change 1), re-derived here per design Sec1's
	// own "re-derived, never cherry-picked" instruction: model-wide constants, read directly
	// from the artifact's own rope_tables manifest, exactly as RunLayerLoopGpu's own
	// `cos_t`/`sin_t` lookup does (superslm_gpu.cpp).
	const superslm::SslmTensorView* cos_t = base->rope_tables.Tensor("cos");
	const superslm::SslmTensorView* sin_t = base->rope_tables.Tensor("sin");
	const bool has_rope = cos_t != nullptr && sin_t != nullptr;
	const uint64_t cos_need = cos_t != nullptr ? static_cast<uint64_t>(cos_t->elem_count) * 8u : 8u;
	const uint64_t sin_need = sin_t != nullptr ? static_cast<uint64_t>(sin_t->elem_count) * 8u : 8u;
	std::vector<uint8_t> cos_bytes(static_cast<size_t>(cos_need), 0);
	if (cos_t != nullptr) std::memcpy(cos_bytes.data(), cos_t->data, cos_bytes.size());
	std::vector<uint8_t> sin_bytes(static_cast<size_t>(sin_need), 0);
	if (sin_t != nullptr) std::memcpy(sin_bytes.data(), sin_t->data, sin_bytes.size());

	// T-2113 (B3.5, design Sec5.1's own amendment / Sec5.3a): the embedding matrix, its
	// site constant, and vocab_size -- read from `*base` exactly where sslm_generate.cpp's
	// own precedent reads them (`model_view.weights.Tensor("embed")`,
	// `ReadCarriedScale(model_view.composition_constants, "embed", &ok)`,
	// `model_view.config.vocab_size`). A malformed artifact missing either takes the same
	// "no more specific status than DeviceLost" disposition every other marshal failure in
	// this function already uses (the missing-tensor case above).
	const superslm::SslmTensorView* embed_w = base->weights.Tensor("embed");
	if (!embed_w) {
		return SSLM_DEVICE_LOST;
	}
	bool embed_scale_ok = true;
	const superslm::CarriedScale embed_site_constant =
	    superslm_marshal::ReadCarriedScale(base->composition_constants, "embed", &embed_scale_ok);
	if (!embed_scale_ok) {
		return SSLM_DEVICE_LOST;
	}
	const int32_t vocab_size = base->config.vocab_size;
	const size_t embed_bytes_needed = static_cast<size_t>(vocab_size) * static_cast<size_t>(H);
	// T-2114 (S3, Claude/Poirot/50f3d5d-t2113-1p0-gpu-core-build-review.md): bound the copy
	// below against the tensor's own declared extent -- the SAME check `cos_need`/`sin_need`
	// above already apply to the RoPE tables (`cos_t->elem_count`/`sin_t->elem_count`), which
	// this one used to skip. `embed_w->data` is `elem_count` int8 bytes (element_size 1 for
	// this dtype); a short `embed` section previously read `embed_bytes_needed` bytes from it
	// regardless, a multi-megabyte out-of-bounds read at map time on a malformed artifact --
	// the CPU precedent this line is derived from (sslm_generate.cpp:287) never takes this
	// risk because it reads one row at a time, bounded by `token_id < vocab_size`, never the
	// whole matrix eagerly.
	if (embed_w->elem_count < static_cast<uint64_t>(embed_bytes_needed)) {
		return SSLM_DEVICE_LOST;  // same "no more specific status than DeviceLost" disposition
		                          // every other marshal failure in this function already uses.
	}

	// G5-5 (T-2132, Brunel): final_norm.gain + head (lm_head or tied embed) -- the SAME two
	// tensor lookups sslm_abi.cpp's own BuildEngineCache already runs (src/sslm_abi.cpp), read
	// here directly from `*base` since this TU has no engine-cache object of its own. Required
	// for the finish-token bridge's own final_norm+logits step; a malformed artifact missing
	// either takes the same "no more specific status than DeviceLost" disposition every other
	// marshal failure in this function already uses.
	const superslm::SslmTensorView* final_gain_w = base->weights.Tensor("final_norm.gain");
	if (!final_gain_w) {
		return SSLM_DEVICE_LOST;
	}
	bool final_norm_scale_ok = true;
	const superslm::CarriedScale final_norm_site_constant = superslm_marshal::ReadCarriedScale(
	    base->composition_constants, "final_norm", &final_norm_scale_ok);
	if (!final_norm_scale_ok) {
		return SSLM_DEVICE_LOST;
	}
	const superslm::SslmTensorView* head_w = nullptr;
	if (base->config.tie_word_embeddings) {
		head_w = embed_w;  // tied embeddings: the head IS the embedding matrix, CPU precedent.
	} else {
		head_w = base->weights.Tensor("lm_head");
		if (!head_w) {
			return SSLM_DEVICE_LOST;
		}
	}
	const size_t head_bytes_needed = static_cast<size_t>(vocab_size) * static_cast<size_t>(H);
	if (head_w->elem_count < static_cast<uint64_t>(head_bytes_needed)) {
		return SSLM_DEVICE_LOST;  // same bounded-copy discipline the embed check above applies.
	}

	// G5-5 (T-2132, Brunel): this artifact's own SchemaMasks/SCM1 section, if present -- absence
	// is a valid, unconstrained-only artifact (design Sec13.1), the SAME disposition the CPU
	// ABI's own sslm_model_map already applies (src/sslm_abi.cpp). Parsed against a HOST-OWNED
	// copy of the section bytes (schema_section_bytes, below) rather than `base`'s own pointer,
	// since `base`'s own lifetime is caller-owned and this handle must serve every schema-bound
	// call for its OWN whole lifetime -- the identical caller-view-lifetime reasoning B3.5's own
	// embed_weights capture (this function, above) already documents.
	const superslm::SslmSectionView* schema_section =
	    base->Section(superslm::SslmSectionType::SchemaMasks);
	std::vector<uint8_t> schema_section_bytes;
	if (schema_section) {
		schema_section_bytes.assign(schema_section->data,
		                             schema_section->data + schema_section->byte_size);
	}

	std::unique_ptr<SslmGpuModelHandle> h(new SslmGpuModelHandle());
	try {
		h->weights_buf = UploadResidentBufferSync(ctx->device, lw_bytes.data(), lw_bytes.size());
		h->rope_cos_buf = UploadResidentBufferSync(ctx->device, cos_bytes.data(), cos_bytes.size());
		h->rope_sin_buf = UploadResidentBufferSync(ctx->device, sin_bytes.data(), sin_bytes.size());
		if (!schema_section_bytes.empty()) {
			// G5-5's own "Builds: mask-page residency on device" (design Sec6 G5-5, VRAM,
			// alongside weights/scales per the existing GPU-tenancy model, Sec8.4) -- the whole
			// SCM1 section, uploaded read-only, the SAME UploadResidentBufferSync three-step
			// shape weights/rope use immediately above.
			h->mask_pages_buf = UploadResidentBufferSync(ctx->device, schema_section_bytes.data(),
			                                              schema_section_bytes.size());
		}
	} catch (const std::bad_alloc&) {
		throw;
	} catch (const std::exception&) {
		// 1.8.0 (TE-426): the fault classifier. An allocation failure (E_OUTOFMEMORY on any heap,
		// or at any step) on a live device, with the list confirmed Closed, is
		// SSLM_GPU_ALLOCATION_FAILED; every other failure SSLM_DEVICE_LOST.
		return ClassifyHandleFault(&ctx->device);
	} catch (...) {
		return ClassifyHandleFault(&ctx->device);  // 1.8.0 (TE-435): every other exception type, as above
	}

	// T-2851 (B, design Sec4.6): the device-resident head. `head_w` above is the one table the
	// host path would read (the tied embedding, or lm_head), selected once; the upload reads that
	// pointer and nothing else. A failure leaves no bundle and no handle (the handle's buffers
	// above are released with `h`).
	if (head_on_device) {
		auto bundle = std::make_unique<DeviceLogitsBuffers>();
		const SslmGpuStatus bundle_status = CreateDeviceLogitsBuffers(
		    ctx, reinterpret_cast<const int8_t*>(head_w->data), static_cast<uint32_t>(vocab_size),
		    H, bundle.get());
		if (bundle_status != SSLM_OK) return bundle_status;
		h->device_logits = std::move(bundle);
	}

	// Host-only copy (Sec5.3a: "not new residency, no additional GPU upload") -- taken
	// after the three real device uploads above have already succeeded, so a failure here
	// never leaves a half-uploaded device handle: this is a plain host memcpy that cannot
	// itself throw a D3D12 exception, so it is outside the try/catch above by construction.
	h->embed_weights.assign(reinterpret_cast<const int8_t*>(embed_w->data),
	                         reinterpret_cast<const int8_t*>(embed_w->data) + embed_bytes_needed);
	h->embed_site_constant = embed_site_constant;
	h->vocab_size = vocab_size;
	h->final_norm_gain = superslm_marshal::WidenGainToInt32(*final_gain_w);
	h->final_norm_site_constant = final_norm_site_constant;
	// T-2851 (design Sec4.6a): one host copy of a tied head. A tied artifact's head IS the
	// embedding, which `embed_weights` already holds, so the finish reads that copy; only an
	// untied artifact's lm_head is copied, and only when the host path can run for this model.
	if (head_on_device) {
		h->head_rows = nullptr;
	} else if (base->config.tie_word_embeddings) {
		h->head_rows = h->embed_weights.data();
	} else {
		h->head_weights.assign(reinterpret_cast<const int8_t*>(head_w->data),
		                        reinterpret_cast<const int8_t*>(head_w->data) + head_bytes_needed);
		h->head_rows = h->head_weights.data();
	}

	// G5-5: parse the host-owned copy (schema_section_bytes, moved into the handle here) --
	// `schemas`'s own SchemaEntry pointers point INTO `h->schema_section_bytes`, never the local
	// `schema_section_bytes`/`base`'s own view, so they stay valid for this handle's whole
	// lifetime. A structurally-invalid SchemaMasks section is a malformed artifact -- same
	// "no more specific status than DeviceLost" disposition every other marshal failure in this
	// function already uses (mirrors sslm_abi.cpp's own SSLM_ARTIFACT_REJECTED-for-any-Sec13.3-
	// violation collapse, translated to this file's own DeviceLost convention).
	if (!schema_section_bytes.empty()) {
		h->schema_section_bytes = std::move(schema_section_bytes);
		std::string schema_err;
		if (!superslm::SchemaMasksTable::Parse(h->schema_section_bytes.data(),
		                                        h->schema_section_bytes.size(),
		                                        static_cast<uint32_t>(vocab_size), h->schemas,
		                                        &schema_err)) {
			return SSLM_DEVICE_LOST;
		}
	}

	h->ctx = ctx;
	h->content_hash = base->RawIntegrityHash();
	h->kv_precision = static_cast<uint32_t>(base->config.kv_precision);
	h->has_rope_tables = has_rope;
	h->rope_cos_elem_count = cos_t != nullptr ? cos_t->elem_count : 0;
	h->rope_sin_elem_count = sin_t != nullptr ? sin_t->elem_count : 0;
	h->layout = layout;
	h->num_hidden_layers = num_hidden_layers;
	h->hidden_size = H;
	h->head_dim = HD;
	h->num_key_value_heads = num_kv_heads;
	h->num_attention_heads = NQH;
	h->intermediate_size = I;
	h->context_cap = static_cast<int64_t>(base->config.context_cap);
	h->kv_block_size = base->config.kv_block_size;
	h->has_qk_norm = has_qk_norm;
	h->dispatches_per_layer = superslm_gpu::DispatchesPerLayer(has_qk_norm);

	ctx->live_handles += 1;
	*out_model = h.release();
	return SSLM_OK;
}

// Design Sec5.1: "frees the residency and returns Ok. Freeing a model handle with live
// sequences bound to it is a caller error: returns ModelHasLiveSequences ... and does NOT
// free the residency." T-2113 (B3): `live_sequences` is now incremented/decremented by
// real `SslmGpuSequenceHandle`s (sslm_gpu_seq_create/release, below) -- this guard reads
// live values as of this section.
//
// T-2113 (design Sec4.2/Sec9, routed `Claude/Curie/t2112-1p0-red-suite-2026-08-15.md` Sec7.2
// item 2 / Sec7.3, D-SLM3417): design Sec9's own Busy row states this call returns `Busy`
// "against a model with any Submitted sequence bound... which takes precedence" over
// `ModelHasLiveSequences`' own "any state, not only Submitted" caveat -- the guard below used
// to check only `live_sequences` and could not distinguish a Submitted-bound model from an
// Idle-bound one, always returning `ModelHasLiveSequences` for both. Fixed: `submitted_sequences`
// (SslmGpuModelHandle's own field, above) is checked FIRST, so a model with at least one
// Submitted sequence bound returns `Busy`; only once that count is zero does the broader
// `live_sequences` (any state) check run.
SslmGpuStatus sslm_gpu_model_unmapImpl(SslmGpuContext* ctx, SslmGpuModelHandle* model) {
	if (!model) {
		return SSLM_OK;  // same null-is-a-no-op reasoning as sslm_gpu_context_destroy(nullptr).
	}
	// T-2124 (D-SLM3446 P1-4): validate this handle was actually mapped against `ctx` -- the
	// external-review gap: this function used to ignore `ctx` entirely (a bare `(void)ctx`, now
	// removed since the parameter is read below and by the decrement further down) and act purely
	// off `model->ctx`, so a caller passing a DIFFERENT (but live) context here was never rejected
	// -- a sequence could be created on one context against a model mapped on another, binding
	// resources from different D3D12 devices (see sslm_gpu_seq_create's own identical fix for the
	// sequence-creation half of this same gap).
	if (model->ctx != ctx) {
		return SSLM_DEVICE_LOST;  // same "no more specific status" disposition sslm_gpu_model_map
		                          // already uses for a malformed/foreign-context argument.
		                          // P1-6 (external review, status-ABI collapse to DEVICE_LOST):
		                          // this rejects a PERMANENT caller-contract violation (a foreign
		                          // handle never becomes valid by retrying) through a status §9
		                          // classifies transient/recoverable -- routing left as-is per
		                          // Claude/Poirot/435f730-t2124-adapter-uaf-review.md M2 (a new
		                          // enumerator is out of this ticket's scope), flagged here so the
		                          // P1-6 taxonomy pass (S-FREEZE) finds this site.
	}
	if (model->submitted_sequences() > 0) {
		return SSLM_BUSY;  // design Sec9's own Busy-precedence, D-SLM3417.
	}
	if (model->live_sequences > 0) {
		return SSLM_MODEL_HAS_LIVE_SEQUENCES;
	}
	// Paged KV (G3): a GPU page pool of this model still lives. The plan states no status for it;
	// a pool is where this model's pooled sequences live, so the live-sequences status is reused.
	if (model->live_pools > 0) {
		return SSLM_MODEL_HAS_LIVE_SEQUENCES;
	}
	// T-2243 (M2, design plan Sec6.1/Sec10 Phase 2 M2, D-SLM3965): an adapter handle retains a
	// raw `model` pointer (SslmGpuAdapterHandle::model) that is never dereferenced today but
	// must not be left dangling -- persistent-liveness, the same reasoning `live_sequences`
	// above already applies to a bound sequence. Not `SSLM_BUSY`: this condition never drains on
	// its own, it holds until every mapped adapter is explicitly unmapped.
	if (model->live_adapters > 0) {
		return SSLM_MODEL_HAS_LIVE_ADAPTERS;
	}
	// T-2114 (M3, Claude/Poirot/50f3d5d-t2113-1p0-gpu-core-build-review.md): decrement the
	// handle's OWN stored context (`model->ctx`, set once at map() time) -- now provably equal to
	// `ctx` by the check above too, kept as `model->ctx` unchanged since every handle already
	// carries the context it was created against for exactly this reason (design Sec5.1's own
	// per-handle ownership).
	DecrementLiveCount(model->ctx->live_handles);
	model->destroyed = true;
	delete model;
	return SSLM_OK;
}

// Design Sec5.2/Sec10 B6: "mirrors the CPU-side LoadAdapterArtifact/AdapterHandle shape exactly,
// on the device side: validates the adapter's base-hash against model's own content hash ...
// and uploads the per-layer, per-projection lora_A/lora_B weights plus DeltaFoldScales/
// UFoldScales tables into their own DEFAULT-heap buffers." Reuses
// superslm_adapter::PopulateAdapterFromView (include/superslm/adapter_marshal.h -- the body of
// LoadAdapterArtifact that runs after SslmModel::Load, extracted this section, D-SLM3368, since
// this call already holds a parsed view and must not re-read/re-parse the artifact's own bytes
// from a path a view does not retain) for the parse/validate half -- the identical B0b checks
// (ValidateAmplifyingFoldBaseHash/Dimension/Projection) the CPU path already runs, run here for
// real against a real converted artifact for the first time on the GPU path.
SslmGpuStatus sslm_gpu_adapter_mapImpl(SslmGpuContext* ctx, SslmGpuModelHandle* model,
                                    const SslmModelView* adapter_artifact,
                                    SslmGpuAdapterHandle** out_adapter) {
	if (!out_adapter) {
		return SSLM_DEVICE_LOST;  // same "no live object to report through" reasoning as
		                          // sslm_gpu_model_map's own null-out-parameter case.
	}
	*out_adapter = nullptr;
	if (!ctx || !model || !adapter_artifact) {  // T-2114 (S2): the ->destroyed read this line
	                                             // used to carry was dead on a live handle and
	                                             // UB on a released one -- removed, see
	                                             // SslmGpuModelHandle's own field comment.
		return SSLM_DEVICE_LOST;
	}
	if (model->ctx != ctx) {  // T-2124 (D-SLM3446 P1-4): cross-context validation -- a model
	                          // handle mapped against a different context names a different
	                          // D3D12 device; same disposition as the malformed-argument case above.
	                          // P1-6 (external review, status-ABI collapse to DEVICE_LOST): this
	                          // rejects a PERMANENT caller-contract violation through a status §9
	                          // classifies transient/recoverable -- routing left as-is per
	                          // Claude/Poirot/435f730-t2124-adapter-uaf-review.md M2, flagged for
	                          // the P1-6 taxonomy pass (S-FREEZE).
		return SSLM_DEVICE_LOST;
	}
	// EntrySubmitGuard (§4.4, rev 8 placement): the map uploads on the context's own device only
	// (no ledger term); an existing entry point on an adapter's model, so its scope is a pooled
	// sequence Submitted. After the cross-context refusal, before the geometry: the population
	// refusals follow it (§4.4, *Status precedence*).
	if (const SslmGpuStatus g = EntrySubmitGuard(GuardScope::kPooled, GuardDevice::kPerContextOnly);
	    g != SSLM_OK) {
		return g;
	}

	superslm_adapter::BaseModelGeometry base_geom;
	base_geom.num_hidden_layers = model->num_hidden_layers;
	base_geom.hidden_size = model->hidden_size;
	base_geom.intermediate_size = model->intermediate_size;
	base_geom.kv_hidden_size = static_cast<uint64_t>(model->num_key_value_heads) * model->head_dim;
	// SSLM-GEOMETRY-SITE: GS-33
	// T-2509 (GS-28/GS-29): q_width, the same num_attention_heads * head_dim construction this
	// file already threads elsewhere (e.g. the /*q_width=*/ call arguments below) -- q_proj's own
	// out_channels and o_proj's own in_channels. This IS q_width's own correct computation, not a
	// stand-in for hidden_size.
	base_geom.q_width = static_cast<uint64_t>(model->num_attention_heads) * model->head_dim;
	base_geom.base_artifact_hash = model->content_hash;

	// T-2113 (B6, D-SLM3368): PopulateAdapterFromView is the extracted body of
	// LoadAdapterArtifact (include/superslm/adapter_marshal.h) that runs AFTER SslmModel::Load --
	// this call site already HAS a parsed view (`*adapter_artifact`, design Sec5.2's own
	// parameter), so it validates/populates against that view directly rather than re-reading
	// and re-parsing the artifact's own bytes from a file path a view does not retain.
	superslm_adapter::AdapterMeta meta;
	std::vector<superslm::LayerAdapter> layer_adapters;
	std::string load_err;
	const superslm_adapter::AdapterLoadStatus load_st = superslm_adapter::PopulateAdapterFromView(
	    *adapter_artifact, base_geom, meta, layer_adapters, &load_err);
	if (load_st == superslm_adapter::AdapterLoadStatus::BaseHashMismatch) {
		return SSLM_ADAPTER_BASE_HASH_MISMATCH;
	}
	if (load_st != superslm_adapter::AdapterLoadStatus::Ok) {
		// Every other rejection (malformed ADP1, missing sections, a dimension/shape
		// mismatch) is a structurally-broken-artifact class design Sec9's own table assigns
		// no dedicated status to -- same "no more specific status" disposition B2/B3 already
		// use for the analogous marshal-failure case (sslm_gpu_model_map's own comment above).
		return SSLM_DEVICE_LOST;
	}

	// q,o,gate,up,down,k,v -- LayerAdapter's own field order (forward_sites.h), matched to
	// AdapterProjSlot's own [7] index by position.
	auto proj_at = [](const superslm::LayerAdapter& la, int p) -> const superslm::LayerAdapterProjection& {
		switch (p) {
			case 0: return la.q;
			case 1: return la.o;
			case 2: return la.gate;
			case 3: return la.up;
			case 4: return la.down;
			case 5: return la.k;
			default: return la.v;
		}
	};
	// T-2114 (M4, Claude/Poirot/50f3d5d-t2113-1p0-gpu-core-build-review.md): the SAME
	// positional order `proj_at` above uses, named -- feeds `AdapterInChannelsFor`
	// (adapter_marshal.h:225, the one function that owns the projection-kind-to-in_channels
	// rule) below instead of this loop re-deriving it from a bare `p == 4` positional check.
	static const char* const kProjNamesByIndex[7] = {"q_proj",   "o_proj", "gate_proj", "up_proj",
	                                                  "down_proj", "k_proj", "v_proj"};

	std::vector<uint8_t> ab_bytes;
	std::vector<uint8_t> fold_bytes;
	std::vector<std::array<AdapterProjSlot, 7>> slots(model->num_hidden_layers);

	for (uint32_t l = 0; l < model->num_hidden_layers; ++l) {
		const superslm::LayerAdapter& la = layer_adapters[l];
		for (int p = 0; p < 7; ++p) {
			const superslm::LayerAdapterProjection& proj = proj_at(la, p);
			if (!proj.a_weight || !proj.b_weight || !proj.delta_fold_entry || !proj.u_fold_entry) {
				continue;  // this (layer, projection) is not covered by this adapter -- present
				           // stays false, the identical NULL-per-projection shape LayerWeights'
				           // own B1c documents on the CPU path (forward_sites.h).
			}
			AdapterProjSlot& slot = slots[l][static_cast<size_t>(p)];
			slot.present = true;
			slot.rank = la.rank;
			slot.out_channels = static_cast<uint32_t>(proj.delta_fold_entry->row_count);
			// SSLM-GEOMETRY-SITE: GS-28
			// a_weight is [rank, in_channels]: in_channels isn't independently recorded on
			// LayerAdapterProjection -- reconstructed here by calling the ONE function that owns
			// the projection-kind-to-in_channels rule (AdapterInChannelsFor,
			// include/superslm/adapter_marshal.h), not a second copy of its own logic. T-2509:
			// base_geom.q_width above threaded through so o_proj's own in_channels (and thus
			// a_bytes, the byte extent read out of proj.a_weight below) is q_width, not
			// hidden_size.
			const uint64_t in_channels = superslm_adapter::AdapterInChannelsFor(
			    kProjNamesByIndex[p], model->hidden_size, model->intermediate_size,
			    base_geom.q_width);
			slot.a_bytes = static_cast<uint64_t>(la.rank) * in_channels;
			slot.b_bytes = static_cast<uint64_t>(slot.out_channels) * la.rank;

			slot.a_offset = ab_bytes.size();
			ab_bytes.insert(ab_bytes.end(), proj.a_weight, proj.a_weight + slot.a_bytes);
			slot.b_offset = ab_bytes.size();
			ab_bytes.insert(ab_bytes.end(), proj.b_weight, proj.b_weight + slot.b_bytes);

			slot.fold_offset = fold_bytes.size();
			const uint64_t dfs_bytes = static_cast<uint64_t>(proj.delta_fold_entry->row_count) * 12u;
			const uint64_t ufs_bytes = static_cast<uint64_t>(proj.u_fold_entry->row_count) * 12u;
			fold_bytes.insert(fold_bytes.end(), proj.delta_fold_entry->data,
			                   proj.delta_fold_entry->data + dfs_bytes);
			fold_bytes.insert(fold_bytes.end(), proj.u_fold_entry->data,
			                   proj.u_fold_entry->data + ufs_bytes);
		}
	}
	// A degenerate adapter (every slot uncovered -- e.g. a target_modules_mask of zero) still
	// maps cleanly; MakeBuffer/Upload need at least one byte, matching sslm_gpu_model_map's own
	// cos_need/sin_need "no rope tables" fallback (>= 8 bytes) above.
	if (ab_bytes.empty()) ab_bytes.resize(1, 0);
	if (fold_bytes.empty()) fold_bytes.resize(1, 0);

	std::unique_ptr<SslmGpuAdapterHandle> h(new SslmGpuAdapterHandle());
	try {
		h->lora_ab_buf = UploadResidentBufferSync(ctx->device, ab_bytes.data(), ab_bytes.size());
		h->fold_buf = UploadResidentBufferSync(ctx->device, fold_bytes.data(), fold_bytes.size());
	} catch (const std::bad_alloc&) {
		throw;
	} catch (const std::exception&) {
		return ClassifyHandleFault(&ctx->device);  // 1.8.0 (TE-426): as sslm_gpu_model_map's uploads
	} catch (...) {
		return ClassifyHandleFault(&ctx->device);  // 1.8.0 (TE-435): every other exception type, as above
	}

	h->ctx = ctx;
	h->model = model;
	h->slots = std::move(slots);
	h->rank = meta.rank;

	ctx->live_handles += 1;
	// T-2243 (M2, design plan Sec6.1): this model now has one more live mapped adapter --
	// mirrored by the decrement in sslm_gpu_adapter_unmap, below.
	model->live_adapters += 1;
	*out_adapter = h.release();
	return SSLM_OK;
}

// Design Sec5.2 states "releases the adapter's own residency and returns Ok ... carries no Busy
// precondition of its own (unlike model/sequence release) because an adapter handle holds no
// in-flight GPU work." T-2124 (D-SLM3446 P0-3, external review): that premise is FALSE on the
// async submit/finish split (design Sec6.2/B5). `sslm_decode_step_gpu`'s own submission path
// (SubmitOneSequenceDecode, this file) passes this handle's `lora_ab_buf`/`fold_buf` into the
// recorded command list via root descriptors (`adapter_bridge.lora_ab_resident`/`fold_resident`)
// and returns BEFORE the fence signals -- unlike every OTHER per-call GPU resource
// (`GpuLayerLoopInFlight`, superslm_gpu.cpp), nothing kept these two buffers alive past that
// return. Deleting this handle here while such a submission is still in flight frees them out
// from under a command list the GPU may still be executing: a real use-after-free/device-removal
// hazard, the exact class `GpuLayerLoopInFlight`'s own header comment already documents being
// reproduced by execution once (T-2039) for the other resident buffers this same split shares.
//
// Fixed the same way `sslm_gpu_model_unmap` already fixes the identical class of gap for MODEL
// handles (design Sec9's own Busy-precedence row, D-SLM3417): `submitted_sequences` (this
// handle's own field, above) is checked before any release proceeds -- an adapter bound to at
// least one Submitted sequence returns Busy, exactly like a model does. Chosen over retaining
// ComPtr copies inside `GpuLayerLoopInFlight` (the alternative this design's own nearest
// precedent, T-2039's fix, would suggest) because Busy-accounting mirrors the model-unmap
// precedent exactly and keeps this call's contract legible at the call site (a caller sees WHY
// release did not happen), rather than adding a second, adapter-shaped retention mechanism next
// to the per-call-resource one `GpuLayerLoopInFlight` already carries.
//
// A sequence still bound to `adapter` in the IDLE state (no submission currently in flight)
// continues decoding against whatever its own buffers still contain until that sequence's next
// call rebinds or releases -- the documented residual design Sec5.2 already names explicitly,
// unaffected by this fix, not a guarded case this call adds one for.
SslmGpuStatus sslm_gpu_adapter_unmapImpl(SslmGpuContext* ctx, SslmGpuAdapterHandle* adapter) {
	if (!adapter) {
		return SSLM_OK;  // same null-is-a-no-op reasoning as sslm_gpu_model_unmap(nullptr).
	}
	// T-2124 (D-SLM3446 P1-4): validate this handle was actually mapped against `ctx` -- see
	// sslm_gpu_model_unmap's own identical check and comment.
	if (adapter->ctx != ctx) {
		return SSLM_DEVICE_LOST;  // same "no more specific status" disposition sslm_gpu_adapter_map
		                          // already uses for a malformed/foreign-context argument.
		                          // P1-6 (external review, status-ABI collapse to DEVICE_LOST): this
		                          // rejects a PERMANENT caller-contract violation through a status
		                          // §9 classifies transient/recoverable -- routing left as-is per
		                          // Claude/Poirot/435f730-t2124-adapter-uaf-review.md M2, flagged
		                          // for the P1-6 taxonomy pass (S-FREEZE).
	}
	if (adapter->submitted_sequences() > 0) {
		return SSLM_BUSY;  // T-2124 (D-SLM3446 P0-3): Busy-precedence, mirroring
		                    // sslm_gpu_model_unmap's own D-SLM3417 fix above.
	}
	// T-2243 (S2, design plan Sec6.1/Sec10 Phase 2 S2(e), D-SLM3965): a sequence that currently
	// HOLDS a bind to this adapter (`sslm_gpu_seq_bind_adapter`) has a `bound_adapter` pointer
	// that would be left dangling by this delete, even though no call against it is in flight
	// this instant -- persistent-liveness, not the transient `submitted_sequences` check above.
	// Remedy: unbind every sequence still holding this adapter
	// (`sslm_gpu_seq_bind_adapter(ctx, seq, nullptr)`) and retry.
	if (adapter->bound_sequences > 0) {
		return SSLM_ADAPTER_HAS_BOUND_SEQUENCES;
	}
	if (adapter->model && adapter->model->live_adapters > 0) {
		// T-2243 (M2, design plan Sec6.1): the symmetric decrement of adapter_map's own
		// increment, above.
		adapter->model->live_adapters -= 1;
	}
	DecrementLiveCount(adapter->ctx->live_handles);
	adapter->destroyed = true;
	delete adapter;
	return SSLM_OK;
}

// Design Sec4.1.1: "on device-acquisition failure, *out_ctx is set to nullptr and
// the call returns DeviceLost (Sec9) -- the same status a later mid-decode
// device-removal produces". Since 1.8.0 a device acquisition that ran out of memory returns
// SSLM_GPU_ALLOCATION_FAILED instead, and the caller may retry. harness::Device::Init() never
// throws: it records why a setup failed (SetupFailure), which this call reads, exactly as
// harness::GetDevice() does for the process's submission device.
//
// GpuContextConfig::shader_dir (include/superslm/gpu_1p0.h, sslm_gpu_context_create's comment):
// a non-null value is validated and checked against the process's fixed shader directory
// BEFORE the device is created, so a refusal creates no device; it fixes the process's
// directory only after device acquisition succeeds. A null value takes the unchanged path.
SslmGpuStatus sslm_gpu_context_createImpl(GpuContextConfig cfg, SslmGpuContext** out_ctx) {
	if (!out_ctx) {
		// Not a case the design's own Sec9 table assigns a status to (a null
		// out-parameter is a caller contract violation the design does not model
		// as a runtime-recoverable condition anywhere else in Sec4/Sec5 either);
		// there is no live object to report through, so there is nothing this
		// call can do except decline to dereference a null pointer. Documented
		// here rather than silently invoking undefined behavior.
		return SSLM_DEVICE_LOST;
	}
	*out_ctx = nullptr;  // before any fallible work, so every refusal below leaves it null

	std::wstring shader_dir;  // normalized override; empty when cfg.shader_dir is null
	if (cfg.shader_dir != nullptr) {
		const superslm_gpu::harness::ShaderDirCheck check =
		    superslm_gpu::harness::CheckShaderDirOverride(cfg.shader_dir, &shader_dir);
		if (check != superslm_gpu::harness::ShaderDirCheck::Ok) {
			*out_ctx = nullptr;
			return check == superslm_gpu::harness::ShaderDirCheck::Conflict
			           ? SSLM_GPU_SHADER_DIR_CONFLICT
			           : SSLM_GPU_SHADER_DIR_INVALID;
		}
	}

	SslmGpuContext* ctx = new SslmGpuContext();
	// Device::Init() does not throw (1.8.0): a failed setup is recorded as a SetupFailure kind. The
	// fault classifier's rules in order: a device that reports itself removed is SSLM_DEVICE_LOST
	// (rule 1); a setup that ran out of memory -- a host allocation, or E_OUTOFMEMORY from any D3D12
	// step, device creation included -- is SSLM_GPU_ALLOCATION_FAILED (rule 2); every other cause
	// SSLM_DEVICE_LOST.
	ctx->device.Init();
	if (!ctx->device.available) {
		const bool removed = superslm_gpu::harness::FailedSetupDeviceReportedRemoved(ctx->device);
		const bool out_of_memory =
		    ctx->device.setup_failure.load() == superslm_gpu::harness::SetupFailure::Allocation;
		delete ctx;
		*out_ctx = nullptr;
		return !removed && out_of_memory ? SSLM_GPU_ALLOCATION_FAILED : SSLM_DEVICE_LOST;
	}

	if (!shader_dir.empty()) {
		// Re-checked under the state's mutex: a concurrent create or default-path shader load
		// may have fixed a different directory since the pre-device check.
		const superslm_gpu::harness::ShaderDirCheck commit =
		    superslm_gpu::harness::CommitShaderDirOverride(shader_dir);
		if (commit != superslm_gpu::harness::ShaderDirCheck::Ok) {
			delete ctx;
			*out_ctx = nullptr;
			return commit == superslm_gpu::harness::ShaderDirCheck::Conflict
			           ? SSLM_GPU_SHADER_DIR_CONFLICT
			           : SSLM_GPU_SHADER_DIR_INVALID;
		}
	}

	*out_ctx = ctx;
	return SSLM_OK;
}

// Design Sec4.1.1: "asserts (debug) or fails loudly (release: returns
// ContextHasLiveHandles ... and does NOT destroy the context, so the caller may
// release the offending handles and retry) if any model, adapter, or sequence
// handle created from this context has not been released first; returns Ok and
// destroys the context otherwise." No handle type above SslmGpuContext itself is
// constructible yet (B2/B3/B6 land those), so `live_handles` is always 0 today --
// the guard is real code, exercised honestly (it reads the same field B2+ will
// increment), not a stand-in that merely returns Ok unconditionally.
SslmGpuStatus sslm_gpu_context_destroyImpl(SslmGpuContext* ctx) {
	if (!ctx) {
		// Destroying a null context is a caller no-op under the same "no status
		// exists for a null-pointer contract violation" reasoning as create()
		// above; nothing to free, nothing to reject.
		return SSLM_OK;
	}
	if (ctx->live_handles > 0) {
		return SSLM_CONTEXT_HAS_LIVE_HANDLES;
	}
	ctx->destroyed = true;
	delete ctx;
	return SSLM_OK;
}

// T-2851 (A, design Sec4.4): copies the host parallel-for hook into the context, or clears it.
// The same field validation as the CPU workspace's setter (sslm_workspace_set_parallel_for).
SslmGpuStatus sslm_gpu_context_set_host_parallel_forImpl(SslmGpuContext* ctx,
                                                        const sslm_parallel_for* pf) {
	if (!ctx) return SSLM_SEQUENCE_KV_BUFFER_MISMATCH;  // this surface's null-handle status
	if (!pf) {
		ctx->host_parallel_for = sslm_parallel_for{};
		return SSLM_OK;
	}
	// The CPU setter's own validation (src/forward/parallel_split.h): the same field domain and the
	// same implemented `reserved` bits. The GPU backend reads its hook only for the finish, so a bit
	// it accepts here changes nothing it does.
	if (!superslm::ParallelForHookValid(*pf)) return SSLM_GPU_PARALLEL_FOR_INVALID;
	ctx->host_parallel_for = *pf;
	return SSLM_OK;
}

// T-2113 (B3, design Sec5.3/Sec10 B3): "allocates a REAL, dedicated DEFAULT-heap K/V
// buffer sized to context_cap for this sequence ... plus the host-mirrored
// SequenceLayerState fields ... the handle owns." Sizing uses the identical overflow-
// guarded product RunLayerLoopGpu's own guard ladder computes (superslm_gpu.cpp,
// `kv_bytes_needed` = num_hidden_layers * context_cap * num_key_value_heads * head_dim
// * 2) -- the same formula, so a buffer this call sizes and the buffer RunLayerLoopGpu's
// own guard would separately compute for the identical inputs can never disagree.
SslmGpuStatus sslm_gpu_seq_createImpl(SslmGpuContext* ctx, SslmGpuModelHandle* model,
                                   int64_t context_cap, SslmGpuSequenceHandle** out_seq) {
	if (!out_seq) {
		return SSLM_SEQUENCE_KV_BUFFER_MISMATCH;  // no live object to report through,
		                                           // same reasoning as the null-out-
		                                           // parameter cases in B1/B2 above.
	}
	*out_seq = nullptr;
	if (!ctx || !model || model->ctx != ctx) {  // T-2114 (S2): ->destroyed read removed, see model
	                                             // handle's own comment. T-2124 (D-SLM3446 P1-4):
	                                             // `model->ctx != ctx` -- a model mapped on a
	                                             // different context names a different D3D12
	                                             // device; a sequence created here would bind
	                                             // resources across two contexts.
		return SSLM_SEQUENCE_KV_BUFFER_MISMATCH;
	}
	if (context_cap < 1) {
		// Mirrors RunLayerLoopGpu's own InvalidContextCap guard (superslm_gpu.cpp:1038,
		// "context_cap < 1") -- design Sec9 assigns this call's own sizing failures to
		// SequenceKvBufferMismatch, so a non-positive context_cap (which the sizing
		// formula below cannot produce a sane buffer from either) takes that status
		// here rather than the CPU/GPU forward-status enum this call never returns.
		return SSLM_SEQUENCE_KV_BUFFER_MISMATCH;
	}

	// Design Sec5.3: "a real, dedicated DEFAULT-heap K/V buffer sized to context_cap" --
	// the identical overflow-guarded kv_bytes_needed product RunLayerLoopGpu's own guard
	// ladder computes (superslm_gpu.cpp, the WorkspaceSizeOrOverflow guard), so this
	// handle's own buffer and what a decode call through it will expect can never
	// silently disagree on size.
	size_t kv_bytes_needed = static_cast<size_t>(model->num_hidden_layers);
	const size_t kv_factors[] = {static_cast<size_t>(context_cap), model->num_key_value_heads,
	                              static_cast<size_t>(model->head_dim), 2u};
	for (size_t factor : kv_factors) {
		if (factor != 0 && kv_bytes_needed > SIZE_MAX / factor) {
			return SSLM_SEQUENCE_KV_BUFFER_MISMATCH;
		}
		kv_bytes_needed *= factor;
	}
	// EntrySubmitGuard (§4.4): the create uploads its zero K/V on the context's own device only (no
	// ledger term); an existing entry point on a model, so its scope is a pooled sequence
	// Submitted. After the argument, cap and overflow refusals, before the handle's allocation.
	if (const SslmGpuStatus g = EntrySubmitGuard(GuardScope::kPooled, GuardDevice::kPerContextOnly);
	    g != SSLM_OK) {
		return g;
	}

	std::unique_ptr<SslmGpuSequenceHandle> h(new SslmGpuSequenceHandle());
	// Zero-initialized: a fresh sequence (context_length == 0) never reads a K/V row
	// before this same call's own commit writes it (RunLayerLoopGpu's own
	// `fresh_sequence` reasoning, superslm_gpu.cpp, applies identically here), so the
	// buffer's initial content is never observed by a real forward pass -- zeroed
	// anyway (rather than left as MakeBuffer's own undefined DEFAULT-heap content) so a
	// hostile or buggy caller reading past `context_length` sees a defined value, never
	// uninitialized VRAM.
	std::vector<uint8_t> zero_kv(kv_bytes_needed, 0);
	try {
		h->kv_buf = UploadResidentUavBufferSync(ctx->device, zero_kv.data(), zero_kv.size());
		superslm_gpu::harness::NameBuffer(h->kv_buf.Get(), L"SuperSLM dedicated K/V");
	} catch (const std::bad_alloc&) {
		throw;
	} catch (const std::exception&) {
		// Mirrors sslm_gpu_model_map's own upload-failure disposition (design Sec5.1/Sec5.3
		// symmetry), the fault classifier since 1.8.0 (TE-426).
		return ClassifyHandleFault(&ctx->device);
	} catch (...) {
		return ClassifyHandleFault(&ctx->device);  // 1.8.0 (TE-435): every other exception type, as above
	}
	// Structural self-check (design Sec9: "context_cap inconsistent with the K/V buffer
	// this call itself just sized -- an internal invariant that should be unreachable,
	// guarded anyway per the substrate's own GpuGemmSplitSite default-case precedent").
	// By construction (the buffer was allocated at exactly `kv_bytes_needed` bytes,
	// immediately above), this cannot fail from any caller input -- real, exercised
	// guard code proving its own invariant true on every call, not a stand-in, matching
	// B1's `ContextHasLiveHandles`/B2's `ModelHasLiveSequences` precedent of a guard
	// that reads a real field even where no reachable input drives it false.
	if (h->kv_buf == nullptr || zero_kv.size() != kv_bytes_needed) {
		return SSLM_SEQUENCE_KV_BUFFER_MISMATCH;
	}

	h->ctx = ctx;
	h->model = model;
	h->context_cap = context_cap;
	h->kv_bytes = kv_bytes_needed;
	h->kv_needs_resume_barrier = false;  // freshly created in UNORDERED_ACCESS state
	                                      // (UploadResidentUavBufferSync above) -- the
	                                      // first call through this handle needs no
	                                      // resume barrier (gpu_port.h's own B3 comment).
	h->hidden_codes.assign(model->hidden_size, 0);
	h->prefill_hidden_codes.assign(model->hidden_size, 0);
	h->prefill_hidden_valid = false;  // prefill snapshot writer: create (and restore through it)
	h->hidden_scale = superslm::CarriedScale{};
	h->layer_index = 0;
	h->kv_saturation_count = 0;
	h->context_length = 0;
	h->bind_eligible = true;
	SetGpuSeqState(h.get(), superslm_gpu::SslmSequenceGpuState::Idle, nullptr);  // a fresh handle: Idle, no count
	// T-2113 (B5): `live_state.hidden_codes` aliases `hidden_codes.data()` for this handle's
	// whole lifetime -- `hidden_codes` never reallocates after this construction (fixed size,
	// never resized), so the pointer stays valid until sslm_gpu_seq_release destroys the
	// handle. `host_kv_mirror` starts zeroed, identically to `zero_kv` above (a fresh
	// sequence's own workspace mirror has never had a row written into it either).
	h->live_state.hidden_codes = h->hidden_codes.data();
	h->host_kv_mirror.assign(kv_bytes_needed, 0);

	model->live_sequences += 1;
	ctx->live_handles += 1;
	*out_seq = h.release();
	return SSLM_OK;
}

// Design Sec5.3/Sec4.2's own state table: "frees the buffer and host state ...
// CallProceedsOrBusy_SeqRelease's existing policy, unchanged: returns Busy against
// Submitted; otherwise ... returns Ok." No `sslm_decode_step_gpu` exists yet (B5 lands
// it), so no `SslmGpuSequenceHandle` this section can construct is ever reachable in
// the `Submitted` state -- `CallProceedsOrBusy_SeqRelease` is reused verbatim anyway
// (not reimplemented as "always proceed"), so B5's own async lifecycle needs no change
// here when it starts setting `state = Submitted` for real.
SslmGpuStatus sslm_gpu_seq_releaseImpl(SslmGpuContext* ctx, SslmGpuSequenceHandle* seq) {
	if (!seq) {
		return SSLM_OK;  // same null-is-a-no-op reasoning as sslm_gpu_context_destroy/
		                  // sslm_gpu_model_unmap(nullptr) above.
	}
	// T-2124 (D-SLM3446 P1-4): validate this handle was actually created against `ctx` -- see
	// sslm_gpu_model_unmap's own identical check and comment. `(void)ctx` (previously here) is
	// removed since the parameter is now read both here and by the decrement further down.
	if (seq->ctx != ctx) {
		return SSLM_SEQUENCE_KV_BUFFER_MISMATCH;
	}
	const bool is_submitted = seq->state() == superslm_gpu::SslmSequenceGpuState::Submitted;
	if (!superslm_gpu::CallProceedsOrBusy_SeqRelease(is_submitted)) {
		return SSLM_BUSY;
	}
	// Paged-KV plan §4 item 6b, step G3: a pooled sequence's release touches only its own table and
	// reserve (its pages back to the pool, a freed dirty page poisoned in the pool's host mirror by
	// kv_pages) and fills nothing, so no neighbour's rows change. HolderRelease consumes the holder
	// whatever it returns (review-g23 F2), so the release is finished either way and a non-OK status
	// (a refcount underflow, cell 11.3's class) is reported at the end, as the CPU twin does
	// (ReleaseHolder, sslm_abi.cpp).
	SslmGpuStatus holder_status = SSLM_OK;
	if (seq->pool != nullptr) {
		if (seq->holder != nullptr) {
			const superslm::kv_pages::PageStatus ps = superslm::kv_pages::HolderRelease(seq->holder);
			seq->holder = nullptr;  // freed by HolderRelease on every return
			if (ps != superslm::kv_pages::PageStatus::kOk) holder_status = SSLM_SEQUENCE_KV_BUFFER_MISMATCH;
		}
		int64_t refs = seq->pool->live_refs.load();
		while (refs > 0 && !seq->pool->live_refs.compare_exchange_weak(refs, refs - 1)) {
		}
		seq->pool = nullptr;
	}
	if (seq->model) DecrementLiveCount(seq->model->live_sequences);
	// T-2243 (S2, design plan Sec6.1): unbind before delete -- without this, every released
	// sequence that was ever bound leaks one count against its adapter's own `bound_sequences`,
	// making that adapter permanently un-unmappable even after every sequence that bound it is
	// gone.
	if (seq->bound_adapter != nullptr) {
		seq->bound_adapter->bound_sequences -= 1;
		seq->bound_adapter = nullptr;
	}
	DecrementLiveCount(seq->ctx->live_handles);
	seq->destroyed = true;
	delete seq;
	return holder_status;
}

// -------------------------------------------------------------------------------------------------
// Paged-KV plan (rev 16.1) §4 items 3 and 7, step G3: the GPU page pool and budgeted sequences.
// -------------------------------------------------------------------------------------------------

// §4 item 3. Refusals in order: a null out-handle; a context, model or model-of-another-context
// mismatch; a zero page count or a model with no page geometry (SSLM_SEQUENCE_KV_BUFFER_MISMATCH,
// the handle-mismatch status, with a null out-handle); page_count * page_bytes above 2^32
// (SSLM_GPU_KV_POOL_TOO_LARGE, before anything is allocated); SSLM_BUSY while any sequence in the
// process is Submitted (the create submits its zero-fill). Then phase A as one block -- the handle,
// the host mirror, the kv_pages allocator and AllocateZeroFillBuffers -- where any exception asks
// rule 1 first (a removed device: SSLM_DEVICE_LOST, whatever the HRESULT) and is otherwise
// SSLM_GPU_ALLOCATION_FAILED (§4 item 3's stated choice); and phase B,
// ZeroFillResidentUavBufferSync, under 1.8.0's classifier (ClassifyHandleFault): a list not
// confirmed Closed or a failed Signal or wait is rule 0, SSLM_DEVICE_LOST, terminal.
SslmGpuStatus sslm_gpu_kv_pool_createImpl(SslmGpuContext* ctx, SslmGpuModelHandle* model, uint32_t page_count,
                                          SslmGpuKvPoolHandle** out_pool) {
	if (!out_pool) return SSLM_SEQUENCE_KV_BUFFER_MISMATCH;
	*out_pool = nullptr;
	if (!ctx || !model || model->ctx != ctx) return SSLM_SEQUENCE_KV_BUFFER_MISMATCH;
	if (page_count == 0) return SSLM_SEQUENCE_KV_BUFFER_MISMATCH;
	const superslm_gpu::kv_pool::GpuPageGeometry geo = superslm_gpu::kv_pool::ComputeGpuPageGeometry(
	    model->num_hidden_layers, model->num_key_value_heads, model->head_dim, model->context_cap,
	    model->kv_block_size);
	if (!geo.ok) return SSLM_SEQUENCE_KV_BUFFER_MISMATCH;
	if (superslm_gpu::kv_pool::PoolBytesTooLarge(page_count, geo.page_bytes,
	                                             superslm_gpu::kv_pool::MaxPoolBytesForAdapter(ctx->device.dedicated_video_memory))) {
		return SSLM_GPU_KV_POOL_TOO_LARGE;
	}
	// EntrySubmitGuard (§4.4): the create submits its zero-fill on the context's own device, so it
	// is a guarded arc row (scope: any sequence Submitted; per-context, so no ledger term), placed
	// before the pool's allocation.
	if (const SslmGpuStatus g = EntrySubmitGuard(GuardScope::kAny, GuardDevice::kPerContextOnly); g != SSLM_OK) {
		return g;
	}
	const uint64_t bytes = static_cast<uint64_t>(page_count) * static_cast<uint64_t>(geo.page_bytes);

	std::unique_ptr<SslmGpuKvPoolHandle> h;
	ZeroFillBuffers fill;
	// Phase A, one block (§4 item 3, rev 15.1): every allocation and Map before the first reset.
	bool phase_a_failed = false;
	bool host_failure = false;
	try {
		h.reset(new SslmGpuKvPoolHandle());
		h->host_mirror.assign(static_cast<size_t>(bytes), 0);
		const superslm::kv_pages::PageStatus ps = superslm::kv_pages::PoolCreate(
		    h->host_mirror.data(), page_count, geo.page_bytes, geo.page_positions, &h->pages);
		if (ps != superslm::kv_pages::PageStatus::kOk) throw std::bad_alloc();  // kAllocationFailed
		fill = AllocateZeroFillBuffers(ctx->device, bytes);
	} catch (const std::bad_alloc&) {
		phase_a_failed = true;
		host_failure = true;
	} catch (const std::length_error&) {
		phase_a_failed = true;
		host_failure = true;
	} catch (...) {
		phase_a_failed = true;  // a D3D12 failure, already logged by SSLM_GPU_HR with its HRESULT
	}
	if (phase_a_failed) {
		if (host_failure) {
			std::fprintf(stderr, "superslm_gpu: sslm_gpu_kv_pool_create: host allocation for %u pages (%llu bytes) failed\n",
			             page_count, static_cast<unsigned long long>(bytes));
		}
		return MapDeviceReportedRemoved(ctx->device) ? SSLM_DEVICE_LOST : SSLM_GPU_ALLOCATION_FAILED;
	}
	// Phase B: the one reset and submission.
	try {
		ZeroFillResidentUavBufferSync(ctx->device, fill);
	} catch (...) {
		return ClassifyHandleFault(&ctx->device);
	}
	h->ctx = ctx;
	h->model = model;
	h->geo = geo;
	h->page_count = page_count;
	h->pool_bytes = static_cast<size_t>(bytes);
	h->pool_buf = fill.resident;
	superslm_gpu::harness::NameBuffer(h->pool_buf.Get(), L"SuperSLM K/V page pool");
	h->kv_needs_resume_barrier = false;  // left in UNORDERED_ACCESS by the zero-fill's barrier
	ctx->live_handles += 1;
	model->live_pools += 1;
	*out_pool = h.release();
	return SSLM_OK;
}

// The largest page count sslm_gpu_kv_pool_create admits for `model` on this context's adapter: the
// pool cap (MaxPoolBytesForAdapter over the adapter's dedicated video memory) over the model's page
// size. Validates as the create does; allocates and submits nothing.
SslmGpuStatus sslm_gpu_kv_pool_max_page_countImpl(SslmGpuContext* ctx, SslmGpuModelHandle* model, uint32_t* out_pages) {
	if (!out_pages) return SSLM_SEQUENCE_KV_BUFFER_MISMATCH;
	if (!ctx || !model || model->ctx != ctx) return SSLM_SEQUENCE_KV_BUFFER_MISMATCH;
	const superslm_gpu::kv_pool::GpuPageGeometry geo = superslm_gpu::kv_pool::ComputeGpuPageGeometry(
	    model->num_hidden_layers, model->num_key_value_heads, model->head_dim, model->context_cap,
	    model->kv_block_size);
	if (!geo.ok) return SSLM_SEQUENCE_KV_BUFFER_MISMATCH;
	*out_pages = superslm_gpu::kv_pool::LargestAdmissiblePageCount(
	    geo.page_bytes, superslm_gpu::kv_pool::MaxPoolBytesForAdapter(ctx->device.dedicated_video_memory));
	return SSLM_OK;
}

// The plan leaves pool destroy with live handles unstated (§4 item 3; gpu_1p0.h's G3 note). The
// CPU twin refuses (SSLM_POOL_HAS_LIVE_HANDLES); the GPU enum has no such status, so the closest,
// SSLM_MODEL_HAS_LIVE_SEQUENCES, is returned while any sequence drawn from the pool lives, and the
// pool is not destroyed. A null pool is a no-op (as every GPU destroy); a pool of another context
// is SSLM_SEQUENCE_KV_BUFFER_MISMATCH.
SslmGpuStatus sslm_gpu_kv_pool_destroyImpl(SslmGpuContext* ctx, SslmGpuKvPoolHandle* pool) {
	if (!pool) return SSLM_OK;
	if (pool->ctx != ctx) return SSLM_SEQUENCE_KV_BUFFER_MISMATCH;
	if (pool->live_refs > 0) return SSLM_MODEL_HAS_LIVE_SEQUENCES;
	if (pool->model) DecrementLiveCount(pool->model->live_pools);
	DecrementLiveCount(pool->ctx->live_handles);
	delete pool;
	return SSLM_OK;
}

// §4 item 7: the budgeted create, §3.4's mirror. Every refusal comes before anything is drawn:
// a null out-handle; a context, model or pool mismatch (a model or pool of another context, a pool
// of another model) and a budget outside [1, cap], each SSLM_SEQUENCE_KV_BUFFER_MISMATCH; then the
// draw of R(budget) pages, SSLM_GPU_KV_POOL_EXHAUSTED with nothing drawn. The table buffer is then
// allocated on the context's device (no submission: it is filled at the head of the sequence's
// first submission, §4 item 5); a failure there returns the draw and is classified as every
// handle create's (ClassifyHandleFault). Submits nothing, so it is not refused Busy.
SslmGpuStatus sslm_gpu_seq_create_budgetedImpl(SslmGpuContext* ctx, SslmGpuModelHandle* model, SslmGpuKvPoolHandle* pool,
                                               int32_t budget, SslmGpuSequenceHandle** out_seq) {
	if (!out_seq) return SSLM_SEQUENCE_KV_BUFFER_MISMATCH;
	*out_seq = nullptr;
	if (!ctx || !model || !pool || model->ctx != ctx || pool->ctx != ctx || pool->model != model) {
		return SSLM_SEQUENCE_KV_BUFFER_MISMATCH;
	}
	if (!superslm_gpu::kv_pool::BudgetInDomain(budget, model->context_cap)) return SSLM_SEQUENCE_KV_BUFFER_MISMATCH;
	const uint32_t reserve = superslm_gpu::kv_pool::ReservationPages(budget, pool->geo, model->context_cap);

	std::unique_ptr<SslmGpuSequenceHandle> h(new SslmGpuSequenceHandle());
	h->hidden_codes.assign(model->hidden_size, 0);
	h->prefill_hidden_codes.assign(model->hidden_size, 0);
	superslm::kv_pages::PageHolder* holder = nullptr;
	const superslm::kv_pages::PageStatus ps =
	    superslm::kv_pages::HolderCreate(pool->pages, pool->geo.cap_pages, reserve, &holder);
	if (ps == superslm::kv_pages::PageStatus::kExhausted) return SSLM_GPU_KV_POOL_EXHAUSTED;
	if (ps == superslm::kv_pages::PageStatus::kAllocationFailed) return SSLM_GPU_ALLOCATION_FAILED;
	if (ps != superslm::kv_pages::PageStatus::kOk) return SSLM_SEQUENCE_KV_BUFFER_MISMATCH;
	try {
		h->table_buf = ctx->device.MakeBuffer(static_cast<UINT64>(pool->geo.cap_pages) * sizeof(uint32_t),
		                                      D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_FLAG_NONE,
		                                      D3D12_RESOURCE_STATE_COMMON);
		superslm_gpu::harness::NameBuffer(h->table_buf.Get(), L"SuperSLM K/V page table");
	} catch (...) {
		superslm::kv_pages::HolderRelease(holder);
		return ClassifyHandleFault(&ctx->device);
	}
	h->ctx = ctx;
	h->model = model;
	h->pool = pool;
	h->holder = holder;
	h->table_dirty = true;
	h->budget = budget;
	h->origin = 0;
	h->limit = superslm_gpu::kv_pool::HolderLimit(0, budget, model->context_cap);
	h->context_cap = model->context_cap;
	h->kv_bytes = 0;  // no dedicated buffer: K/V live in the pool
	h->kv_needs_resume_barrier = false;  // unused: a pooled sequence passes the pool's latch
	h->prefill_hidden_valid = false;
	h->hidden_scale = superslm::CarriedScale{};
	h->layer_index = 0;
	h->kv_saturation_count = 0;
	h->context_length = 0;
	h->bind_eligible = true;
	SetGpuSeqState(h.get(), superslm_gpu::SslmSequenceGpuState::Idle, nullptr);  // a fresh handle: Idle, no count
	h->live_state.hidden_codes = h->hidden_codes.data();
	pool->live_refs.fetch_add(1);
	model->live_sequences += 1;
	ctx->live_handles += 1;
	*out_seq = h.release();
	return SSLM_OK;
}

// §4 item 9: a diagnostic no verdict reads. Read under the pool mutex (kv_pages::PoolCounts).
SslmGpuStatus sslm_gpu_kv_pool_statsImpl(const SslmGpuKvPoolHandle* pool, SslmGpuKvPoolStats* out) {
	if (!pool || !out || !pool->pages) return SSLM_SEQUENCE_KV_BUFFER_MISMATCH;
	if (out->struct_size != sizeof(SslmGpuKvPoolStats)) return SSLM_SEQUENCE_KV_BUFFER_MISMATCH;
	uint32_t free_pages = 0, shared = 0;
	superslm::kv_pages::PoolCounts(pool->pages, &free_pages, &shared);
	out->page_count = pool->page_count;
	out->free_pages = free_pages;
	out->shared_pages = shared;
	out->page_positions = pool->geo.page_positions;
	return SSLM_OK;
}

// §4 item 9: the GPU twin of the CPU page peek, bounds-checked against hostile arguments (the
// offset and length checked without forming offset + len).
SslmGpuStatus sslm_gpu_test_only_peek_pool_mirror_bytesImpl(const SslmGpuKvPoolHandle* pool, uint32_t page,
                                                             size_t offset, size_t len, uint8_t* out) {
	if (!pool || (!out && len != 0)) return SSLM_SEQUENCE_KV_BUFFER_MISMATCH;
	if (page >= pool->page_count) return SSLM_SEQUENCE_KV_BUFFER_MISMATCH;
	const size_t page_bytes = pool->geo.page_bytes;
	if (offset > page_bytes || len > page_bytes - offset) return SSLM_SEQUENCE_KV_BUFFER_MISMATCH;
	if (len != 0) {
		std::memcpy(out, pool->host_mirror.data() + static_cast<size_t>(page) * page_bytes + offset, len);
	}
	return SSLM_OK;
}

// T-2243 (S2, design plan Sec6.1/Sec10 Phase 2 S2, D-SLM3954/D-SLM3996): see gpu_1p0.h's own
// header comment on this declaration for the full rule ordering. Rule 2's mid-token predicate
// deliberately admits BOTH `layer_index == 0` (a fresh/just-finished-token rest) AND
// `layer_index == model->num_hidden_layers` (a drained, at-rest sequence that has not yet
// called SslmGpuSeqFinishTokenForG5Bridge -- the shipped public header states this is a valid
// resting state, gpu_1p0.h's own SslmGpuSeqFinishTokenForG5Bridge precondition comment, and
// SslmGpuSeqFinishTokenForG5Bridge's own body already tests the SAME two-sided condition,
// `:2029` above) -- only the OPEN interval `0 < layer_index < num_hidden_layers` is genuinely
// mid-token.
// Paged-KV plan §4.4, step G3: the wait seam. Waits on `seq`'s in-flight fence through the ledger
// (Wait(v) observes it) without finishing the token: `seq` stays Submitted with its counts raised,
// and nothing is left executing or unobserved. Submits nothing; a blocking waiter, so inside P1's
// precondition. SSLM_OK with nothing to wait on (not Submitted, or Submitted with no token, the
// bench seam's shape); SSLM_SEQUENCE_KV_BUFFER_MISMATCH for a null or another context's handle; a
// wait that cannot be confirmed strands the ledger and throws GpuSubmissionStrandedError, which the
// boundary reports as SSLM_DEVICE_LOST.
SslmGpuStatus sslm_gpu_test_only_wait_in_flightImpl(SslmGpuContext* ctx, SslmGpuSequenceHandle* seq) {
	if (!ctx || !seq || seq->ctx != ctx) return SSLM_SEQUENCE_KV_BUFFER_MISMATCH;
	if (seq->state() != superslm_gpu::SslmSequenceGpuState::Submitted || !seq->in_flight) return SSLM_OK;
	seq->in_flight.ObserveWithoutFinishing();
	return SSLM_OK;
}

SslmGpuStatus sslm_gpu_seq_bind_adapterImpl(SslmGpuContext* ctx, SslmGpuSequenceHandle* seq,
                                         const SslmGpuAdapterHandle* adapter_or_null) {
	if (!ctx || !seq || seq->ctx != ctx) {  // no channel exists for a malformed handle, the same
	                                          // disposition every other 1.0 entry point uses.
		return SSLM_SEQUENCE_KV_BUFFER_MISMATCH;
	}
	// T-2243 review finding 1 (D-SLM4113): this is a host-mutating sequence call and carries
	// the same `is_submitted` precondition every sibling (release/embed_token/decode_step/
	// save/reset) already enforces -- reusing CallProceedsOrBusy_SeqReset, the same predicate
	// sslm_gpu_seq_embed_token reuses for its own distinct call. Without this, seq->layer_index
	// (only advanced by sslm_gpu_ready and the chunk guard's close) still reads its pre-submit
	// value throughout a Submitted window, so the mid-token guard below cannot see it: a bind
	// admitted between sslm_decode_step_gpu and sslm_gpu_ready splits one token's layer walk
	// across two adapters, silently.
	const bool is_submitted = seq->state() == superslm_gpu::SslmSequenceGpuState::Submitted;
	if (!superslm_gpu::CallProceedsOrBusy_SeqReset(is_submitted)) {
		return SSLM_BUSY;
	}
	SslmGpuModelHandle* model = seq->model;
	if (model != nullptr && seq->layer_index != 0 && seq->layer_index != model->num_hidden_layers) {
		return SSLM_BUSY;  // genuinely mid-token -- resolves through the caller's own continued
		                    // decoding of the token already in progress, not automatically.
	}
	if (adapter_or_null != nullptr) {
		if (adapter_or_null->model != seq->model) {  // bind-time application of the identical
		                                              // per-call check sslm_decode_step_gpu
		                                              // already runs.
			return SSLM_ADAPTER_MODEL_MISMATCH;
		}
		if (adapter_or_null->ctx != ctx) {  // a bound adapter mapped against a different context
		                                      // names a different device.
			return SSLM_SEQUENCE_KV_BUFFER_MISMATCH;
		}
	}
	if (seq->bound_adapter == adapter_or_null) {
		return SSLM_OK;  // idempotent no-op, mirroring sslm_seq_set_adapter's own short-circuit.
	}
	if (seq->bound_adapter != nullptr) {
		seq->bound_adapter->bound_sequences -= 1;
	}
	if (adapter_or_null != nullptr) {
		adapter_or_null->bound_sequences += 1;
	}
	seq->bound_adapter = adapter_or_null;
	return SSLM_OK;
}

// T-2113 (B3.5, design Sec5.3a/Sec10 B3.5, mini-fold 2026-08-15 routing D-SLM3367): the
// production token-feed entry point. Host-only -- no dispatch, no GPU state touched, no
// interaction with the async lifecycle or the dispatch geometry (design's own "issues no
// dispatch" framing is why this call needs neither B4 nor B5). Reuses the CPU path's own
// `EmbedEntry` primitive verbatim (forward_sites.h) against this handle's own model's
// host-retained embed_weights/embed_site_constant/vocab_size (B2's own amendment, above)
// -- the identical arithmetic RunWholeToken's own closure calls (forward_sites.cpp),
// called with the same trailing defaults (site="", token_index=0, no trace hook) so a
// caller driving this call pair reproduces that closure's own output byte-for-byte.
SslmGpuStatus sslm_gpu_seq_embed_tokenImpl(SslmGpuContext* ctx, SslmGpuSequenceHandle* seq,
                                        int32_t token_id) {
	if (!ctx || !seq || !seq->model || seq->ctx != ctx) {  // T-2114 (S2): ->destroyed reads
	                                     // removed, see the sequence/model handles' own field
	                                     // comments. T-2124 (D-SLM3446 P1-4): `seq->ctx != ctx`.
		return SSLM_SEQUENCE_KV_BUFFER_MISMATCH;  // no channel exists for a malformed handle
		                                           // at this call, same "no more specific
		                                           // status" disposition every other
		                                           // malformed-handle site in this file uses.
	}
	// Design Sec5.3a: "seq state Idle (never Submitted -- returns Busy, the same convention
	// every other host-mutating call on a sequence handle already follows)."
	const bool is_submitted = seq->state() == superslm_gpu::SslmSequenceGpuState::Submitted;
	if (!superslm_gpu::CallProceedsOrBusy_SeqReset(is_submitted)) {
		return SSLM_BUSY;  // CallProceedsOrBusy_SeqReset is the correct policy predicate to
		                    // reuse here: it is the SAME "proceed iff Idle/Completed, Busy
		                    // iff Submitted" rule design Sec5.9 assigns every host-mutating
		                    // sequence call (seq_save/seq_reset/seq_release all reuse the
		                    // identical predicate against their own distinct calls today).
	}

	SslmGpuModelHandle* model = seq->model;
	// F-S3-8 (forward_sites.cpp): validate token_id against [0, vocab_size) BEFORE any row
	// of embed_weights is read -- the identical bounds check EmbedEntry itself performs,
	// checked HERE too (not merely relying on EmbedEntry's own internal check) so this
	// call's own documented contract ("leaves seq's own state untouched" on rejection) is
	// satisfied by construction: nothing below this check ever writes to `seq`.
	if (token_id < 0 || token_id >= model->vocab_size) {
		return SSLM_TOKEN_ID_OUT_OF_RANGE;
	}
	seq->bind_eligible = false;

	std::vector<int8_t> embed_codes(model->hidden_size);
	superslm::CarriedScale embed_scale{};
	const superslm::SslmForwardStatus est = superslm::EmbedEntry(
	    token_id, model->vocab_size, model->embed_weights.data(),
	    static_cast<size_t>(model->hidden_size), model->embed_site_constant, embed_codes.data(),
	    &embed_scale);
	if (est != superslm::SslmForwardStatus::Ok) {
		// EmbedEntry's own only rejection is the identical token_id bounds check already
		// performed above (forward_sites.cpp:757) -- unreachable in practice given the
		// guard above, handled anyway per this file's own "an unhandled path throws loudly
		// rather than silently miscomputing" precedent (SslmGpuSequenceHandle's own
		// SequenceKvBufferMismatch structural self-check, sslm_gpu_seq_create, above).
		return SSLM_TOKEN_ID_OUT_OF_RANGE;
	}

	// Design Sec5.3a: "overwrites seq's own host-mirrored hidden_codes/hidden_scale ...
	// and resets layer_index to 0" -- matching RunWholeToken's own reset before a fresh
	// layer pass (forward_sites.cpp). `seq->live_state.hidden_codes` already aliases
	// `seq->hidden_codes.data()` (set once at sslm_gpu_seq_create) -- writing through
	// `seq->hidden_codes` keeps both in sync without a second copy.
	std::copy(embed_codes.begin(), embed_codes.end(), seq->hidden_codes.begin());
	seq->hidden_scale = embed_scale;
	seq->layer_index = 0;
	seq->live_state.hidden_scale = seq->hidden_scale;
	seq->live_state.layer_index = 0;
	return SSLM_OK;
}

namespace {
// T-2113 (B5): `RunLayerLoopGpuSubmit`'s own guard ladder (superslm_gpu.cpp, unchanged
// from the pre-1.0 substrate) rejects for reasons the 1.0 API's own construction already
// makes unreachable in ordinary use for MOST guards -- context_cap/hidden_size/head_dim
// are validated once, at sslm_gpu_seq_create/sslm_gpu_model_map time, never re-supplied
// per decode call. Numeric-domain guards (ChainInputOutOfDomain,
// SiluCompositionScaleOutOfDomain, SoftmaxRowWidthOutOfDomain, and the rest of
// superslm::SslmForwardStatus's own CPU-domain family) ARE reachable through this API --
// a poisoned adapter delta or a pathological activation value reaches them exactly the
// way the CPU path always could.
//
}  // namespace

// T-2114 (S1, Claude/Poirot/50f3d5d-t2113-1p0-gpu-core-build-review.md): `st` is
// whatever `RunLayerLoopGpuSubmit` RETURNED when `inflight == nullptr` -- confirmed by
// execution (not assumed) to include TWO distinct classes, not one: the ordinary
// CPU-domain guard ladder (ChainInputOutOfDomain and the rest of the family, checked
// BEFORE any device work begins), AND `GpuDeviceRemoved`/`GpuAllocationFailed`, which
// `RunLayerLoopGpuSubmit`'s own internal `catch (const std::runtime_error&)`
// (superslm_gpu.cpp) produces when a REAL device/HR failure happens mid-recording and
// converts it to a normal return rather than letting it escape as an exception. An
// earlier version of this function (and this comment) assumed only the first class
// could reach here -- refuted by running this suite's own dim9 cells, which surfaced a
// real `GpuAllocationFailed` through this exact path. Design Sec9 still assigns no
// per-guard 1.0 status to the CPU-domain family (SSLM_SEQUENCE_REJECTED covers all of
// it, undifferentiated, on purpose), but the two device-derived statuses must map to
// SSLM_DEVICE_LOST. T-2578 adds the one deployment-derived exception:
// GpuShaderBinaryStale maps to SSLM_GPU_SHADER_BINARY_STALE, never to either family.
//
// 1.8.0 (TE-426): the device-derived statuses split three ways. GpuAllocationFailed -- an
// allocation failed on a device not removed, with the submission clean -- maps to
// SSLM_GPU_ALLOCATION_FAILED; GpuDeviceRemoved and GpuOperationFailed (a non-allocation GPU
// failure) map to SSLM_DEVICE_LOST.
SslmGpuStatus MapSubmitRejectionToGpuStatus(superslm::SslmForwardStatus st) {
	if (st == superslm::SslmForwardStatus::GpuShaderBinaryStale) {
		return SSLM_GPU_SHADER_BINARY_STALE;
	}
	if (st == superslm::SslmForwardStatus::GpuAllocationFailed) {
		return SSLM_GPU_ALLOCATION_FAILED;
	}
	if (st == superslm::SslmForwardStatus::GpuDeviceRemoved ||
	    st == superslm::SslmForwardStatus::GpuOperationFailed) {
		return SSLM_DEVICE_LOST;
	}
	return SSLM_SEQUENCE_REJECTED;
}
// T-2113 (B5): the FINAL, GPU-decoded per-call result (DecodeStickyTag's own return,
// design Sec4.2's own `sslm_gpu_ready`'s `out_status` channel) -- `Ok` is the only
// non-rejecting value.
//
// T-2114 (S1): `RunLayerLoopGpuFinish`'s own catch block (superslm_gpu.cpp) is the ONE
// path that can deliver a genuine device-level failure here -- it returns
// `GpuDeviceRemoved` (device confirmed gone, or a wait it could not confirm),
// `GpuAllocationFailed` (an allocation failure at the readback, device still alive) or, since
// 1.8.0, `GpuOperationFailed` (any other readback failure) when a real exception
// unwound the finish call. Every OTHER non-Ok value `DecodeStickyTag` can produce is a
// CPU-domain-equivalent guard rejection the GPU dispatch chain itself found (mirroring
// the CPU oracle's own sticky-tag family) -- a healthy-device, per-sequence fact, not a
// device loss. The device-derived statuses map as stated below;
// GpuShaderBinaryStale maps to its deployment-specific public status; every other rejecting
// value maps to SSLM_SEQUENCE_REJECTED, the same status
// MapSubmitRejectionToGpuStatus uses for the identical class of fact discovered before
// submission instead of after.
//
// 1.8.0 (TE-426): the same three-way split as MapSubmitRejectionToGpuStatus -- GpuAllocationFailed
// to SSLM_GPU_ALLOCATION_FAILED; GpuDeviceRemoved and GpuOperationFailed to SSLM_DEVICE_LOST.
SslmGpuStatus MapDecodedStatusToGpuStatus(superslm::SslmForwardStatus st) {
	if (st == superslm::SslmForwardStatus::Ok) return SSLM_OK;
	if (st == superslm::SslmForwardStatus::GpuShaderBinaryStale) {
		return SSLM_GPU_SHADER_BINARY_STALE;
	}
	if (st == superslm::SslmForwardStatus::GpuAllocationFailed) {
		return SSLM_GPU_ALLOCATION_FAILED;
	}
	if (st == superslm::SslmForwardStatus::GpuDeviceRemoved ||
	    st == superslm::SslmForwardStatus::GpuOperationFailed) {
		return SSLM_DEVICE_LOST;
	}
	return SSLM_SEQUENCE_REJECTED;
}

namespace {

// T-2113 (B7, design Sec4.3/Sec7/Sec10 B7): the actual submission logic shared, byte-for-byte,
// between the single-sequence call (`sslm_decode_step_gpu`) and every per-sequence slot of the
// batch call (`sslm_decode_step_batch_gpu`) below -- extracted so the two entry points can never
// silently drift on what "submit one sequence's own dispatch chain" means (StandardsDocument.md
// Sec5.4's own "one implementation, not two copies that could drift" discipline, the identical
// reasoning D-SLM3352 already applied to weight packing). Takes an ALREADY-DECIDED
// `layers_to_issue` (design Sec7: "each sequence taking whole layers only... a sequence reached
// after the budget is exhausted" is a decision the CALLER makes against whichever budget
// (single-call: the call's own `dispatch_budget`; batch: the running batch-wide remainder) --
// this function has no opinion on where the number came from, only on submitting exactly that
// many layers for exactly this one sequence. Caller-ensures `ctx`/`seq`/`seq->model` are non-null,
// non-destroyed, and `seq` is not `Submitted` (every guard above this call's own callers already
// check, per-sequence, before reaching here) -- this function re-validates none of that, matching
// `RunLayerLoopGpuSubmit`'s own "guards are the caller's problem below this layer" convention.
SslmGpuStatus SubmitOneSequenceDecode(SslmGpuContext* ctx, SslmGpuSequenceHandle* seq,
                                       const SslmGpuAdapterHandle* adapter_or_null,
                                       uint32_t layers_to_issue) {
	(void)ctx;  // not read directly -- every resource this call touches is reached through `seq`/
	            // `seq->model`/`adapter_or_null`'s own already-resident buffers (B2/B3/B6), the
	            // identical shape `sslm_decode_step_gpu`'s own pre-B7 body already had.
	SslmGpuModelHandle* model = seq->model;

	// Paged-KV plan §3.4 / §4 item 7, step G3: a pooled holder's budget check, beside the cap
	// check (the ladder's KvCapacityExhausted), before any state moves. It fires only below the cap
	// (limit < cap), so a holder whose limit is the cap sees exactly the dedicated twin's status. The
	// single call and every batch slot share it here, so the batch reports it per sequence with the
	// cap check's batch behaviour.
	superslm_gpu::GpuPagedKvBinding paged{};
	const superslm_gpu::GpuPagedKvBinding* paged_ptr = nullptr;
	if (seq->pool != nullptr) {
		if (seq->limit < seq->context_cap && seq->context_length >= seq->limit) {
			return SSLM_GPU_KV_BUDGET_EXCEEDED;
		}
		// The hot-path map (§3.6): the page this token's row lands in, from the holder's own
		// reserve, while the handle is Idle (§4 item 5). No pool mutex, no allocation. At or past
		// the cap nothing is mapped and the ladder's cap guard refuses as for a dedicated holder.
		if (seq->context_length >= 0 && seq->context_length < seq->context_cap) {
			const SslmGpuStatus map_st = MapPooledPagesThrough(seq, seq->context_length + 1);
			if (map_st != SSLM_OK) return map_st;
		}
		paged.view = superslm::KvPageView{seq->pool->host_mirror.data(), seq->holder->table.get(),
		                                  superslm::kv_pages::HolderMapped(seq->holder),
		                                  seq->pool->geo.page_positions, seq->pool->geo.page_bytes};
		paged.table_resident = seq->table_buf.Get();
		paged.upload_table = seq->table_dirty;
		// sslm_gpu_test_only_arm_view_fault: one shot, on this submission's view only.
		switch (seq->armed_view_fault) {
			case 1: paged.view.mapped_pages = paged.view.mapped_pages > 0 ? paged.view.mapped_pages - 1 : 0; break;
			case 2: paged.view.page_bytes = paged.view.page_bytes > 0 ? paged.view.page_bytes - 1 : 0; break;
			case 3: paged.view.pool_base = nullptr; break;
			case 4: paged.view.page_positions = int64_t{1} << 61; break;
			default: break;
		}
		seq->armed_view_fault = 0;
		paged_ptr = &paged;
	}

	// Sync the handle's own canonical scalar fields into `live_state` (its own `hidden_codes`
	// pointer was set once, at sslm_gpu_seq_create, and needs no per-call refresh).
	seq->live_state.hidden_scale = seq->hidden_scale;
	seq->live_state.layer_index = seq->layer_index;
	seq->live_state.kv_saturation_count = seq->kv_saturation_count;
	seq->live_state.context_length = seq->context_length;

	static const superslm::SslmTensorManifest kEmptyManifest;

	superslm_gpu::GpuAdapterBridge adapter_bridge;
	const superslm_gpu::GpuAdapterBridge* adapter_bridge_ptr = nullptr;
	if (adapter_or_null != nullptr) {
		adapter_bridge.lora_ab_resident = adapter_or_null->lora_ab_buf.Get();
		adapter_bridge.fold_resident = adapter_or_null->fold_buf.Get();
		adapter_bridge.rank = adapter_or_null->rank;
		adapter_bridge.slots = &adapter_or_null->slots[0][0];
		adapter_bridge.num_hidden_layers = model->num_hidden_layers;
		adapter_bridge_ptr = &adapter_bridge;
	}

	superslm_gpu::GpuInFlightToken inflight;
	// T-2432 (Track A step 2/3): q_width threaded explicitly, matching
	// SubmitChunkToFullDepthForG5Bridge's own call site above.
	// T-2578 confirmation remedy: pass the same stable model identity as the sibling G5
	// chunk path. This external-residency path does not consume the process-global caches
	// today, but its call contract is now complete if that routing ever changes.
	uint64_t model_generation = 0;
	std::memcpy(&model_generation, model->content_hash.data(), sizeof(model_generation));
	const superslm::SslmForwardStatus submit_status = superslm_gpu::RunLayerLoopGpuSubmit(
	    seq->live_state, /*layers=*/nullptr, model->num_hidden_layers, layers_to_issue,
	    model->hidden_size, model->head_dim, model->num_key_value_heads, model->intermediate_size,
	    seq->context_cap, kEmptyManifest, KvHostBase(seq), KvHostBytes(seq),
	    seq->pool != nullptr ? seq->pool->pool_buf.Get() : seq->kv_buf.Get(),
	    seq->pool != nullptr ? &seq->pool->kv_needs_resume_barrier : &seq->kv_needs_resume_barrier,
	    &inflight, model->weights_buf.Get(),
	    model->rope_cos_buf.Get(), model->rope_sin_buf.Get(), model->has_rope_tables,
	    model->rope_cos_elem_count, model->rope_sin_elem_count, adapter_bridge_ptr,
	    /*q_width=*/static_cast<size_t>(model->num_attention_heads) * model->head_dim,
	    /*out_q_codes=*/nullptr, /*out_q_codes_capacity=*/0, model_generation,
	    model->has_qk_norm, paged_ptr);
	seq->last_ladder_status = inflight ? superslm::SslmForwardStatus::Ok : submit_status;

	if (!inflight) {
		// Rejected before submission (a guard, or an exception) -- seq/host state untouched,
		// still Idle, exactly RunLayerLoopGpu's own existing contract. A pooled holder keeps its
		// table dirty, so its next submission uploads it.
		return MapSubmitRejectionToGpuStatus(submit_status);
	}
	if (seq->pool != nullptr) seq->table_dirty = false;  // this submission's head uploaded it
	seq->in_flight = std::move(inflight);
	// T-2113 (design Sec4.2/Sec9, D-SLM3417), T-2243 review finding 2 (D-SLM4113) and T-2124
	// (D-SLM3446 P0-3): the model's, the process's and the adapter's counts, and the pooled count
	// and the pool's holder pointer for a pooled handle, raised by the counting accessor (G3, §4.4),
	// which also records `in_flight_adapter` for sslm_gpu_ready's lowering.
	SetGpuSeqState(seq, superslm_gpu::SslmSequenceGpuState::Submitted, adapter_or_null);
	return SSLM_OK;
}
}  // namespace

// Design Sec4.3/Sec6.2 (T-2113 B5): "records up to dispatch_budget dispatches ... into
// ONE command list, closes it, submits it, and returns without waiting for the fence."
// Routes weights/RoPE through `model`'s own resident buffers (B2, `weights_buf`/
// `rope_cos_buf`/`rope_sin_buf` -- the external-weights/external-rope bridge B5 adds to
// `RunLayerLoopGpuSubmit`) and K/V through `seq`'s own dedicated buffer (B3, the existing
// external-KV bridge) -- this is the production measurement path that retires
// `g_resident_rope`/`g_resident_weights` for every call that reaches here (D-SLM3362):
// neither process-global cache is ever read or written on this path.
SslmGpuStatus sslm_decode_step_gpuImpl(SslmGpuContext* ctx, SslmGpuSequenceHandle* seq,
                                    const SslmGpuAdapterHandle* adapter_or_null,
                                    uint32_t dispatch_budget) {
	// Design Sec8/Sec10 B6: per-sequence adapter binding is a call argument, checked at every
	// call (design Sec5.2: "checked at every sslm_decode_step_gpu/_batch_gpu call... a
	// pointer-equality check against the handle each side already carries, not a re-hash").
	// T-2113 (B6b, design Sec8): the residency and guard checks below are real, AND the GPU
	// dispatch-recording path this call submits through (RunLayerLoopGpuSubmit) now reads the
	// adapter's own resident buffers when `adapter_or_null` validates -- the GEMM-site
	// delta-application dispatches, via the GpuAdapterBridge built below.
	if (!ctx || !seq || !seq->model || seq->ctx != ctx) {  // T-2114 (S2): ->destroyed reads
	                                     // removed, see the sequence/model handles' own field
	                                     // comments. T-2124 (D-SLM3446 P1-4): `seq->ctx != ctx`.
		return SSLM_SEQUENCE_KV_BUFFER_MISMATCH;  // no channel exists for a malformed handle
		                                           // at this call besides the structural-
		                                           // invariant status (same "no more specific
		                                           // status" disposition as the helpers above).
	}
	if (adapter_or_null && adapter_or_null->model != seq->model) {  // T-2114 (S2): ->destroyed
	                                                                  // read removed.
		return SSLM_ADAPTER_MODEL_MISMATCH;  // design Sec5.2/Sec9: pointer-equality against the
		                                      // model handle each side already carries.
	}
	if (adapter_or_null && adapter_or_null->ctx != ctx) {  // T-2124 (D-SLM3446 P1-4): a bound
	                                                        // adapter mapped against a different
	                                                        // context names a different device.
		return SSLM_SEQUENCE_KV_BUFFER_MISMATCH;  // same "no channel exists for a malformed
		                                           // handle" disposition as the seq check above.
	}
	// The same-sequence claim, before the state read (t2112 dim3 M2): a second thread's decode of
	// this handle is refused here, so only one call at a time runs the checks below and submits.
	const SeqDecodeClaim claim(seq);
	if (!claim.held()) {
		return SSLM_BUSY;  // the state machine's own collision signal; nothing was touched.
	}
	const bool is_submitted = seq->state() == superslm_gpu::SslmSequenceGpuState::Submitted;
	if (!superslm_gpu::CallProceedsOrBusy_DecodeStepGpu(is_submitted)) {
		return SSLM_BUSY;  // design Sec4.2's own state table, unchanged policy.
	}
	SslmGpuModelHandle* model = seq->model;
	uint32_t layers_to_issue = 0;
	const superslm_gpu::SslmGpuStatus plan = superslm_gpu::PlanDispatchBudgetGpu(
	    dispatch_budget, model->num_hidden_layers, seq->layer_index, &layers_to_issue,
	    model->dispatches_per_layer);
	if (plan == superslm_gpu::SslmGpuStatus::DispatchBudgetTooSmall) {
		return SSLM_DISPATCH_BUDGET_TOO_SMALL;
	}
	// EntrySubmitGuard (§4.4): the decode reaches the singleton; scope by holder (any sequence
	// Submitted for a pooled holder, a pooled one for a dedicated holder), plus the singleton
	// ledger term. After the dispatch-budget refusal (§4.4's precedence: "the decode budget refusal
	// precedes the guard"; cell 3.6d/G3:precedence:budget_too_small), before the first state change.
	if (const SslmGpuStatus g = EntrySubmitGuard(ScopeForHolder(seq), GuardDevice::kReachesSingleton);
	    g != SSLM_OK) {
		return g;
	}
	seq->bind_eligible = false;

	// T-2113 (B7): submission itself is shared, byte-for-byte, with every per-sequence slot
	// of sslm_decode_step_batch_gpu (below) -- see SubmitOneSequenceDecode's own header
	// comment. This call has already decided layers_to_issue against its OWN full
	// dispatch_budget (immediately above); the batch call decides the identical quantity
	// against a running batch-wide remainder instead -- the only difference between the two
	// call sites, and it lives entirely in the budget arithmetic surrounding this shared call,
	// never inside it.
	return SubmitOneSequenceDecode(ctx, seq, adapter_or_null, layers_to_issue);
}

// Design Sec4.3/Sec7/Sec10 B7 (T-2113 B7, repaired signature per the T-2111 fold, D-SLM3344):
// "records n sequences' own, independent, per-sequence dispatch chains ... the batch call
// changes submission grouping, never any sequence's own numerics." `dispatch_budget` is
// BATCH-WIDE (design Sec7): consumed by sequences strictly in `seqs[]`'s own array order, each
// sequence taking whole layers only from whatever remains when its own turn comes. A guard
// rejection for `seqs[i]` (a malformed handle, an adapter/model mismatch, a `Busy` sequence)
// does not abort the batch -- `out_statuses[i]` carries that sequence's own outcome and
// recording continues with `seqs[i+1]`. A sequence reached once the running remainder cannot
// cover even one whole layer records zero dispatches and reads `SSLM_BATCH_BUDGET_EXHAUSTED`
// (never `SSLM_DISPATCH_BUDGET_TOO_SMALL`, which design Sec9's own channel table reserves for
// the single-sequence call) -- its own state stays `Idle`, exactly as if it had never been
// passed to this call at all. The function's own RETURN VALUE is call-level only (design
// Sec4.3): whether the call proceeded at all (`Ok`) -- never a reduction of `out_statuses`.
//
// **Named limitation, stated per Brunel's own "stop at genuine limits, honest stage report"
// discipline, not silently absorbed**: this implementation submits each recorded sequence
// through the IDENTICAL `RunLayerLoopGpuSubmit` call the single-sequence path uses --
// `SubmitOneSequenceDecode`, shared verbatim -- which means every sequence in the batch still
// gets its OWN `ExecuteCommandLists`/`Signal` pair underneath, not the ONE-command-list fusion
// design Sec7's own text describes ("records n sequences ... into ONE command list and
// submits/fences them together"). The CORRECTNESS contract this section's own gate names
// (design Sec10 B7: "per sequence, output bit-identical to that same sequence decoded alone")
// is satisfied by construction -- reusing the exact same, already-proven submission call for
// every sequence makes bit-identity to the serial case a property of the code being literally
// the same code, not a separately-proven claim -- and the per-sequence `out_statuses`/
// batch-wide-budget/mixed-adapter semantics are fully and correctly implemented. What is NOT
// realized is the submission-side throughput amortization design Sec7 names as the batching
// mechanism's OWN rationale (a real, but explicitly out-of-scope-for-this-gate, cost: design
// Sec13 states the batch call's own throughput is "UNDERIVED... an empirical question," never
// a design-time claim, and Sec10 B7's own gate text is itself "a correctness proof, not a
// throughput proof"). §14's own throughput measurement, below, reports this honestly rather
// than assuming batching helps.
//
// **A REAL device hang, found at the bench, is why this is not merely a missed optimization**:
// every 1.0-API decode call -- single or batched -- ultimately submits through the process's ONE
// shared `harness::GetDevice()` singleton (every `SslmGpuContext::device`, B1, is used only for
// RESIDENCY uploads at map/create time, never for the decode dispatch chain itself, which the
// pre-1.0 substrate's own `RunLayerLoopGpuSubmit` still routes through the singleton).
// `ID3D12CommandAllocator::Reset()` is undefined behavior while a command list allocated from it
// is still executing on the GPU -- calling `SubmitOneSequenceDecode` for `seqs[1]` before
// `seqs[0]`'s own fence has signaled resets that SAME shared allocator out from under a submission
// that may still be executing. A first draft of this function left every recorded sequence
// Submitted (undrained) until the whole loop finished, matching the design's own async-batch
// framing literally -- and hung the test process indefinitely on the very first multi-sequence
// batch it ever ran (`t2113_b7_batch_smoke.cpp`'s own Gate 1a, reproduced by execution: 12.12
// CPU-seconds, motionless, for several minutes -- a real device-level stall, not a slow
// computation). Fixed below: each sequence is drained via the public `sslm_gpu_ready(block=1)`
// the INSTANT its own submission succeeds, before the loop's next iteration can touch the shared
// allocator again -- at most one sequence is ever Submitted against the device at once. This is
// the same constraint that makes the "ONE command list" fusion design Sec7 describes a genuine
// correctness requirement for a from-scratch implementation, not merely a throughput nicety --
// this implementation avoids needing that fusion by never having two sequences in flight
// simultaneously in the first place, at the cost named above (no cross-sequence submission
// overlap, so no submission-side amortization either).
SslmGpuStatus sslm_decode_step_batch_gpuImpl(SslmGpuContext* ctx, SslmGpuSequenceHandle* const* seqs,
                                          const SslmGpuAdapterHandle* const* adapters_or_null,
                                          uint32_t n_sequences, uint32_t dispatch_budget,
                                          SslmGpuStatus* out_statuses) {
	if (!ctx) {
		return SSLM_SEQUENCE_KV_BUFFER_MISMATCH;  // call-level: no live object to report through,
		                                           // same disposition every other null-ctx site
		                                           // in this file already uses.
	}
	if (n_sequences > 0 && (!seqs || !out_statuses)) {
		// A non-zero count with a null seqs/out_statuses array is a caller contract violation
		// this call cannot record anything against -- no per-sequence channel exists to report
		// through (out_statuses is itself null), so this is call-level, matching the "no live
		// object to report through" disposition every other malformed-argument site uses.
		return SSLM_SEQUENCE_KV_BUFFER_MISMATCH;
	}
	// n_sequences == 0: a legal, vacuous call (design Sec4.3 states no floor on n_sequences) --
	// nothing to record, nothing to report, proceeds and returns Ok.

	// EntrySubmitGuard (§4.4): after the call-level refusals, before the member loop. The batch is
	// pooled (scope: any sequence Submitted) when any member is pooled, read before any member is
	// validated; a null member or a member of another context counts as dedicated for this test.
	// It reaches the singleton, so the ledger term applies. Refused, the batch's two channels both
	// carry the guard's status: the call returns it and every `out_statuses[i]` is written with it,
	// and no member is validated or changed (rev 8, N1).
	{
		bool any_pooled = false;
		for (uint32_t i = 0; i < n_sequences; ++i) {
			const SslmGpuSequenceHandle* m = seqs[i];
			if (m != nullptr && m->ctx == ctx && m->pool != nullptr) any_pooled = true;
		}
		const SslmGpuStatus g = EntrySubmitGuard(any_pooled ? GuardScope::kAny : GuardScope::kPooled,
		                                         GuardDevice::kReachesSingleton);
		if (g != SSLM_OK) {
			for (uint32_t i = 0; i < n_sequences; ++i) out_statuses[i] = g;
			return g;
		}
	}

	uint32_t remaining_budget = dispatch_budget;
	for (uint32_t i = 0; i < n_sequences; ++i) {
		SslmGpuSequenceHandle* seq = seqs[i];
		const SslmGpuAdapterHandle* adapter_or_null = adapters_or_null ? adapters_or_null[i] : nullptr;

		// Per-sequence guard ladder -- IDENTICAL checks and IDENTICAL statuses to the
		// single-sequence call (design Sec9: AdapterModelMismatch's own channel table entry
		// names `out_statuses[i]` explicitly as the batch call's own per-sequence channel for
		// exactly this check). A guard rejection for seqs[i] does NOT abort the batch (design
		// Sec7's own "records every sequence's chain up to... that sequence's own guard
		// failure... skips that sequence's own remaining dispatches, and continues recording
		// the next sequence") -- `continue` to seqs[i+1], remaining_budget UNCHANGED (a
		// rejected sequence consumes no budget, since nothing was recorded for it).
		if (!seq || !seq->model || seq->ctx != ctx) {  // T-2114 (S2): ->destroyed reads removed,
		                            // see the sequence/model handles' own field comments.
		                            // T-2124 (D-SLM3446 P1-4): `seq->ctx != ctx`.
			out_statuses[i] = SSLM_SEQUENCE_KV_BUFFER_MISMATCH;
			continue;
		}
		if (adapter_or_null && adapter_or_null->model != seq->model) {  // T-2114 (S2): ->destroyed
		                                                                 // read removed.
			out_statuses[i] = SSLM_ADAPTER_MODEL_MISMATCH;
			continue;
		}
		if (adapter_or_null && adapter_or_null->ctx != ctx) {  // T-2124 (D-SLM3446 P1-4): a bound
		                                                        // adapter mapped against a
		                                                        // different context.
			out_statuses[i] = SSLM_SEQUENCE_KV_BUFFER_MISMATCH;
			continue;
		}
		// The same-sequence claim (t2112 dim3 M2), held for this slot's submission and its drain:
		// another thread's decode of this handle makes the slot Busy, untouched.
		const SeqDecodeClaim claim(seq);
		if (!claim.held()) {
			out_statuses[i] = SSLM_BUSY;
			continue;
		}
		const bool is_submitted = seq->state() == superslm_gpu::SslmSequenceGpuState::Submitted;
		if (!superslm_gpu::CallProceedsOrBusy_DecodeStepGpu(is_submitted)) {
			out_statuses[i] = SSLM_BUSY;
			continue;
		}
		seq->bind_eligible = false;

		// Design Sec7: "dispatch_budget is consumed by sequences strictly in the recording
		// order... each sequence taking whole layers only... from whatever remains" -- planned
		// against the RUNNING remainder, never the call's own original dispatch_budget past
		// the first sequence. Zero layers available (whether because the ORIGINAL budget never
		// covered even seqs[0]'s own first layer, or because earlier sequences in this same
		// call already consumed it) is uniformly SSLM_BATCH_BUDGET_EXHAUSTED at the batch call
		// -- design Sec9's own table reserves SSLM_DISPATCH_BUDGET_TOO_SMALL for the
		// single-sequence call alone, never naming it as a batch-call channel.
		// Paged-KV plan §3.4 / cell 5.2 (5.7 at G3): a pooled holder at its limit (below the cap) is
		// refused SSLM_GPU_KV_BUDGET_EXCEEDED in its own slot whatever the running dispatch remainder,
		// consuming none of it, so the other slots proceed exactly as beside a holder at its cap.
		// Checked before the dispatch planning, so the status is the budget's even when an earlier
		// slot used the whole remainder (SubmitOneSequenceDecode repeats the check for the single call).
		if (seq->pool != nullptr && seq->limit < seq->context_cap && seq->context_length >= seq->limit) {
			out_statuses[i] = SSLM_GPU_KV_BUDGET_EXCEEDED;
			continue;
		}
		uint32_t layers_to_issue = 0;
		const superslm_gpu::SslmGpuStatus plan = superslm_gpu::PlanDispatchBudgetGpu(
		    remaining_budget, seq->model->num_hidden_layers, seq->layer_index, &layers_to_issue,
		    seq->model->dispatches_per_layer);
		if (plan == superslm_gpu::SslmGpuStatus::DispatchBudgetTooSmall) {
			out_statuses[i] = SSLM_BATCH_BUDGET_EXHAUSTED;  // seq->state stays Idle -- nothing
			                                                  // recorded, nothing submitted.
			continue;
		}

		out_statuses[i] = SubmitOneSequenceDecode(ctx, seq, adapter_or_null, layers_to_issue);
		if (out_statuses[i] == SSLM_OK) {
			// Design Sec7's own budget arithmetic: exactly this sequence's own consumed
			// dispatches (layers_to_issue whole layers, using this model's 24-or-25 divisor) leave
			// the running remainder for seqs[i+1..]. A submission REJECTED after planning
			// succeeded (e.g. a device-lost mid-batch, design Sec9's own DeviceLost row: "every
			// not-yet-reached out_statuses[i] reading DeviceLost") consumes no budget either --
			// only a call that actually reached SSLM_OK genuinely used GPU dispatch slots.
			// Model mapping derives dispatches_per_layer once; planning and subtraction both
			// read that value so the two operations cannot drift by architecture.
			remaining_budget -= layers_to_issue * seq->model->dispatches_per_layer;

			// T-2113 (B7): a REAL device hang found at the bench, fixed here, not merely
			// worked around -- see this function's own header comment for the fuller account.
			// `RunLayerLoopGpuSubmit` always operates on the process's ONE shared
			// `harness::GetDevice()` singleton's own command allocator/list/queue (every 1.0
			// context's own `SslmGpuContext::device` is used only for RESIDENCY uploads, B1/B2,
			// never for the decode dispatch chain itself). `ID3D12CommandAllocator::Reset()` is
			// undefined behavior while a command list allocated from it is still executing on
			// the GPU (D3D12's own documented contract) -- submitting seqs[i+1]'s own dispatches
			// (which calls `dev.alloc->Reset()` again, unconditionally, at the top of the NEXT
			// `RunLayerLoopGpuSubmit`) BEFORE seqs[i]'s own fence has signaled is exactly that:
			// two submissions against the SAME shared allocator with no fence boundary between
			// them. Reproduced by execution: `t2113_b7_batch_smoke.cpp`'s own Gate 1a (a
			// batch-of-3, back-to-back submissions with no drain between them) hung the process
			// indefinitely (12.12 CPU-seconds, unmoving, for several minutes -- a genuine device-
			// level stall, not a slow computation) the first time this loop left more than one
			// sequence Submitted at once. Fixed: each sequence is drained (via the EXISTING
			// public `sslm_gpu_ready(block=1)`, never a second, parallel implementation) the
			// MOMENT its own submission succeeds, before the loop's next iteration ever calls
			// `dev.alloc->Reset()` again -- at most one sequence is ever Submitted against the
			// shared device at a time. This is also the concrete, now-necessary (not merely
			// named-as-a-cost) form of this function's own "no one-command-list fusion" limit:
			// the batch call is observably synchronous per sequence internally, though `seq`'s
			// own state still transitions through `Submitted` correctly, so a caller polling via
			// `sslm_gpu_ready` afterward always finds `ready=1` immediately -- a legal realization
			// of the async contract (design Sec4.2 never requires a call to still be pending).
			int32_t drained_ready = 0;
			SslmGpuStatus drained_status = SSLM_OK;
			const SslmGpuStatus ready_call_status =
			    sslm_gpu_ready(ctx, seq, /*block=*/1, &drained_ready, &drained_status);
			out_statuses[i] = (ready_call_status == SSLM_OK) ? drained_status : ready_call_status;
		}
		if (out_statuses[i] == SSLM_DEVICE_LOST) {
			// Design Sec9's own DeviceLost row for the batch call: "every not-yet-reached
			// out_statuses[i] reading DeviceLost" -- a device removal discovered mid-batch must
			// not let the REST of the batch's own not-yet-recorded sequences silently proceed
			// as if the device were still healthy (their own dispatches, if recorded, would
			// execute against a lost device with undefined results). Every remaining slot is
			// marked DeviceLost and recording stops -- this is the one case design Sec7's own
			// per-sequence-independence text does not apply to, because the SHARED device
			// itself, not any one sequence's own state, is what failed.
			for (uint32_t j = i + 1; j < n_sequences; ++j) out_statuses[j] = SSLM_DEVICE_LOST;
			break;
		}
	}
	return SSLM_OK;  // call-level: recording proceeded (design Sec4.3) -- per-sequence outcomes
	                  // are exclusively in out_statuses, never folded into this return value.
}

// Design Sec4.2 (T-2113 B5): "block(=0): a pure poll -- if the fence has not yet
// signaled, *out_ready=0, no state change ... block(=1): blocks the calling thread on
// the fence rather than returning *out_ready=0 -- the only call this design allows to
// legitimately fence-wait." `Idle`/`Completed` (nothing outstanding): `*out_ready=1`
// immediately, `*out_status=Ok`.
SslmGpuStatus sslm_gpu_readyImpl(SslmGpuContext* ctx, SslmGpuSequenceHandle* seq, int32_t block,
                               int32_t* out_ready, SslmGpuStatus* out_status) {
	if (out_ready) *out_ready = 0;
	if (out_status) *out_status = SSLM_OK;
	if (!ctx || !seq || seq->ctx != ctx) {  // T-2114 (S2): ->destroyed read removed, see sequence
	                                         // handle's own field comment. T-2124 (D-SLM3446
	                                         // P1-4): `seq->ctx != ctx`.
		return SSLM_SEQUENCE_KV_BUFFER_MISMATCH;  // no channel exists for a malformed handle
	}
	seq->bind_eligible = false;
#if defined(SUPERSLM_ENABLE_GPU_CHUNK_DISPATCH_INSTRUMENT)
	++superslm_test::g_gpu_ready_poll_count_probe;
#endif
	if (seq->state() != superslm_gpu::SslmSequenceGpuState::Submitted) {
		// design Sec4.2: "Idle/Completed: *out_ready=1 immediately, *out_status=Ok
		// (nothing outstanding)."
		if (out_ready) *out_ready = 1;
		return SSLM_OK;
	}

	// T-2243 review finding 10, Observation (D-SLM4113): `seq->in_flight == nullptr` while
	// `seq->state == Submitted` is an internal-invariant violation reachable only by a caller
	// bypassing this handle's own lifecycle (RunLayerLoopGpuFinish's `!inflight` guard exists
	// for exactly this, its own comment: "caller error: no live token") -- never a genuine
	// device/allocation fault. Checked here, before Finish is ever called, so this call's own
	// `decoded` value stays exactly what a real device-derived fault would produce (GpuDeviceRemoved/
	// GpuAllocationFailed/GpuOperationFailed only ever arrive from inside Finish's try/catch, past
	// this point) and O2's status surfaces the caller-error cause directly instead of being folded
	// into a device-derived status by MapDecodedStatusToGpuStatus below (before 1.8.0,
	// GpuAllocationFailed; since 1.8.0 Finish returns GpuOperationFailed for it).
	if (!seq->in_flight) {
		if (out_status) *out_status = SSLM_SEQUENCE_REJECTED;
		return SSLM_OK;
	}

	int32_t ready = 0;
	const superslm::SslmForwardStatus decoded = superslm_gpu::RunLayerLoopGpuFinish(
	    std::move(seq->in_flight), seq->live_state, KvHostBase(seq), block, &ready);
	if (!ready) {
		// T-2243 (O2, Mendeleev F-5, plan Sec10 Phase 2 O2): RunLayerLoopGpuFinish's own
		// `!inflight` guard is now unreachable from here (checked above) -- this branch only
		// ever sees the genuine device-derived faults Finish's try/catch can produce, plus the
		// ordinary poll case immediately below. `decoded != Ok` on this branch is never the
		// ordinary poll case (an unsignaled fence reports `Ok`, not a rejection), so it is
		// surfaced through `out_status` instead of the silent SSLM_OK/*out_ready=0 this call
		// used to return unconditionally.
		if (decoded != superslm::SslmForwardStatus::Ok) {
			if (out_status) *out_status = MapDecodedStatusToGpuStatus(decoded);
			return SSLM_OK;
		}
		// Still Submitted -- a non-blocking poll against an unsignaled fence, design
		// Sec4.2's own "no state change" case. `seq->in_flight` is untouched by Finish
		// on this path (still owned by this handle, still pollable next call).
		return SSLM_OK;
	}

	// Fence signaled (or block=1 waited for it) -- collapse Submitted -> Idle in one
	// call, design Sec4.2's own contract, and copy the decoded result back into this
	// handle's own canonical fields.
	// RunLayerLoopGpuFinish moved from `seq->in_flight` and freed the record on this path, so the
	// token is already empty (G3: it moves from the token only once the fence reached its value).
	// T-2113 (D-SLM3417), T-2243 review finding 2 (D-SLM4113), T-2124 (D-SLM3446 P0-3): the move
	// out of Submitted lowers exactly the counts the submission raised -- the model's, the
	// process's, the adapter's (the fence has signaled, so the GPU is done reading the adapter's
	// buffers) and, for a pooled handle, the pooled count and the pool's holder pointer -- through
	// the counting accessor (G3, §4.4).
	SetGpuSeqState(seq, superslm_gpu::SslmSequenceGpuState::Idle, nullptr);
	seq->hidden_scale = seq->live_state.hidden_scale;
	seq->layer_index = seq->live_state.layer_index;
	seq->kv_saturation_count = seq->live_state.kv_saturation_count;
	seq->context_length = seq->live_state.context_length;
	if (out_ready) *out_ready = 1;
	if (out_status) *out_status = MapDecodedStatusToGpuStatus(decoded);
	return SSLM_OK;
}

namespace {
// T-2895 (closing TE-362): untrusted-input validation on a restored schema binding, mirroring
// sslm_seq_restore's own C1 fix (src/sslm_abi.cpp) -- the restored walk-state is bounded against the
// RESOLVED schema's own state_count before it is ever used to index mask_pages, and a bound index this
// model does not have is refused. Unbound (restored_schema_index < 0) passes only with the unused-walk
// sentinel. Paged KV (G5): factored out of sslm_gpu_seq_restoreImpl unchanged, so both GPU restore verbs
// run the same two checks (and tests/t2899-schema-deadend-red-suite/make_mut_slm5_bounds.py, re-anchored
// here, mutates them for both).
bool RestoredSchemaBindingValid(const SslmGpuModelHandle* model, int32_t restored_schema_index,
                                uint32_t restored_walk_state) {
	if (restored_schema_index >= 0) {
		// T-2917 (folding TE-370 S3, D-SLM7600): the `>= model->schemas.Count()` pre-check
		// this comment used to gate on is removed -- provably dead, not merely redundant.
		// SchemaMasksTable::ByIndex (include/superslm/schema_masks.h) is itself bounds-safe
		// (`index < entries_.size() ? &entries_[index] : nullptr`), so no `restored_schema_index`
		// this check could have caught can ever make it return non-null; the null-check below
		// already refuses every input the removed check refused, by the same mechanism. T-2916
		// confirmed by execution (mutant deletion): the removed check's own mutant survives
		// (failures=0) because the adjacent null-check kills every input that would exercise it.
		// The null-check IS the bound.
		const superslm::SchemaEntry* resolved_entry =
		    model->schemas.ByIndex(static_cast<size_t>(restored_schema_index));
		if (!resolved_entry || restored_walk_state >= resolved_entry->state_count) {
			return false;
		}
	} else if (restored_walk_state != 0xFFFFFFFFu) {
		// Symmetric malformation, mirroring sslm_seq_restore's own identical check: an unbound
		// blob (schema index < 0) must carry the unused-walk sentinel, never a real state id.
		return false;
	}
	return true;
}

// Paged-KV plan (rev 16.2) §4 item 10, step G5: the 'SLM6' blob, which only a POOLED sequence writes
// (the writer rule: a dedicated sequence keeps writing 'SLM5', SaveGpuSequenceState, unchanged).
// "SLM5 fields, plus kv_mode, budget, origin and the canonical [0, L') K/V gather of §3.7". The plan fixes
// the fields; this is the layout, the one the GPU red suite pins (tests/paged-kv-gpu/pkvg_slm6_layout.h):
//   [0, 120)    SLM5's header (superslm_gpu.cpp GpuSeqBlobHeader, field for field), magic 'SLM6', with
//               hidden_codes_size = the residual's bytes and workspace_size = L' * bytes_per_token (the
//               K/V rows the blob carries, as SLM5's workspace_size is the K/V bytes it carries)
//   [120, 132)  SLM5's twelve-byte tail (bound_schema_index i32, dfa_walk_state u32, ready u32)
//   132         kv_mode u32 (1 budget: the writer rule makes every SLM6 a budget blob)
//   136         budget i32 (the declared budget)
//   140         origin i64
//   148         the residual (hidden_codes_size bytes)
//   148 + R     kv_positions u64 = L' = context_length + (0 < layer_index < num_hidden_layers ? 1 : 0)
//   156 + R     the rows [0, L') in canonical layout [layer][K|V][kv_head][pos < L'][d] (int8)
// L' is §3.7's rule applied as the CPU twin applies it: row `context_length` is in the blob only for a
// mid-token save. The GPU rests a full-depth token at layer_index == num_hidden_layers with context_length
// already advanced -- a completed token, so row `context_length` holds no state (the CPU rests the same
// history at layer_index 0, and both write L' = context_length). The blob still carries layer_index, so a
// restore keeps the GPU's rest. The rows are gathered from the pool's host mirror (complete by G24)
// through the holder's table, and at a mid-token save's row `context_length` the layers >= layer_index
// are written as zero bytes whatever the page holds (§3.7), so the blob is a function of the sequence's
// valid state alone. No relation between the per-site counters and the aggregate is validated (§3.7).
constexpr uint32_t kGpuSeqBlobMagicSlm6 = 0x364D4C53u;  // "SLM6" little-endian u32
constexpr size_t kSlm6OffLayerIndex = 4;
constexpr size_t kSlm6OffScaleM = 8;
constexpr size_t kSlm6OffScaleE = 16;
constexpr size_t kSlm6OffKvSaturation = 24;
constexpr size_t kSlm6OffContextLength = 32;
constexpr size_t kSlm6OffHiddenCodesSize = 40;
constexpr size_t kSlm6OffWorkspaceSize = 48;
constexpr size_t kSlm6OffModelHash = 56;
constexpr size_t kSlm6OffKvLandingSaturation = 88;
constexpr size_t kSlm6OffKChannelLandingSaturation = 96;
constexpr size_t kSlm6OffRopeQSaturation = 104;
constexpr size_t kSlm6OffRopeKSaturation = 112;
constexpr size_t kSlm6OffSchemaIndex = 120;
constexpr size_t kSlm6OffWalkState = 124;
constexpr size_t kSlm6OffReady = 128;
constexpr size_t kSlm6OffKvMode = 132;
constexpr size_t kSlm6OffBudget = 136;
constexpr size_t kSlm6OffOrigin = 140;
constexpr size_t kSlm6FixedBytes = 148;
constexpr size_t kSlm6KvPositionsBytes = 8;
constexpr uint32_t kSlm6KvModeBudget = 1;
static_assert(kSlm6OffModelHash + superslm::kIntegrityHashBytes == kSlm6OffKvLandingSaturation, "SLM6 layout");

template <typename T>
void Slm6Put(uint8_t* blob, size_t off, T v) {
	std::memcpy(blob + off, &v, sizeof(T));
}
template <typename T>
T Slm6Get(const uint8_t* blob, size_t off) {
	T v{};
	std::memcpy(&v, blob + off, sizeof(T));
	return v;
}

// The 4-byte magic, or 0 when the blob is too short to hold one.
uint32_t GpuSeqBlobMagic(const void* blob, size_t blob_size) {
	if (!blob || blob_size < sizeof(uint32_t)) return 0;
	return Slm6Get<uint32_t>(static_cast<const uint8_t*>(blob), 0);
}

// One K/V row's bytes, and one token's across every (layer, half, kv_head) run, at the GPU's int8 element.
size_t GpuKvRuns(const SslmGpuModelHandle* model) {
	return static_cast<size_t>(model->num_hidden_layers) * 2u * static_cast<size_t>(model->num_key_value_heads);
}

// The pooled holder's row `position` of run `run` in the pool's host mirror (§3.1's in-page layout
// [run][pos < B][d]). The holder maps the position's page.
uint8_t* PooledMirrorRow(const SslmGpuKvPoolHandle* pool, const superslm::kv_pages::PageHolder* holder, size_t run,
                         int64_t position, size_t head_dim) {
	const int64_t B = pool->geo.page_positions;
	const uint32_t page = superslm::kv_pages::HolderTableEntryForAddress(holder, static_cast<uint32_t>(position / B));
	return superslm::kv_pages::PageBase(pool->pages, page) + run * static_cast<size_t>(B) * head_dim +
	       static_cast<size_t>(position % B) * head_dim;
}

// §4 item 10: a pooled sequence's save, the 'SLM6' writer. The size contract is SaveGpuSequenceState's:
// a null or too-small `out_blob` reports the required size through *out_blob_size and returns false, and
// the caller maps false to the status the 'SLM5' save returns for it.
// L' (§3.7, as the CPU twin applies it): row `context_length` is carried only mid-token. layer_index ==
// num_hidden_layers is the GPU's rest after a completed token, whose row `context_length` holds no state.
int64_t Slm6Lprime(int64_t context_length, uint32_t layer_index, uint32_t num_hidden_layers) {
	return context_length + ((layer_index > 0 && layer_index < num_hidden_layers) ? 1 : 0);
}

bool SavePooledSlm6(const SslmGpuSequenceHandle* seq, void* out_blob, size_t* out_blob_size) {
	if (!out_blob_size) return false;
	// An internal invariant failing reports size 0, so a size query never returns a stale size (F4).
	if (seq->pool == nullptr || seq->holder == nullptr || seq->model == nullptr) {
		*out_blob_size = 0;
		return false;
	}
	const SslmGpuModelHandle* model = seq->model;
	const SslmGpuKvPoolHandle* pool = seq->pool;
	const superslm::SequenceLayerState& st = seq->live_state;
	const size_t D = model->head_dim;
	const size_t runs = GpuKvRuns(model);
	const int64_t L = st.context_length;
	const uint32_t layer_index = st.layer_index;
	const int64_t lprime = Slm6Lprime(L, layer_index, model->num_hidden_layers);
	if (L < 0 || lprime > seq->context_cap) {
		*out_blob_size = 0;
		return false;
	}
	const size_t residual = seq->hidden_codes.size();
	const size_t rows_bytes = static_cast<size_t>(lprime) * runs * D;
	const size_t required = kSlm6FixedBytes + residual + kSlm6KvPositionsBytes + rows_bytes;
	if (!out_blob || *out_blob_size < required) {
		*out_blob_size = required;  // report the size a real call needs, never truncate
		return false;
	}
	// Every row read lies in a page the holder maps: rows [0, L') and, mid-token, row L's layers below
	// layer_index (the token being computed wrote them).
	if (superslm_gpu::kv_pool::PagesToCover(lprime, pool->geo.page_positions) >
	    superslm::kv_pages::HolderMapped(seq->holder)) {
		*out_blob_size = 0;
		return false;
	}
	uint8_t* dst = static_cast<uint8_t*>(out_blob);
	Slm6Put<uint32_t>(dst, 0, kGpuSeqBlobMagicSlm6);
	Slm6Put<uint32_t>(dst, kSlm6OffLayerIndex, layer_index);
	Slm6Put<int64_t>(dst, kSlm6OffScaleM, st.hidden_scale.m);
	Slm6Put<int64_t>(dst, kSlm6OffScaleE, st.hidden_scale.e);
	Slm6Put<uint64_t>(dst, kSlm6OffKvSaturation, st.kv_saturation_count);
	Slm6Put<int64_t>(dst, kSlm6OffContextLength, L);
	Slm6Put<uint64_t>(dst, kSlm6OffHiddenCodesSize, static_cast<uint64_t>(residual));
	Slm6Put<uint64_t>(dst, kSlm6OffWorkspaceSize, static_cast<uint64_t>(rows_bytes));
	std::memcpy(dst + kSlm6OffModelHash, model->content_hash.data(), superslm::kIntegrityHashBytes);
	Slm6Put<uint64_t>(dst, kSlm6OffKvLandingSaturation, st.kv_landing_saturation_count);
	Slm6Put<uint64_t>(dst, kSlm6OffKChannelLandingSaturation, st.k_channel_landing_saturation_count);
	Slm6Put<uint64_t>(dst, kSlm6OffRopeQSaturation, st.rope_q_saturation_count);
	Slm6Put<uint64_t>(dst, kSlm6OffRopeKSaturation, st.rope_k_saturation_count);
	Slm6Put<int32_t>(dst, kSlm6OffSchemaIndex, seq->bound_schema_index);
	Slm6Put<uint32_t>(dst, kSlm6OffWalkState, seq->dfa_walk_state);
	Slm6Put<uint32_t>(dst, kSlm6OffReady, seq->ready_for_logits ? 1u : 0u);
	Slm6Put<uint32_t>(dst, kSlm6OffKvMode, kSlm6KvModeBudget);
	Slm6Put<int32_t>(dst, kSlm6OffBudget, seq->budget);
	Slm6Put<int64_t>(dst, kSlm6OffOrigin, seq->origin);
	size_t off = kSlm6FixedBytes;
	if (residual > 0) std::memcpy(dst + off, seq->hidden_codes.data(), residual);
	off += residual;
	Slm6Put<uint64_t>(dst, off, static_cast<uint64_t>(lprime));
	off += kSlm6KvPositionsBytes;
	// The canonical gather (§3.7): rows [0, L) page by page, then the mid-token row at L.
	const int64_t B = pool->geo.page_positions;
	const size_t per_layer = 2u * static_cast<size_t>(model->num_key_value_heads);
	for (size_t r = 0; r < runs; ++r) {
		uint8_t* run_dst = dst + off + r * static_cast<size_t>(lprime) * D;
		for (int64_t p0 = 0; p0 < L; p0 += B) {
			const int64_t n = std::min<int64_t>(B, L - p0);
			std::memcpy(run_dst + static_cast<size_t>(p0) * D, PooledMirrorRow(pool, seq->holder, r, p0, D),
			            static_cast<size_t>(n) * D);
		}
		if (lprime > L) {
			uint8_t* row = run_dst + static_cast<size_t>(L) * D;
			if (r / per_layer < layer_index) {
				std::memcpy(row, PooledMirrorRow(pool, seq->holder, r, L, D), D);
			} else {
				std::memset(row, 0, D);  // §3.7's zero substitution
			}
		}
	}
	*out_blob_size = required;
	return true;
}

// An 'SLM6' blob read and validated (§3.7's restore validation and 2.6's cases), nothing drawn.
struct Slm6Restore {
	uint32_t layer_index = 0;
	superslm::CarriedScale hidden_scale{};
	uint64_t kv_saturation_count = 0;
	uint64_t kv_landing_saturation_count = 0;
	uint64_t k_channel_landing_saturation_count = 0;
	uint64_t rope_q_saturation_count = 0;
	uint64_t rope_k_saturation_count = 0;
	int64_t context_length = 0;
	int32_t bound_schema_index = -1;
	uint32_t dfa_walk_state = 0xFFFFFFFFu;
	bool ready_for_logits = false;
	int32_t budget = 0;
	int64_t origin = 0;
	int64_t lprime = 0;
	const uint8_t* residual = nullptr;
	const uint8_t* rows = nullptr;  // [run][pos < L'][d]
};

// §4.10 step 4, the body of an 'SLM6' (the magic is the caller's step 3): every check of 2.6 and 2.3 with
// SSLM_SEQUENCE_KV_BUFFER_MISMATCH, the status today's GPU restore returns for a malformed blob -- a
// kv_mode other than budget (the writer rule makes every SLM6 a budget blob, so a whole_reserve or unknown
// kv_mode is malformed here), a budget outside [1, cap], an origin outside [0, context_length], a
// context_length above min(origin + budget, cap), a layer_index past the model's layers, an L' above the
// cap, kv_positions or workspace_size inconsistent with L', a residual of another size, and any size other
// than the exact one; then SSLM_RESTORE_MODEL_MISMATCH for another model's blob (as sslm_gpu_seq_restore);
// then the schema binding's two checks (RestoredSchemaBindingValid).
SslmGpuStatus ParseSlm6Body(const SslmGpuModelHandle* model, const void* blob, size_t blob_size, Slm6Restore* out) {
	const uint8_t* p = static_cast<const uint8_t*>(blob);
	if (blob_size < kSlm6FixedBytes) return SSLM_SEQUENCE_KV_BUFFER_MISMATCH;
	const int64_t cap = model->context_cap;
	const uint32_t kv_mode = Slm6Get<uint32_t>(p, kSlm6OffKvMode);
	const int32_t budget = Slm6Get<int32_t>(p, kSlm6OffBudget);
	const int64_t origin = Slm6Get<int64_t>(p, kSlm6OffOrigin);
	const int64_t context_length = Slm6Get<int64_t>(p, kSlm6OffContextLength);
	const uint32_t layer_index = Slm6Get<uint32_t>(p, kSlm6OffLayerIndex);
	const uint64_t residual = Slm6Get<uint64_t>(p, kSlm6OffHiddenCodesSize);
	const uint64_t workspace = Slm6Get<uint64_t>(p, kSlm6OffWorkspaceSize);
	if (kv_mode != kSlm6KvModeBudget) return SSLM_SEQUENCE_KV_BUFFER_MISMATCH;
	if (!superslm_gpu::kv_pool::BudgetInDomain(budget, cap)) return SSLM_SEQUENCE_KV_BUFFER_MISMATCH;
	if (context_length < 0 || context_length > cap) return SSLM_SEQUENCE_KV_BUFFER_MISMATCH;
	if (origin < 0 || origin > context_length) return SSLM_SEQUENCE_KV_BUFFER_MISMATCH;
	if (context_length > superslm_gpu::kv_pool::HolderLimit(origin, budget, cap)) return SSLM_SEQUENCE_KV_BUFFER_MISMATCH;
	if (layer_index > model->num_hidden_layers) return SSLM_SEQUENCE_KV_BUFFER_MISMATCH;
	const int64_t lprime = Slm6Lprime(context_length, layer_index, model->num_hidden_layers);
	if (lprime > cap) return SSLM_SEQUENCE_KV_BUFFER_MISMATCH;
	if (residual != static_cast<uint64_t>(model->hidden_size)) return SSLM_SEQUENCE_KV_BUFFER_MISMATCH;
	// Every term below is bounded: lprime <= cap (a u32), runs and head_dim are the model's own.
	const uint64_t rows_bytes = static_cast<uint64_t>(lprime) * GpuKvRuns(model) * model->head_dim;
	if (workspace != rows_bytes) return SSLM_SEQUENCE_KV_BUFFER_MISMATCH;
	const uint64_t kvpos_off = kSlm6FixedBytes + residual;
	if (static_cast<uint64_t>(blob_size) < kvpos_off + kSlm6KvPositionsBytes) return SSLM_SEQUENCE_KV_BUFFER_MISMATCH;
	if (Slm6Get<uint64_t>(p, static_cast<size_t>(kvpos_off)) != static_cast<uint64_t>(lprime)) {
		return SSLM_SEQUENCE_KV_BUFFER_MISMATCH;
	}
	if (static_cast<uint64_t>(blob_size) != kvpos_off + kSlm6KvPositionsBytes + rows_bytes) {
		return SSLM_SEQUENCE_KV_BUFFER_MISMATCH;  // the exact size: truncated or trailing bytes
	}
	if (std::memcmp(p + kSlm6OffModelHash, model->content_hash.data(), superslm::kIntegrityHashBytes) != 0) {
		return SSLM_RESTORE_MODEL_MISMATCH;
	}
	const int32_t schema_index = Slm6Get<int32_t>(p, kSlm6OffSchemaIndex);
	const uint32_t walk_state = Slm6Get<uint32_t>(p, kSlm6OffWalkState);
	if (!RestoredSchemaBindingValid(model, schema_index, walk_state)) return SSLM_SEQUENCE_KV_BUFFER_MISMATCH;
	out->layer_index = layer_index;
	out->hidden_scale.m = Slm6Get<int64_t>(p, kSlm6OffScaleM);
	out->hidden_scale.e = Slm6Get<int64_t>(p, kSlm6OffScaleE);
	out->kv_saturation_count = Slm6Get<uint64_t>(p, kSlm6OffKvSaturation);
	out->kv_landing_saturation_count = Slm6Get<uint64_t>(p, kSlm6OffKvLandingSaturation);
	out->k_channel_landing_saturation_count = Slm6Get<uint64_t>(p, kSlm6OffKChannelLandingSaturation);
	out->rope_q_saturation_count = Slm6Get<uint64_t>(p, kSlm6OffRopeQSaturation);
	out->rope_k_saturation_count = Slm6Get<uint64_t>(p, kSlm6OffRopeKSaturation);
	out->context_length = context_length;
	out->bound_schema_index = schema_index;
	out->dfa_walk_state = walk_state;
	out->ready_for_logits = Slm6Get<uint32_t>(p, kSlm6OffReady) != 0;
	out->budget = budget;
	out->origin = origin;
	out->lprime = lprime;
	out->residual = p + kSlm6FixedBytes;
	out->rows = p + kvpos_off + kSlm6KvPositionsBytes;
	return SSLM_OK;
}
}  // namespace

// Design Sec4.2/Sec5.3 (T-2113 B5): the device-resident save/restore/reset trio,
// declared for B3/B5 (gpu_1p0.h's own header comment) -- reusing
// `SaveGpuSequenceState`/`RestoreGpuSequenceState` (gpu_port.h, the substrate's own
// existing blob mechanism, D-SLM3080) rather than a second implementation. T-2114 (C1)
// extended both functions to also round-trip `hidden_codes` -- see gpu_port.h's own
// header comment on the pair for the current contract.
SslmGpuStatus sslm_gpu_seq_saveImpl(SslmGpuContext* ctx, const SslmGpuSequenceHandle* seq,
                                 void* out_blob, size_t* out_blob_size) {
	if (!ctx || !seq || seq->ctx != ctx) {  // T-2114 (S2): ->destroyed read removed, see sequence
	                                         // handle's own field comment. T-2124 (D-SLM3446
	                                         // P1-4): `seq->ctx != ctx`.
		return SSLM_SEQUENCE_KV_BUFFER_MISMATCH;
	}
	const bool is_submitted = seq->state() == superslm_gpu::SslmSequenceGpuState::Submitted;
	if (!superslm_gpu::CallProceedsOrBusy_SeqSave(is_submitted)) {
		return SSLM_BUSY;  // design Sec4.2's own state table, unchanged policy.
	}
	// Paged KV (G5, §4 item 10's writer rule): a pooled sequence saves 'SLM6', gathered from the pool's
	// host mirror through its table; a dedicated sequence keeps saving 'SLM5', below, unchanged. A
	// too-small buffer reports the required size and returns the status the 'SLM5' save returns.
	if (seq->pool != nullptr) return SavePooledSlm6(seq, out_blob, out_blob_size) ? SSLM_OK : SSLM_DEVICE_LOST;
	// T-2114 (C1): hidden_codes_size = seq->hidden_codes.size() -- the 1.0 handle owns this
	// vector (design Sec5.3), so the residual stream now round-trips through the blob
	// alongside the K/V workspace bytes, closing the class of loss a restored, non-zero
	// hidden_scale paired with a zeroed residual used to produce silently.
	// T-2113 (P2, design Sec4.2/Sec22, D-SLM3415): `seq->model->content_hash` -- the model this
	// sequence was created against, already retained (design Sec5.1) -- is written into the v3
	// blob's own trailing `model_content_hash` field so a later restore can detect a mismatch
	// against a DIFFERENT target model.
	const bool ok = superslm_gpu::SaveGpuSequenceState(
	    seq->live_state, seq->hidden_codes.size(), seq->host_kv_mirror.data(),
	    seq->host_kv_mirror.size(), seq->model->content_hash,
	    // T-2895: the G5-5 lifecycle triple, appended to the v5 blob (closing TE-362).
	    seq->bound_schema_index, seq->dfa_walk_state, seq->ready_for_logits, out_blob,
	    out_blob_size);
	// SaveGpuSequenceState's own contract (gpu_port.h): a too-small out_blob reports the
	// required size via *out_blob_size and returns false -- not a device/guard failure,
	// no dedicated 1.0 status exists for it either, same disposition as the guards above.
	return ok ? SSLM_OK : SSLM_DEVICE_LOST;
}

SslmGpuStatus sslm_gpu_seq_restoreImpl(SslmGpuContext* ctx, SslmGpuModelHandle* model,
                                    const void* blob, size_t blob_size,
                                    SslmGpuSequenceHandle** out_seq) {
	if (!out_seq) {
		return SSLM_SEQUENCE_KV_BUFFER_MISMATCH;
	}
	*out_seq = nullptr;
	if (!ctx || !model || !blob || model->ctx != ctx) {  // T-2114 (S2): ->destroyed read removed,
	                                                       // see model handle's own field comment.
	                                                       // T-2124 (D-SLM3446 P1-4):
	                                                       // `model->ctx != ctx`.
		return SSLM_SEQUENCE_KV_BUFFER_MISMATCH;
	}
	// T-2243 (C1, design plan Sec8/Sec10 Phase 2 C1, D-SLM3991), scope corrected by review
	// finding 2 (D-SLM4113): a sibling sequence with an unfenced, in-flight submission has
	// already recorded a command list against `harness::GetDevice()`'s process-global, single
	// command allocator/list and has not yet fenced -- restoring a fresh sequence proceeds to
	// allocate and upload a NEW K/V buffer against the identical device without waiting for
	// that fence, a real ordering hazard on the shared device/command-queue state. The hazard is
	// process-global, not per-model: `model->submitted_sequences() > 0` alone missed a sibling in
	// flight on a DIFFERENT model or a different context, which reaches the identical
	// `dev.alloc->Reset()`/`dev.list->Reset(...)` (superslm_gpu.cpp `:3620-3628`) just the same.
	// The process count (GpuSubmittedCounts::Process()) is the scoped-correctly count -- see its own
	// comment.
	// Genuinely transient (drains the instant the in-flight sequence's own fence signals):
	// `SSLM_BUSY` is the correct disposition here, the same Busy-precedence shape
	// `sslm_gpu_model_unmap`'s own `submitted_sequences` check already uses, and is untouched by
	// D-SLM3965 (which concerns only the new PERSISTENT-liveness guards below, M2's
	// `live_adapters` and S2's `bound_sequences`). Remedy: drain the in-flight sequence
	// (`sslm_gpu_ready` to Idle) and retry -- costs the caller nothing.
	//
	// From G3 this check is restore's EntrySubmitGuard (§4.4): the restore reaches both devices
	// (its upload on the context's own, its round trip on the singleton), so the ledger term
	// applies, and it keeps its existing any-sequence scope (G37). Placed where today's check is.
	if (const SslmGpuStatus g = EntrySubmitGuard(GuardScope::kAny, GuardDevice::kReachesSingleton);
	    g != SSLM_OK) {
		return g;
	}
	// Design Sec4.2/Sec21 (T-2114 N1, corrected 2026-08-15, routing
	// `Claude/Poirot/50f3d5d-t2113-1p0-gpu-core-build-review.md` Sec6/Sec8): "sslm_gpu_seq_restore
	// allocates a fresh SslmGpuSequenceHandle... never reusing an existing handle's" -- sized to the
	// BLOB's own recorded context_cap, not the model's. Reading the header first (never the body,
	// never a device call) lets this derive that cap before anything is allocated: `workspace_size`
	// solves `workspace_size = num_hidden_layers * context_cap * num_key_value_heads * head_dim * 2`
	// (Sec5.1/D-SLM3310's own K/V sizing formula) for context_cap, with every other term already
	// known through `model`. A blob too small/wrong-magic to hold a header, a division with a
	// nonzero remainder, or a derived context_cap <= 0 is malformed; a derived context_cap that
	// EXCEEDS model->context_cap is inadmissible (this model cannot represent a sequence that large)
	// -- both rejected under the same disposition every other structural refusal in this function
	// uses. A derived context_cap strictly less than the model's own maximum is the ordinary case
	// this correction exists to admit, not an error: the prior always-model-cap sizing made restoring
	// a short, cheaply-capped sequence -- the common case, since context_cap is a sslm_gpu_seq_create
	// parameter for exactly the reason not every sequence needs the model's full capacity -- the one
	// case that always failed the size check below, loudly but wrongly scoped.
	// Paged KV (G5, §4.10 step 3, §3.9 A11): an 'SLM6' is a pooled sequence's budget blob, and a
	// dedicated holder's per-handle cap has no budget, origin or pool to restore it into. The magic alone
	// decides it, so a hostile 'SLM6' is a mode mismatch too; nothing is allocated.
	if (GpuSeqBlobMagic(blob, blob_size) == kGpuSeqBlobMagicSlm6) return SSLM_GPU_HOLDER_MODE_MISMATCH;
	uint64_t blob_workspace_size = 0;
	if (!superslm_gpu::PeekGpuSeqBlobWorkspaceSize(blob, blob_size, &blob_workspace_size)) {
		return SSLM_SEQUENCE_KV_BUFFER_MISMATCH;
	}
	// Identical per-context_cap-unit byte count to sslm_gpu_seq_create's own kv_bytes_needed
	// computation (this file, above), factored out of context_cap so dividing workspace_size by it
	// recovers context_cap -- the same overflow-guarded product, computed the same way.
	size_t per_cap_unit_bytes = static_cast<size_t>(model->num_hidden_layers);
	const size_t per_cap_unit_factors[] = {model->num_key_value_heads,
	                                        static_cast<size_t>(model->head_dim), 2u};
	bool per_cap_unit_overflowed = false;
	for (size_t factor : per_cap_unit_factors) {
		if (factor != 0 && per_cap_unit_bytes > SIZE_MAX / factor) {
			per_cap_unit_overflowed = true;
			break;
		}
		per_cap_unit_bytes *= factor;
	}
	if (per_cap_unit_overflowed || per_cap_unit_bytes == 0 ||
	    blob_workspace_size % static_cast<uint64_t>(per_cap_unit_bytes) != 0) {
		return SSLM_SEQUENCE_KV_BUFFER_MISMATCH;  // malformed: not an integral context_cap
	}
	const uint64_t derived_context_cap = blob_workspace_size / static_cast<uint64_t>(per_cap_unit_bytes);
	if (derived_context_cap == 0 ||
	    derived_context_cap > static_cast<uint64_t>(model->context_cap)) {
		// derived_context_cap == 0: malformed (a real save never produces a zero-cap sequence).
		// derived_context_cap > model->context_cap: inadmissible -- this model cannot create a
		// handle large enough to hold what the blob claims to be. Both share the one disposition
		// every structural rejection in this function already uses (design Sec4.2/Sec21: "no new
		// status is minted for this").
		return SSLM_SEQUENCE_KV_BUFFER_MISMATCH;
	}

	// T-2113 (P2, design Sec4.2/Sec9/Sec22, routed `Claude/Poirot/50f3d5d-t2113-1p0-gpu-core-
	// build-review.md` Sec15, D-SLM3415): the identity check the N1 size-admissibility widening
	// (above) left open -- a foreign blob whose derived context_cap happens to be admissible
	// against a same-shape-but-different model would otherwise restore silently. Checked AFTER
	// the size-derivation ladder (a structurally malformed blob is rejected for that reason
	// first, before its content hash is even read) and BEFORE any device work -- the fresh
	// handle is not created until this passes. A blob whose header cannot even be re-read here
	// (should be unreachable: PeekGpuSeqBlobWorkspaceSize already validated the identical
	// magic/size preconditions above) falls under the same structural-malformed disposition
	// every other rejection in this function uses; a hash that IS read but does not match
	// `model`'s own returns the distinctly-named SSLM_RESTORE_MODEL_MISMATCH (design Sec9) --
	// never folded into the generic malformed-blob status, since a wrong size and a wrong model
	// are different facts a caller needs told apart.
	std::array<uint8_t, superslm::kIntegrityHashBytes> blob_model_hash{};
	if (!superslm_gpu::PeekGpuSeqBlobModelHash(blob, blob_size, &blob_model_hash)) {
		return SSLM_SEQUENCE_KV_BUFFER_MISMATCH;
	}
	if (blob_model_hash != model->content_hash) {
		return SSLM_RESTORE_MODEL_MISMATCH;
	}

	SslmGpuSequenceHandle* fresh = nullptr;
	const SslmGpuStatus create_status =
	    sslm_gpu_seq_create(ctx, model, static_cast<int64_t>(derived_context_cap), &fresh);
	if (create_status != SSLM_OK || !fresh) {
		return create_status;
	}
	// RestoreGpuSequenceState's own device round-trip (gpu_port.h/superslm_gpu.cpp,
	// D-SLM3080, unchanged): decodes the blob's header fields into `live_state` and the
	// K/V bytes into `host_kv_mirror` -- a malformed or size-mismatched blob returns
	// false, translated here the same way every other structural rejection in this file
	// is (SSLM_SEQUENCE_KV_BUFFER_MISMATCH, the create-time disposition this fresh
	// handle would have carried had its own sizing been the problem).
	// T-2114 (C1): hidden_codes_size = fresh->hidden_codes.size() -- fresh->live_state.hidden_codes
	// already aliases fresh->hidden_codes.data() (set in sslm_gpu_seq_create, above), so
	// RestoreGpuSequenceState writes the blob's residual-stream bytes directly into this
	// handle's own storage; no separate sync step is needed the way the scalar fields below
	// need one, because this is the same memory, not a mirror.
	//
	// T-2114 (S4, Claude/Poirot/50f3d5d-t2113-1p0-gpu-core-build-review.md): this call is now
	// inside a try -- it submits, fence-waits, and maps a readback entirely through
	// `SSLM_GPU_HR`, which throws on a real device/HR failure (the identical boundary B5 already
	// closed for `sslm_gpu_ready`/`RunLayerLoopGpuFinish`, cited in this same function's own
	// header comment above). Before this fix, an exception here escaped `sslm_gpu_seq_restore`
	// uncaught, crossing the status-returning API boundary design Sec4.1.1 establishes, and
	// stranded `fresh` with `ctx->live_handles`/`model->live_sequences` already incremented (by
	// the `sslm_gpu_seq_create` call above) with no release ever reached -- the context could
	// never afterwards be destroyed (`ContextHasLiveHandles`/`ModelHasLiveSequences`). The catch
	// below releases `fresh` exactly like the `ok == false` and `UploadResidentUavBufferSync`
	// failure paths immediately around it already do, and returns a status instead of unwinding.
	bool ok = false;
	int32_t restored_schema_index = -1;
	uint32_t restored_walk_state = 0xFFFFFFFFu;
	bool restored_ready_for_logits = false;
	superslm::SslmForwardStatus setup_status = superslm::SslmForwardStatus::Ok;
	try {
		ok = superslm_gpu::RestoreGpuSequenceState(
		    blob, blob_size, &fresh->live_state, fresh->hidden_codes.size(),
		    fresh->host_kv_mirror.data(), fresh->host_kv_mirror.size(),
		    // T-2895: the G5-5 lifecycle triple, read back from the v5 blob (closing TE-362).
		    &restored_schema_index, &restored_walk_state, &restored_ready_for_logits,
		    // 1.8.0 (TE-426): a submission device that could not be set up, told apart from a
		    // malformed blob.
		    &setup_status);
	} catch (const std::bad_alloc&) {
		sslm_gpu_seq_release(ctx, fresh);
		throw;
	} catch (const std::exception&) {
		// 1.8.0 (TE-426): the fault classifier over the round trip on the process's submission
		// device, which is set up (the round trip runs only once it is), so this GetDevice() call
		// makes no setup attempt.
		const SslmGpuStatus fault = ClassifyHandleFault(&superslm_gpu::harness::GetDevice());
		sslm_gpu_seq_release(ctx, fresh);
		return fault;
	} catch (...) {
		// 1.8.0 (TE-435): any other exception type, as the clause above -- without this clause a
		// non-standard type left `fresh` unreleased, its context and model counts held for the life
		// of the process.
		const SslmGpuStatus fault = ClassifyHandleFault(&superslm_gpu::harness::GetDevice());
		sslm_gpu_seq_release(ctx, fresh);
		return fault;
	}
	if (!ok) {
		sslm_gpu_seq_release(ctx, fresh);
		// The process's submission device could not be set up: one status per cause, the same
		// from every entry point (gpu_1p0.h, SSLM_DEVICE_LOST and SSLM_GPU_ALLOCATION_FAILED).
		if (setup_status == superslm::SslmForwardStatus::GpuAllocationFailed) {
			return SSLM_GPU_ALLOCATION_FAILED;
		}
		if (setup_status != superslm::SslmForwardStatus::Ok) return SSLM_DEVICE_LOST;
		return SSLM_SEQUENCE_KV_BUFFER_MISMATCH;
	}
	// Upload the restored K/V bytes into this handle's own REAL resident buffer (design
	// Sec5.3: K/V is device-resident-only in the 1.0 API -- RestoreGpuSequenceState's own
	// host-memory round-trip is the substrate's pre-1.0 contract, reused verbatim for the
	// header/blob decode; the device residency itself is this call's own added step).
	try {
		fresh->kv_buf = UploadResidentUavBufferSync(ctx->device, fresh->host_kv_mirror.data(),
		                                             fresh->host_kv_mirror.size());
		superslm_gpu::harness::NameBuffer(fresh->kv_buf.Get(), L"SuperSLM dedicated K/V (restored)");
	} catch (const std::bad_alloc&) {
		sslm_gpu_seq_release(ctx, fresh);
		throw;
	} catch (const std::exception&) {
		const SslmGpuStatus fault = ClassifyHandleFault(&ctx->device);  // 1.8.0 (TE-426)
		sslm_gpu_seq_release(ctx, fresh);
		return fault;
	} catch (...) {
		const SslmGpuStatus fault = ClassifyHandleFault(&ctx->device);  // 1.8.0 (TE-435), as above
		sslm_gpu_seq_release(ctx, fresh);
		return fault;
	}
	fresh->kv_needs_resume_barrier = false;  // freshly (re)created in UNORDERED_ACCESS state
	fresh->hidden_scale = fresh->live_state.hidden_scale;
	fresh->layer_index = fresh->live_state.layer_index;
	fresh->kv_saturation_count = fresh->live_state.kv_saturation_count;
	fresh->context_length = fresh->live_state.context_length;
	// T-2895 (closing TE-362): untrusted-input validation on the restored schema binding,
	// mirroring sslm_seq_restore's own C1 fix (src/sslm_abi.cpp) -- bound the restored
	// walk-state against the RESOLVED schema's own state_count before it is ever used to
	// index mask_pages, and reject a bound index this model does not have. Unbound
	// (restored_schema_index < 0) always passes: no state to validate.
	// Paged KV (G5): the two checks live in RestoredSchemaBindingValid (above), shared with
	// sslm_gpu_seq_restore_into_pool, which runs them before it draws anything.
	if (!RestoredSchemaBindingValid(model, restored_schema_index, restored_walk_state)) {
		sslm_gpu_seq_release(ctx, fresh);
		return SSLM_SEQUENCE_KV_BUFFER_MISMATCH;
	}
	fresh->bound_schema_index = restored_schema_index;
	fresh->dfa_walk_state = restored_walk_state;
	fresh->ready_for_logits = restored_ready_for_logits;
	fresh->bind_eligible = false;
	*out_seq = fresh;
	return SSLM_OK;
}

SslmGpuStatus sslm_gpu_seq_resetImpl(SslmGpuContext* ctx, SslmGpuSequenceHandle* seq) {
	if (!ctx || !seq || seq->ctx != ctx) {  // T-2114 (S2): ->destroyed read removed, see sequence
	                                         // handle's own field comment. T-2124 (D-SLM3446
	                                         // P1-4): `seq->ctx != ctx`.
		return SSLM_SEQUENCE_KV_BUFFER_MISMATCH;
	}
	const bool is_submitted = seq->state() == superslm_gpu::SslmSequenceGpuState::Submitted;
	if (!superslm_gpu::CallProceedsOrBusy_SeqReset(is_submitted)) {
		return SSLM_BUSY;  // design Sec4.2's own state table, unchanged policy.
	}
	SslmGpuStatus reset_status = SSLM_OK;
	std::fill(seq->hidden_codes.begin(), seq->hidden_codes.end(), 0);
	seq->hidden_scale = superslm::CarriedScale{};
	seq->layer_index = 0;
	seq->kv_saturation_count = 0;
	seq->context_length = 0;
	seq->live_state.hidden_scale = seq->hidden_scale;
	seq->live_state.layer_index = 0;
	seq->live_state.kv_saturation_count = 0;
	seq->live_state.kv_landing_saturation_count = 0;
	seq->live_state.k_channel_landing_saturation_count = 0;
	seq->live_state.rope_q_saturation_count = 0;
	seq->live_state.rope_k_saturation_count = 0;
	seq->live_state.context_length = 0;
	if (seq->pool != nullptr) {
		// Paged-KV plan §4 item 6b, step G3: the mirror is the POOL's, so the dedicated fill below
		// would erase every neighbour's rows. A pooled reset touches only its own table and reserve
		// (its pages back to its reserve, kv_pages) and fills nothing; no read goes past a holder's
		// live width, so stale rows are never observed. The table changed, so it is re-uploaded.
		// HolderUnmapAll completes the unmap whatever it returns, so a non-OK status (a refcount
		// underflow) finishes the reset and is reported at the end (review-g23 nit).
		if (seq->holder != nullptr) {
			const superslm::kv_pages::PageStatus ps = superslm::kv_pages::HolderUnmapAll(seq->holder);
			if (ps != superslm::kv_pages::PageStatus::kOk) reset_status = SSLM_SEQUENCE_KV_BUFFER_MISMATCH;
		}
		seq->table_dirty = true;
		// Paged-KV plan §3.4's reset row (G4): origin := 0, so the limit is the declared budget's
		// again after an adopt moved it to the prefix's length.
		seq->origin = 0;
		seq->limit = superslm_gpu::kv_pool::HolderLimit(0, seq->budget, seq->context_cap);
	} else {
		std::fill(seq->host_kv_mirror.begin(), seq->host_kv_mirror.end(), 0);
	}
	// G5-5 (T-2132, Brunel): mirrors `sslm_seq_reset`'s own extension (design Sec5,
	// src/sslm_abi.cpp) -- reset clears the DFA-walk-state back to the bound schema's own start
	// state (0) but PRESERVES the schema binding itself (`bound_schema_index` untouched); a
	// sequence with no schema bound stays at the unused sentinel, exactly the CPU ABI's own
	// disposition.
	if (seq->bound_schema_index >= 0) {
		seq->dfa_walk_state = 0u;
	}
	// G5-5 session 3 fix (T-2132, Brunel): mirrors `sslm_seq_reset`'s own identical
	// `ready_for_logits = false` clear (src/sslm_abi.cpp:1488) -- a reset sequence must not carry
	// a stale "finish without embedding" flag into whatever decode call comes next.
	seq->ready_for_logits = false;
	seq->prefill_hidden_valid = false;  // prefill snapshot writer: reset
	seq->bind_eligible = true;
	return reset_status;
}

// T-2113 (B3): the bench-only accessor bodies declared in gpu_1p0_bench_bridge.h -- B5
// has now landed the real async path above; these accessors are kept only because
// tools/t2113_b3_sequence_smoke.cpp (a B3-era bench tool, not part of Curie's suite)
// still calls them, and retiring that tool is not part of this section's own scope.
// Every NEW caller (the B5 smoke tool, the red suite) drives the real API instead.
ID3D12Resource* SslmGpuSeqHandleKvBufferForBench(SslmGpuSequenceHandle* seq) {
	superslm_gpu::inflight::GpuApiCallScope scope(__func__);  // G3 (§4.4): every C ABI entry point
	if (!seq) return nullptr;
	return seq->pool != nullptr ? seq->pool->pool_buf.Get() : seq->kv_buf.Get();  // G3: the pool's
}
// G3 (§9's G3 row): the latch the handle passes -- the pool's for a pooled handle.
bool* SslmGpuSeqHandleKvResumeFlagForBench(SslmGpuSequenceHandle* seq) {
	superslm_gpu::inflight::GpuApiCallScope scope(__func__);  // G3 (§4.4): every C ABI entry point
	if (!seq) return nullptr;
	return seq->pool != nullptr ? &seq->pool->kv_needs_resume_barrier : &seq->kv_needs_resume_barrier;
}
int8_t* SslmGpuSeqHandleHiddenCodesForBench(SslmGpuSequenceHandle* seq) {
	superslm_gpu::inflight::GpuApiCallScope scope(__func__);  // G3 (§4.4): every C ABI entry point
	return seq ? seq->hidden_codes.data() : nullptr;
}
superslm::CarriedScale* SslmGpuSeqHandleHiddenScaleForBench(SslmGpuSequenceHandle* seq) {
	superslm_gpu::inflight::GpuApiCallScope scope(__func__);  // G3 (§4.4): every C ABI entry point
	return seq ? &seq->hidden_scale : nullptr;
}
uint32_t* SslmGpuSeqHandleLayerIndexForBench(SslmGpuSequenceHandle* seq) {
	superslm_gpu::inflight::GpuApiCallScope scope(__func__);  // G3 (§4.4): every C ABI entry point
	return seq ? &seq->layer_index : nullptr;
}
uint64_t* SslmGpuSeqHandleKvSaturationForBench(SslmGpuSequenceHandle* seq) {
	superslm_gpu::inflight::GpuApiCallScope scope(__func__);  // G3 (§4.4): every C ABI entry point
	return seq ? &seq->kv_saturation_count : nullptr;
}
superslm::SequenceLayerState* SslmGpuSeqHandleLiveStateForBench(SslmGpuSequenceHandle* seq) {
	superslm_gpu::inflight::GpuApiCallScope scope(__func__);  // G3 (§4.4): every C ABI entry point
	return seq ? &seq->live_state : nullptr;
}
int64_t* SslmGpuSeqHandleContextLengthForBench(SslmGpuSequenceHandle* seq) {
	superslm_gpu::inflight::GpuApiCallScope scope(__func__);  // G3 (§4.4): every C ABI entry point
	return seq ? &seq->context_length : nullptr;
}
size_t SslmGpuSeqHandleHiddenSizeForBench(SslmGpuSequenceHandle* seq) {
	superslm_gpu::inflight::GpuApiCallScope scope(__func__);  // G3 (§4.4): every C ABI entry point
	return seq ? seq->hidden_codes.size() : 0;
}
// T-2113 (N1, design Sec4.2/Sec21, Sec11 dim9's own product cell): exposes the handle's own
// context_cap so a bench driver can confirm sslm_gpu_seq_restore sized the FRESH handle to the
// blob's own recorded context_cap, not the model's -- the exact property N1's fix makes true and
// the property a red cell needs a real accessor, not a hardcoded assumption, to check.
int64_t SslmGpuSeqHandleContextCapForBench(SslmGpuSequenceHandle* seq) {
	superslm_gpu::inflight::GpuApiCallScope scope(__func__);  // G3 (§4.4): every C ABI entry point
	return seq ? seq->context_cap : 0;
}

// T-2243 (S2/O2, red suite scaffolding, plan Sec10 Phase 2 S2/O2, Sec5.2's own fixture_common.h
// additions): global-scope, `extern`-linkage per the `:142-151` note above -- declared in the
// suite's own `fixture_common.h`, defined here beside the other ForBench accessors.
//
// `std::atomic<int64_t>` is lock-free and layout-compatible with `int64_t` on every platform
// this project builds for (MSVC/x64) -- the identical assumption this accessor's own signature
// (a raw `int64_t*`, matching every other ForBench accessor's shape rather than an atomic-typed
// one) already makes explicit.
int64_t* SslmGpuAdapterHandleBoundSequencesForBench(SslmGpuAdapterHandle* adapter) {
	return adapter ? reinterpret_cast<int64_t*>(&adapter->bound_sequences) : nullptr;
}
const SslmGpuAdapterHandle* const* SslmGpuSequenceHandleBoundAdapterForBench(SslmGpuSequenceHandle* seq) {
	return seq ? &seq->bound_adapter : nullptr;
}
// O2 only (Mendeleev F-5, plan Sec10 Phase 2 O2): forces this handle's internal state directly
// to reconstruct the window `sslm_gpu_ready` cannot otherwise be driven into through the public
// API alone -- Submitted, with the in-flight token already cleared -- so the PUBLIC
// `sslm_gpu_ready` call the cell drives afterward exercises `RunLayerLoopGpuFinish(nullptr, ...)`
// for real, through the real function the O2 fix lands in, rather than bypassing it with a
// direct call into the internal finish path (which would prove nothing, per F-5's own
// reasoning).
void SslmGpuSeqForceSubmittedNoInflightForBench(SslmGpuSequenceHandle* seq) {
	if (!seq) return;
	// G3 (§4.4 rev 11): through the counting accessor, so its counts are raised (and left raised:
	// the forced state is never left, by design of the cell that uses it).
	SetGpuSeqState(seq, superslm_gpu::SslmSequenceGpuState::Submitted, nullptr);
	seq->in_flight = superslm_gpu::GpuInFlightToken();  // empty (a token held drains first)
}

// -----------------------------------------------------------------------------------------
// G5-5 (T-2132, Brunel): gpu_1p0_g5_bridge.h's own body -- see that header for the full
// contract each function below implements. "No new arithmetic" (design Sec4) is enforced by
// construction here: every logits/mask/argmax step below is a direct call into
// superslm::RmsNormSite/LogitsSite/ApplyMaskAndArgmax/ArgmaxLowestIndexTieBreak
// (forward_sites.h) -- the SAME functions sslm_decode_step's own finishing block
// (src/sslm_abi.cpp) calls on its own hidden state, never a parallel reimplementation.
// -----------------------------------------------------------------------------------------

bool SslmGpuModelHasSchemasForG5Bridge(SslmGpuModelHandle* model) {
	superslm_gpu::inflight::GpuApiCallScope scope(__func__);  // G3 (§4.4): every C ABI entry point
	return model != nullptr && model->schemas.Count() > 0;
}

int32_t SslmGpuSchemaLookupForG5Bridge(SslmGpuModelHandle* model, const char* name) {
	superslm_gpu::inflight::GpuApiCallScope scope(__func__);  // G3 (§4.4): every C ABI entry point
	if (!model || !name) return -1;
	size_t index = 0;
	if (!model->schemas.ByName(name, &index)) return -1;
	return static_cast<int32_t>(index);
}

SslmGpuStatus SslmGpuSeqSetSchemaForG5BridgeImpl(SslmGpuContext* ctx, SslmGpuSequenceHandle* seq,
                                               int32_t schema_index) {
	if (!ctx || !seq || !seq->model || seq->ctx != ctx) {
		return SSLM_SEQUENCE_KV_BUFFER_MISMATCH;
	}
	if (seq->state() == superslm_gpu::SslmSequenceGpuState::Submitted) return SSLM_BUSY;
	if (!seq->bind_eligible) return SSLM_SEQUENCE_REJECTED; // D-SLM7625 bind gate
	if (schema_index < 0) {
		seq->bound_schema_index = -1;
		seq->dfa_walk_state = kSslmGpuDfaWalkStateUnused;
		return SSLM_OK;
	}
	if (static_cast<size_t>(schema_index) >= seq->model->schemas.Count()) {
		return SSLM_SEQUENCE_REJECTED;
	}
	seq->bound_schema_index = schema_index;
	seq->dfa_walk_state = 0u;
	return SSLM_OK;
}

namespace {
// Exact host-side twin of the CPU ABI's accepting-set lookup. SCM1 stores strictly ascending
// little-endian uint32 values, so this performs O(log accepting_count) host reads and no GPU work.
bool IsAcceptingState(const superslm::SchemaEntry& entry, uint32_t state) {
	using superslm::schema_masks_detail::ReadLE32;
	uint32_t lo = 0, hi = entry.accepting_count;
	while (lo < hi) {
		const uint32_t mid = lo + (hi - lo) / 2;
		const uint32_t value = ReadLE32(entry.accepting_le + static_cast<size_t>(mid) * 4);
		if (value == state) return true;
		if (value < state) lo = mid + 1; else hi = mid;
	}
	return false;
}
}  // namespace

SslmGpuStatus SslmGpuSeqSchemaAcceptingForG5BridgeImpl(
    SslmGpuContext* ctx, SslmGpuSequenceHandle* seq, int32_t* out_schema_accepting) {
	if (!ctx || !seq || !seq->model || seq->ctx != ctx || !out_schema_accepting) {
		return SSLM_SEQUENCE_KV_BUFFER_MISMATCH;
	}
	if (seq->state() == superslm_gpu::SslmSequenceGpuState::Submitted) return SSLM_BUSY;
	if (seq->bound_schema_index < 0) {
		*out_schema_accepting = 0; // unbound contract
		return SSLM_OK;
	}
	const superslm::SchemaEntry* entry =
	    seq->model->schemas.ByIndex(static_cast<size_t>(seq->bound_schema_index));
	if (!entry) return SSLM_SEQUENCE_KV_BUFFER_MISMATCH;
	const bool accepting = IsAcceptingState(*entry, seq->dfa_walk_state);
	*out_schema_accepting = accepting ? 1 : 0;
	return SSLM_OK;
}

SslmGpuStatus SslmGpuSeqSchemaBoundForG5BridgeImpl(
    SslmGpuContext* ctx, SslmGpuSequenceHandle* seq, int32_t* out_schema_bound) {
	if (!ctx || !seq || !seq->model || seq->ctx != ctx || !out_schema_bound) {
		return SSLM_SEQUENCE_KV_BUFFER_MISMATCH;
	}
	if (seq->state() == superslm_gpu::SslmSequenceGpuState::Submitted) return SSLM_BUSY;
	if (seq->bound_schema_index < 0) {
		*out_schema_bound = 0; // unbound contract
		return SSLM_OK;
	}
	const int32_t schema_index = seq->bound_schema_index;
	*out_schema_bound = schema_index >= 0 ? 1 : 0;
	return SSLM_OK;
}

uint32_t SslmGpuSeqWalkStateForG5Bridge(SslmGpuSequenceHandle* seq) {
	superslm_gpu::inflight::GpuApiCallScope scope(__func__);  // G3 (§4.4): every C ABI entry point
	return seq ? seq->dfa_walk_state : kSslmGpuDfaWalkStateUnused;
}

// T-2866 (correcting TE-338 Sec8's finding that this cell had no seam on the real GPU path).
// Mirrors `SUPERSLM_T2169_CHUNK_RECORDING_FAULT_INJECTION`'s own single-shot armed-flag idiom
// exactly (this file, below) but is its OWN macro, not a third named site under that one: this
// seam fires unconditionally, once, on the NEXT finish call, with no token-index or site
// selector to match -- a different parameter shape that does not fit the existing macros' own
// convention. Fires once inside `SslmGpuSeqFinishTokenForG5BridgeImpl` (below), after `LogitsSite`
// populates `logit_row` and before the schema branch's `ApplyMaskAndArgmax` call: overwrites
// every element of `logit_row` to `INT32_MIN`, presenting the degenerate row to the REAL finish
// bridge's own masking/argmax/transition code on a live forward's own logits buffer, not a
// host-only latch model -- the CPU twin (`ArmCpuFinishDegenerateLogitRowInjection`,
// src/sslm_abi.cpp) presents the identical construction on the host code both backends share for
// this step (`superslm::ApplyMaskAndArgmax`).
#if defined(SUPERSLM_GPU_G5_FINISH_ROW_FAULT_INJECTION)
namespace {
bool g_gpu_g5_finish_row_fault_armed = false;
}  // namespace

// `extern "C"` linkage, global scope -- matches the test author's own reservation
// (`tests/t2899-schema-deadend-red-suite/cell_gpu_cell2_degenerate.cpp`:
// `extern "C" void ArmGpuFinishDegenerateLogitRowInjection();`) and the CPU twin's own identical
// placement (`ArmCpuFinishDegenerateLogitRowInjection`, src/sslm_abi.cpp), so a different
// translation unit's own local `extern "C"` declaration resolves against this definition with no
// header of its own needed -- the same pattern `SslmSeqLiveStateForTest` already uses.
extern "C" void ArmGpuFinishDegenerateLogitRowInjection() { g_gpu_g5_finish_row_fault_armed = true; }

namespace {
inline void MaybeInjectGpuFinishDegenerateLogitRow(int32_t* logit_row, int32_t vocab_size) {
	if (g_gpu_g5_finish_row_fault_armed) {
		// Single-shot: cleared before presenting the degenerate row, matching every other seam in
		// this file's own idiom -- a re-armed-forever flag would fire on every later, unrelated
		// call in the same process.
		g_gpu_g5_finish_row_fault_armed = false;
		std::fill(logit_row, logit_row + vocab_size, INT32_MIN);
	}
}
}  // namespace
#else
namespace {
inline void MaybeInjectGpuFinishDegenerateLogitRow(int32_t*, int32_t) {}
}  // namespace
#endif  // SUPERSLM_GPU_G5_FINISH_ROW_FAULT_INJECTION

SslmGpuStatus SslmGpuSeqFinishTokenForG5BridgeImpl(SslmGpuContext* ctx, SslmGpuSequenceHandle* seq,
                                                int32_t* out_token) {
	if (!out_token) return SSLM_SEQUENCE_KV_BUFFER_MISMATCH;
	*out_token = -1;
	if (!ctx || !seq || !seq->model || seq->ctx != ctx) {
		return SSLM_SEQUENCE_KV_BUFFER_MISMATCH;
	}
	// The same-sequence claim (strike B1): another thread's decode or finish of this handle makes
	// this call SSLM_BUSY before it reads anything; re-entered when the bridge decode calls it.
	const SeqDecodeClaim claim(seq);
	if (!claim.held()) return SSLM_BUSY;
	if (seq->state() != superslm_gpu::SslmSequenceGpuState::Idle) {
		return SSLM_BUSY;  // an in-flight Submitted call must be drained (sslm_gpu_ready) first.
	}
	// EntrySubmitGuard (§4.4): the finish's device logits (RunDeviceLogits) reset and submit on the
	// context's own device only, so no ledger term; scope by holder. After the own-handle Busy
	// refusal, before `bind_eligible`.
	if (const SslmGpuStatus g = EntrySubmitGuard(ScopeForHolder(seq), GuardDevice::kPerContextOnly);
	    g != SSLM_OK) {
		return g;
	}
	seq->bind_eligible = false;
	SslmGpuModelHandle* model = seq->model;
	if (seq->layer_index != model->num_hidden_layers) {
		// This token's own layer loop has not reached full depth yet -- the caller's own
		// dispatch-budget loop must submit more sslm_decode_step_gpu calls first. No dedicated
		// enumerator exists for "not ready yet" on this bridge's own small surface; same
		// "no more specific status" disposition the whole production file already uses.
		return SSLM_SEQUENCE_REJECTED;
	}

	// final_norm + logits -- the EXACT two calls sslm_decode_step's own finishing block makes
	// (src/sslm_abi.cpp), on this sequence's own GPU-derived, already-read-back hidden_codes
	// (`seq->hidden_codes` -- `live_state.hidden_codes` aliases it directly, so sslm_gpu_ready's
	// own readback already landed here, no separate sync needed).
	std::vector<int8_t> final_codes(model->hidden_size);
	superslm::CarriedScale final_scale{};
	superslm::SslmForwardStatus fst = superslm::RmsNormSite(
	    seq->hidden_codes.data(), model->final_norm_gain.data(), model->hidden_size,
	    seq->hidden_scale, model->final_norm_site_constant, final_codes.data(), &final_scale,
	    "final_norm", /*token_index=*/0, /*trace_hook_state=*/nullptr,
	    /*external_wide_scratch=*/nullptr);
	if (fst != superslm::SslmForwardStatus::Ok) return MapDecodedStatusToGpuStatus(fst);

	std::vector<int64_t> wide_logits(static_cast<size_t>(model->vocab_size));
	std::vector<int32_t> logit_row(static_cast<size_t>(model->vocab_size));
	if (model->device_logits) {
		// T-2851 (B): the exact int64 row, computed on the device from the resident head, then the
		// unchanged host narrowing -- the same call on the same values as the host path below.
		const SslmGpuStatus device_status = RunDeviceLogits(ctx, *model->device_logits,
		                                                    final_codes.data(), wide_logits.data());
		if (device_status != SSLM_OK) return device_status;
		fst = superslm::NarrowRowChecked(wide_logits.data(), static_cast<size_t>(model->vocab_size),
		                                 logit_row.data());
	} else {
		// T-2851 (A): the host row, through the context's parallel-for hook when one is installed;
		// with none, LogitsSiteParallel runs LogitsSite itself. Reads `head_rows`, the model's one
		// host copy of its head (design Sec4.6a).
		fst = superslm::LogitsSiteParallel(final_codes.data(), model->hidden_size, model->head_rows,
		                                   static_cast<size_t>(model->vocab_size),
		                                   wide_logits.data(), logit_row.data(),
		                                   &ctx->host_parallel_for);
		if (fst == superslm::SslmForwardStatus::ParallelForIncomplete) {
			// The host's run broke the exactly-once contract; no token was produced. Rest as the
			// dead end does: ready_for_logits re-armed, walk state and layer index unchanged, so
			// the finish can be retried.
			seq->ready_for_logits = true;
			return SSLM_GPU_PARALLEL_FOR_INCOMPLETE;
		}
	}
	if (fst != superslm::SslmForwardStatus::Ok) return MapDecodedStatusToGpuStatus(fst);

	MaybeInjectGpuFinishDegenerateLogitRow(logit_row.data(), model->vocab_size);

	// G5-5: masking applies to int32 logits BEFORE argmax, indexed by this sequence's own
	// DFA-walk-state -- SAME code (superslm::ApplyMaskAndArgmax) as the CPU path's masked-argmax
	// step; no schema bound is byte-for-byte the pre-G5 path (plain ArgmaxLowestIndexTieBreak),
	// exactly the base-level GPU parity this bridge extends.
	int32_t produced = 0;
	if (seq->bound_schema_index >= 0) {
		const superslm::SchemaEntry* entry =
		    model->schemas.ByIndex(static_cast<size_t>(seq->bound_schema_index));
		if (!entry) return SSLM_SEQUENCE_REJECTED;
		const uint8_t* page = entry->mask_pages +
		                       static_cast<size_t>(seq->dfa_walk_state) * model->schemas.MaskPageBytes();
		superslm::ApplyMaskAndArgmax(logit_row.data(), page, model->vocab_size, &produced);
		// T-2844/D-SLM7463 (§3.10.1): the walk advances to whatever state `produced` reaches ONLY
		// when a matching CSR transition entry actually exists -- Sec13.2 permits an accepting
		// state's own mask page to be legitimately all-zero, and at such a state (or at a
		// synthetic degenerate row, above) masked argmax forces every logit to INT32_MIN and the
		// lowest-index tie-break returns token 0, for which no CSR row entry exists. Before this
		// fix that failure was silently discarded (`next_state` written unconditionally) and the
		// walk froze in place, emitting token 0 forever at SSLM_OK. This is a genuine, defined
		// dead end, not a caller error -- `*out_token` reserves -2 for it (D-SLM3476), the
		// identical status the CPU path returns for the same event (`sslm_abi.cpp`'s own two miss
		// sites) -- never `SSLM_SEQUENCE_REJECTED`, which this bridge already uses for a different
		// condition (layer loop not at full depth, above). `dfa_walk_state` is left unchanged
		// (`Transition` does not write `next_state` on a miss, so simply not writing it back closes
		// the pre-existing unconditional-write bug at this same call site). `ready_for_logits` is
		// re-armed: GPU's own composition gates its embed-skip on `ready_for_logits`, not on
		// `layer_index` the way CPU's ABI does, and `sslm_gpu_seq_embed_tokenImpl` carries no
		// `layer_index` precondition at all -- left un-re-armed, a further decode call on the same
		// sequence would embed the caller's next token and re-drive the full layer loop before
		// Finish runs again. With `ready_for_logits` re-armed, the next composed call takes its own
		// `ready_for_logits` branch, skips the embed and the layer drive, and calls Finish directly
		// on the unchanged `hidden_codes`, reproducing the identical `-2`/`SSLM_OK` result
		// deterministically and consuming no new token from the caller.
		uint32_t next_state = seq->dfa_walk_state;
		const bool has_transition = model->schemas.Transition(
		    *entry, seq->dfa_walk_state, static_cast<uint32_t>(produced), &next_state);
		if (!has_transition) {
			// dead-end preserves dfa_walk_state
			*out_token = -2;
			seq->ready_for_logits = true;
			return SSLM_OK;
		}
		seq->dfa_walk_state = next_state;
	} else {
		produced = superslm::ArgmaxLowestIndexTieBreak(logit_row.data(),
		                                                static_cast<size_t>(model->vocab_size));
	}
	*out_token = produced;
	// Resting convention -- matches CPU's identical `seq->state.layer_index = 0` reset
	// (sslm_decode_stepImpl, src/sslm_abi.cpp) so the next embed starts a fresh token boundary.
	seq->layer_index = 0;
	seq->live_state.layer_index = 0;
	return SSLM_OK;
}

namespace {
// G5-5 session 3 fix (T-2132, Brunel): the SAME "drive one token's own layer loop to full depth,
// possibly across several sslm_decode_step_gpu calls" loop every bridge entry point below needs
// -- extracted once here (internal linkage) so `SslmGpuSeqPrefillPromptForG5Bridge` and
// `SslmGpuSeqDecodeStepForG5Bridge` (below) share one implementation rather than each hand-rolling
// its own copy, which is exactly the shape of duplication that let the ready_for_logits gap slip
// past this ticket's own first parity harness (session 2/3's own finding: the harness's `tools/
// t2132_g5_gpu_parity_gpu.cpp` had its own private copy of this loop, and its own private, buggy
// prompt-prefill composition around it).
// 1.8.0 (TE-426): returns the status that stopped the drive, as it got it -- the submit's, the
// ready call's, or the drained step's -- instead of a bool its callers turned into
// SSLM_DEVICE_LOST, which discarded SSLM_GPU_ALLOCATION_FAILED. SSLM_OK once at full depth.
SslmGpuStatus DriveGpuSeqToFullDepthForG5Bridge(SslmGpuContext* ctx, SslmGpuSequenceHandle* seq,
                                                uint32_t dispatch_budget) {
	SslmGpuModelHandle* model = seq->model;
	uint32_t guard = 0;
	while (seq->layer_index < model->num_hidden_layers) {
		// T-2243 (S2, design plan Sec6.1): reads seq's own currently-bound adapter (null if
		// none) and threads it into the existing per-call `adapter_or_null` machinery, which is
		// what populates `in_flight_adapter`/`submitted_sequences` for this call exactly as it
		// does today for a direct caller of `sslm_decode_step_gpu`.
		const SslmGpuStatus submitted =
		    sslm_decode_step_gpu(ctx, seq, seq->bound_adapter, dispatch_budget);
		if (submitted != SSLM_OK) return submitted;
		int32_t ready = 0;
		SslmGpuStatus drained = SSLM_OK;
		const SslmGpuStatus ready_status = sslm_gpu_ready(ctx, seq, /*block=*/1, &ready, &drained);
		if (ready_status != SSLM_OK) return ready_status;
		if (drained != SSLM_OK) return drained;
		// Runaway-loop guard, never expected in practice; SSLM_DEVICE_LOST, the value this drive
		// returned for it before 1.8.0.
		if (++guard > 10000) return SSLM_DEVICE_LOST;
	}
	return SSLM_OK;
}
}  // namespace

namespace {

// T-2169 (Rung 3/4, design Claude/Vitruvius/t2169-gpu-batched-prefill-design-2026-08-18.md Sec5/
// Sec5.1/Sec8, D-SLM3590/3595/3611-3613/3619-3622/3631-3634): the cause a chunk's own admission
// count is attributable to, resolved once per call by the pre-scan below -- reused by both G5
// bridge functions' own cause-routed return (Sec5's table).
// Paged KV (G3): kBudget is kPositionCap's twin for a pooled holder whose limit (§3.4) is below
// the cap -- the position term stopped at the limit, not the cap (SSLM_GPU_KV_BUDGET_EXCEEDED).
enum class ChunkAdmissionCause { kNone, kDfa, kEmbed, kPositionCap, kBudget };

// Host-side sequential pre-scan (design Sec5), run to completion over the intended chunk before
// any dispatch is recorded. `dfa_entry` is null for the prompt twin (no reachability check --
// `dfa_admit_count` defaults to `chunk_len`, design Sec5's own "the two GPU-schema-only terms
// default to chunk_len for the prompt twin"); non-null for the schema twin. Produces `admit_count`
// (the min of the applicable terms), the `cause` that produced it (DFA > EMBED > POSITION_CAP,
// matching the shipped per-token loop's own check order so a tie resolves identically to what one
// more per-token round-trip would find), every admitted token's own pre-packed embedding bytes
// (the `[0, SeqScaleOff(hidden_size)+16)` layout `SubmitChunkToFullDepthForG5Bridge` expects,
// `superslm_gpu::T2169SeqEmbeddingBlockBytes`), and -- schema twin only -- the DFA walk state
// after each admitted transition, so the caller can commit the state after exactly however many
// tokens actually end up committed (`admit_count`, or a device-derived shorter count, D-SLM3622).
struct ChunkPreScanResult {
	uint32_t admit_count = 0;
	ChunkAdmissionCause cause = ChunkAdmissionCause::kNone;
	SslmGpuStatus embed_reject_status = SSLM_OK;    // meaningful iff cause == kEmbed.
	std::vector<uint8_t> chunk_embedding_bytes;     // admit_count * block_bytes, packed.
	std::vector<uint32_t> dfa_states_after;         // schema twin only; dfa_states_after[t] is the
	                                                 // walk state after admitting token t, valid
	                                                 // for t in [0, min(dfa_admit_count,
	                                                 // dfa_embed_scan_limit)) -- the vector holds
	                                                 // at most dfa_embed_scan_limit entries even
	                                                 // when dfa_admit_count defaults to n (D-SLM3689
	                                                 // bounded the scan without narrowing this
	                                                 // default); admit_count never exceeds either
	                                                 // bound, so every caller's own index stays in
	                                                 // range.
};

ChunkPreScanResult RunChunkAdmissionPreScan(SslmGpuModelHandle* model, int64_t chunk_open_ctxlen,
                                             int64_t context_cap, const int32_t* tokens,
                                             int32_t chunk_len, const superslm::SchemaEntry* dfa_entry,
                                             uint32_t dfa_walk_state_start, int64_t limit) {
	ChunkPreScanResult r;
	const uint32_t n = static_cast<uint32_t>(chunk_len);

	// D-SLM3613: position_admit_count, O(1) -- position advances by exactly 1 per token, so no
	// per-token loop is needed (design Sec5). Paged-KV plan §3.6 (G3): the holder's limit headroom
	// beside the cap headroom -- `limit` is min(origin + budget, cap) for a pooled holder and the cap
	// for a dedicated one, so a dedicated holder's headroom is today's exactly.
	const bool budget_limited = limit < context_cap;
	int64_t cap_room = (budget_limited ? limit : context_cap) - chunk_open_ctxlen;
	if (cap_room < 0) cap_room = 0;
	const uint32_t position_admit_count = static_cast<uint32_t>(std::min<int64_t>(n, cap_room));

	// T-2189 finding 4 (P2, D-SLM3689): NO token past `position_admit_count` can ever be admitted
	// -- the position cap already excludes it regardless of what the DFA/embed scans would find
	// there -- so scanning DFA/embed all the way to `n` (the full REQUESTED chunk length, which a
	// caller can set arbitrarily large) does unbounded work/allocation for a chunk that is mostly
	// or entirely inadmissible on cap grounds alone: O(count * hidden_size) instead of the O(1)-ish
	// cost a full-cap rejection should have.
	//
	// Bounding the scan to `position_admit_count` tokens is not quite enough on its own: `cause`
	// (not just the numeric `admit_count`) must match the shipped per-token loop's own DFA-before-
	// EMBED-before-POSITION_CAP priority (D-SLM3620) at a TIE -- if the one token exactly AT the
	// position boundary (index `position_admit_count`, the first index the cap itself already
	// excludes) would ALSO be DFA- or embed-rejected, the correct `cause` is still the higher-
	// priority DFA/EMBED, even though the numeric `admit_count` is identical either way. Examining
	// that one extra boundary token (`+ 1`) is what disambiguates the tie without re-introducing
	// the unbounded scan: `dfa_embed_scan_limit` is `position_admit_count + 1` tokens (clamped to
	// `n`, so a chunk that fits entirely under the cap is unaffected and still scans its own full
	// length). A huge `count` at a tight cap therefore costs O(position_admit_count) work, not
	// O(count) -- O(1)-ish at a full cap (`position_admit_count == 0`, scan_limit == 1).
	const uint32_t dfa_embed_scan_limit =
	    std::min(n, position_admit_count + 1u);  // `n <= INT32_MAX` (chunk_len is int32_t), so this
	                                              // addition cannot overflow uint32_t.

	// D-SLM3590: dfa_admit_count, run over [0, dfa_embed_scan_limit) -- schema twin only. Tracks
	// the walk state after each admitted transition. A rejection found in this bounded range is
	// identical to what the unbounded scan would have found (nothing past the position boundary
	// can change `admit_count`, and the boundary token itself is examined for `cause` priority,
	// per this function's own header comment above) -- defaulting to `n` (not `dfa_embed_scan_limit`)
	// when nothing is rejected preserves the existing "unrejected within range" semantics the
	// admit_count/cause resolution below already depends on.
	uint32_t dfa_admit_count = n;
	if (dfa_entry) {
		r.dfa_states_after.reserve(dfa_embed_scan_limit);
		uint32_t state = dfa_walk_state_start;
		for (uint32_t t = 0; t < dfa_embed_scan_limit; ++t) {
			uint32_t next_state = 0;
			if (!model->schemas.Transition(*dfa_entry, state, static_cast<uint32_t>(tokens[t]),
			                                &next_state)) {
				dfa_admit_count = t;
				break;
			}
			state = next_state;
			r.dfa_states_after.push_back(state);
		}
	}

	// D-SLM3621: embed_admit_count, run over [0, dfa_embed_scan_limit) -- the SAME pass that
	// produces the record-time embedding bytes 2b needs, for every candidate index within the
	// bounded range (see this function's own header comment above for why bounding here is sound).
	uint32_t embed_admit_count = n;
	SslmGpuStatus embed_reject_status = SSLM_OK;
	const uint32_t block_bytes = superslm_gpu::T2169SeqEmbeddingBlockBytes(model->hidden_size);
	const uint32_t scale_offset = block_bytes - 16u;
	std::vector<uint8_t> embed_bytes_cache;  // block_bytes per validated candidate index
	embed_bytes_cache.reserve(static_cast<size_t>(block_bytes) * dfa_embed_scan_limit);
	for (uint32_t t = 0; t < dfa_embed_scan_limit; ++t) {
		const int32_t token_id = tokens[t];
		if (token_id < 0 || token_id >= model->vocab_size) {
			embed_admit_count = t;
			embed_reject_status = SSLM_TOKEN_ID_OUT_OF_RANGE;
			break;
		}
		std::vector<int8_t> embed_codes(model->hidden_size);
		superslm::CarriedScale embed_scale{};
		const superslm::SslmForwardStatus est = superslm::EmbedEntry(
		    token_id, model->vocab_size, model->embed_weights.data(),
		    static_cast<size_t>(model->hidden_size), model->embed_site_constant, embed_codes.data(),
		    &embed_scale);
		if (est != superslm::SslmForwardStatus::Ok) {
			// EmbedEntry's own only rejection is the identical token_id bounds check already
			// performed above -- unreachable in practice, handled the same way
			// sslm_gpu_seq_embed_token handles it (this file, above).
			embed_admit_count = t;
			embed_reject_status = SSLM_TOKEN_ID_OUT_OF_RANGE;
			break;
		}
		const size_t base = embed_bytes_cache.size();
		embed_bytes_cache.resize(base + block_bytes, 0);
		for (uint32_t i = 0; i < model->hidden_size; ++i) {
			const int32_t v = static_cast<int32_t>(embed_codes[i]);
			std::memcpy(embed_bytes_cache.data() + base + static_cast<size_t>(i) * 4u, &v, 4);
		}
		std::memcpy(embed_bytes_cache.data() + base + scale_offset, &embed_scale.m, 8);
		std::memcpy(embed_bytes_cache.data() + base + scale_offset + 8, &embed_scale.e, 8);
	}

	// D-SLM3620: admit_count = min of the applicable terms. `admit_count == chunk_len` is checked
	// FIRST and is unconditionally NONE (every term trivially equals chunk_len when nothing is
	// rejected, so the priority chain below is only meaningful once an actual rejection -- a term
	// strictly less than chunk_len -- exists); otherwise cause is resolved by the shipped
	// per-token loop's own check order (DFA before embed before the submit-time position-cap
	// guard).
	r.admit_count = std::min({dfa_admit_count, embed_admit_count, position_admit_count});
	if (r.admit_count == n) {
		r.cause = ChunkAdmissionCause::kNone;
	} else if (dfa_entry && dfa_admit_count == r.admit_count) {
		r.cause = ChunkAdmissionCause::kDfa;
	} else if (embed_admit_count == r.admit_count) {
		r.cause = ChunkAdmissionCause::kEmbed;
		r.embed_reject_status = embed_reject_status;
	} else {
		r.cause = budget_limited ? ChunkAdmissionCause::kBudget : ChunkAdmissionCause::kPositionCap;
	}

	r.chunk_embedding_bytes.assign(
	    embed_bytes_cache.begin(),
	    embed_bytes_cache.begin() + static_cast<size_t>(r.admit_count) * block_bytes);
	return r;
}

// T-2169 (Rung 3/4, design Sec5 step 5/D-SLM3614/D-SLM3622): dispatches exactly `admit_count`
// pre-scanned, pre-embedded tokens through the chunk-submission primitive
// (`superslm_gpu::SubmitChunkToFullDepthForG5Bridge`, internally TDR-safe-split) and drains the
// final (sub-)chunk's own async token to completion, copying the decoded result back into this
// handle's own canonical fields -- the same copy-back `sslm_gpu_ready` performs, since this call
// bypasses that function entirely (this bridge owns the whole Submit/Finish pair for a chunk
// call). `*out_derived_count` is the device's own actually-committed count, derived from
// `seq->context_length`'s post-chunk delta against `chunk_open_ctxlen` -- never assumed to equal
// `admit_count` (D-SLM3614): a device-domain rejection the pre-scan could not see (the sticky-tag
// mechanism, P3(b)/S4(b)) or a mid-recording infrastructural fault (the tenth failure origin,
// D-SLM3634) both show up here as a short count, which the caller checks against `admit_count` to
// decide whether the device-computed override (D-SLM3622) applies -- this function itself reports
// no status of its own beyond the derived count.
//
// `*out_guard_rejected` (plan te266 Sec3.6 item 2) says which of those two a short count came from:
// true iff a device-side domain guard refused, i.e. a non-final sub-chunk's finish decoded a
// sticky tag (`SubmitChunkToFullDepthForG5Bridge`'s own flag) or this function's final
// `RunLayerLoopGpuFinish` returned a status other than `Ok`, `GpuDeviceRemoved` or
// `GpuAllocationFailed`. It is classified by where the status came from, never by its value alone:
// a submission failure (including the recording fault, whose status is an arithmetic-guard value)
// and every catch clause below leaves it false.
//
// `*out_fault` (1.8.0, TE-426) is the forward status that cut the chunk short when the cause was
// not a guard refusal: the failing submission's status, the final finish's `GpuDeviceRemoved`/
// `GpuAllocationFailed`/`GpuOperationFailed`, or the classified fault the `runtime_error` clause
// or the catch-all (TE-435) below caught. `Ok` otherwise. The twins read it on a short count, so an allocation failure on a
// live device reaches the caller as SSLM_GPU_ALLOCATION_FAILED instead of being discarded.
void SubmitAdmittedChunkForG5Bridge(SslmGpuModelHandle* model, SslmGpuSequenceHandle* seq,
                                     const uint8_t* chunk_embedding_bytes, uint32_t admit_count,
                                     int64_t chunk_open_ctxlen, int64_t* out_derived_count,
                                     bool* out_guard_rejected,
                                     superslm::SslmForwardStatus* out_fault) {
	*out_derived_count = 0;
	*out_guard_rejected = false;
	*out_fault = superslm::SslmForwardStatus::Ok;
	if (admit_count == 0) return;
	// Mirrors `sslm_gpu_seq_embed_token`'s own unconditional `layer_index = 0` reset for a VALID
	// token, which the shipped per-token loop always performs for token 0 before any guard or
	// dispatch runs -- this batched path bypasses `sslm_gpu_seq_embed_token` entirely (design
	// Sec5.1's own "bypassing that wrapper's own state-transition/host-mirror side effects"), so
	// without this, `PrepareGpuLayerLoopChunkOpenState`'s own `SequenceAlreadyComplete` guard
	// (`seq.layer_index >= num_hidden_layers`) would reject every call after the first: a
	// committed token's own last-layer commit leaves the DEVICE's `layer_index` at
	// `num_hidden_layers` (D-SLM3649's own finding), and nothing else on this path ever resets
	// the HOST mirror between separate calls. Only fires when a token WILL actually be dispatched
	// (admit_count > 0), matching embed_token's own reset firing only for a validating token.
	seq->layer_index = 0;
	seq->live_state.layer_index = 0;
	static const superslm::SslmTensorManifest kEmptyManifest;
	// T-2185 remedy N1 (Brunel fix round 2, D-SLM3674): the T-2184 remedy (S1) opened this window
	// AFTER `SubmitChunkToFullDepthForG5Bridge` returned, which is also after that call's own
	// internal loop has already submitted-and-synchronously-finished every non-final sub-chunk
	// (`SubmitChunkToFullDepthForG5Bridge`'s own header comment, superslm_gpu.cpp: "Every
	// non-final sub-chunk is finished SYNCHRONOUSLY, here" -- `RunLayerLoopGpuFinish(...,
	// block=1)` inside that loop). So the window covered only the FINAL sub-chunk's own fence
	// wait -- under 3.2% of the call's wall time at a 128-token chunk (T-2185 N1's own derivation)
	// -- and every earlier sub-chunk's fence wait ran with `model->submitted_sequences == 0`,
	// exactly the reachable consequence the T-2184 review named, still reachable across the
	// other 96.8% of the call. The window now opens HERE, before the first sub-chunk is ever
	// submitted, so every sub-chunk's own fence wait -- not only the final one -- falls inside it;
	// `seq->in_flight` is assigned once the call returns the final sub-chunk's own inflight token
	// (the only one handed back to this caller unfenced; every earlier one already completed
	// inside the call).
	//
	// T-2186 remedy P1 (Brunel fix round 3, D-SLM3682, confirmation
	// Claude/Poirot/8642652-t2186-t2169-fix2-confirmation.md): opening and closing this window as
	// two separate manual statements -- open here, close unconditionally below -- was NOT
	// symmetric with the shipped per-token path in the property that matters: the call in between
	// can throw. `SubmitChunkToFullDepthForG5Bridge`'s own tail
	// (`SubmitOneSubChunkToFullDepthForG5Bridge`'s closing statements -- `dev.list->Close()`,
	// `dev.queue->Signal()`, `new GpuLayerLoopInFlight()`, cited by name, not line) sits OUTSIDE that
	// function's own try/catch, so a failed `Close()`/`Signal()` (`SSLM_GPU_HR` throws
	// `std::runtime_error` -- precisely the device-removed channel design Sec9 promises is
	// deliverable) or a `std::bad_alloc` from the `new` can escape this call with no catch
	// anywhere between here and the ABI boundary. A manual "close below" statement is never
	// reached on that path: `seq->state` stays `Submitted` and `model->submitted_sequences` stays
	// incremented forever, and `sslm_gpu_model_unmap` (`SslmGpuModelHandle::submitted_sequences`'
	// own check, above) returns `SSLM_BUSY` on every later call, permanently -- the model can
	// never be unmapped. The shipped per-token twin (`SubmitOneSequenceDecode`, this file,
	// `:1379-1381`'s own comment: "Rejected before submission (a guard, or an exception) -- seq/
	// host state untouched, still Idle") avoids this class entirely by opening its window only
	// AFTER submission returns; this batched path cannot do the same (design Sec9 wants the
	// window held across the whole chunk, submitted or not, not only after it succeeds), so it is
	// made safe by construction instead: an RAII guard opens the window in its constructor and
	// closes it in its destructor, which C++ runs on every exit from this function -- normal
	// return AND exception unwind alike -- so there is no longer any path between open and close
	// where a throw can leave the window stuck open. This replaces the manual open above and the
	// manual unconditional close that used to sit after the call, below; the close is no longer a
	// second statement that can be skipped, it is a property of the guard's lifetime.
	// T-2243 (S2, design plan Sec6.1): the local GpuAdapterBridge SubmitOneSequenceDecode
	// already builds for the per-token path -- built here, on this path, from seq's own
	// currently-bound adapter, since this chunk path bypasses SubmitOneSequenceDecode entirely
	// and previously had "no adapter bookkeeping to mirror" (this function's own prior comment,
	// now superseded by this fix). `num_hidden_layers` is taken from `model`, not from
	// `seq->bound_adapter` -- the adapter handle carries no such field (confirmed at
	// SubmitOneSequenceDecode's own identical construction, this file, `:1368`).
	superslm_gpu::GpuAdapterBridge adapter_bridge;
	const superslm_gpu::GpuAdapterBridge* adapter_bridge_ptr = nullptr;
	if (seq->bound_adapter != nullptr) {
		adapter_bridge.lora_ab_resident = seq->bound_adapter->lora_ab_buf.Get();
		adapter_bridge.fold_resident = seq->bound_adapter->fold_buf.Get();
		adapter_bridge.rank = seq->bound_adapter->rank;
		adapter_bridge.slots = &seq->bound_adapter->slots[0][0];
		adapter_bridge.num_hidden_layers = model->num_hidden_layers;
		adapter_bridge_ptr = &adapter_bridge;
	}
	// Paged-KV plan §4 items 2, 5 and 6, step G3: a pooled holder binds its own view over the pool's
	// host mirror, its table buffer (uploaded at the head of the chunk's first sub-chunk when its
	// mappings changed), the pool's buffer and the pool's latch.
	superslm_gpu::GpuPagedKvBinding paged{};
	const superslm_gpu::GpuPagedKvBinding* paged_ptr = nullptr;
	if (seq->pool != nullptr) {
		paged.view = superslm::KvPageView{seq->pool->host_mirror.data(), seq->holder->table.get(),
		                                  superslm::kv_pages::HolderMapped(seq->holder),
		                                  seq->pool->geo.page_positions, seq->pool->geo.page_bytes};
		paged.table_resident = seq->table_buf.Get();
		paged.upload_table = seq->table_dirty;
		paged_ptr = &paged;
	}
	ID3D12Resource* const kv_resident = seq->pool != nullptr ? seq->pool->pool_buf.Get() : seq->kv_buf.Get();
	bool* const kv_latch = seq->pool != nullptr ? &seq->pool->kv_needs_resume_barrier : &seq->kv_needs_resume_barrier;
	struct SubmittedWindowScopeGuard {
		SslmGpuSequenceHandle* seq;
		SslmGpuModelHandle* model;
		SubmittedWindowScopeGuard(SslmGpuSequenceHandle* s, SslmGpuModelHandle* m) : seq(s), model(m) {
			// T-2243 review finding 2 (D-SLM4113) and T-2243 (S2, design plan Sec6.1): the model's
			// and the process's counts and the bound adapter's (`in_flight_adapter`), so a
			// concurrent `sslm_gpu_adapter_unmap` sees the in-flight bind on this path too -- and,
			// for a pooled handle, the pooled count and the pool's holder pointer -- raised by the
			// counting accessor (G3, §4.4). `model` is the handle's own.
			(void)model;
			SetGpuSeqState(seq, superslm_gpu::SslmSequenceGpuState::Submitted, seq->bound_adapter);
		}
		~SubmittedWindowScopeGuard() {
			SetGpuSeqState(seq, superslm_gpu::SslmSequenceGpuState::Idle, nullptr);
		}
	} submitted_window_guard(seq, model);
	superslm_gpu::GpuInFlightToken inflight;
	// The containment every non-rethrowing clause below applies, written once so the clauses
	// cannot diverge. Call only from inside a catch handler (ClassifyInFlightException rethrows
	// the exception being handled). 1.8.0 (TE-426): the fault is classified and recorded instead
	// of dropped -- an allocation type (GpuAllocationError, std::length_error) is an allocation
	// failure, every other type a non-allocation one. Nothing that can hold the process's command
	// list open reaches here (the recording windows and tails catch first), so rules 0 and 1 are
	// the recording sites' own.
	auto record_contained_fault = [&]() {
		seq->in_flight = superslm_gpu::GpuInFlightToken();
		*out_guard_rejected = false;
		*out_fault = superslm_gpu::harness::ClassifyInFlightException() ==
		                     superslm_gpu::harness::GpuFaultKind::Allocation
		                 ? superslm::SslmForwardStatus::GpuAllocationFailed
		                 : superslm::SslmForwardStatus::GpuOperationFailed;
	};
	// T-2189 finding 2 (D-SLM3689) added this try/catch because `SubmitChunkToFullDepthForG5Bridge`'s
	// own tail (`SubmitOneSubChunkToFullDepthForG5Bridge`'s closing statements -- `dev.list->Close()`,
	// `dev.queue->Signal()`, `new GpuLayerLoopInFlight()`, superslm_gpu.cpp) sat OUTSIDE that
	// function's own internal try/catch at the time (see the T-2186 P1 comment on
	// `SubmittedWindowScopeGuard` above for the full history), so a failed `Close()`/`Signal()`
	// (`SSLM_GPU_HR` throws `std::runtime_error`) or a `std::bad_alloc` from the `new` could escape
	// this call with the underlying D3D12 command list left open/recording -- T-2191 S6/D-SLM3692's
	// own out-of-scope observation. T-2189 finding 6's own fix round closed that gap at its source
	// (`SubmitOneSubChunkToFullDepthForG5Bridge` now catches its own tail's throw, closes the list,
	// and returns a `superslm::SslmForwardStatus` instead of throwing), so neither exception type
	// reaches this catch through that call chain any more -- `submit_status` carries the outcome
	// (`GpuDeviceRemoved`/`GpuAllocationFailed`) through the ordinary `submit_status != Ok` path
	// below instead. This try/catch is kept as defense-in-depth for any exception type not caught
	// at the source (a third, unenumerated type escaping `SubmitOneSubChunkToFullDepthForG5Bridge`
	// would otherwise propagate uncaught to this call's own caller), not as this failure class's
	// own containment path any more. `RunLayerLoopGpuFinish` is included in the same try for
	// symmetry with this call's own tail -- both are the same class of D3D12 call -- though
	// `RunLayerLoopGpuFinish` cannot itself throw: since 1.8.0 (TE-435) its ladder ends in
	// `catch (...)` and converts every exception type to a status before returning (see its own
	// definition, superslm_gpu.cpp). The exception that reaches this try is the sub-chunk
	// recording window's catch-all rethrow, after it has closed the list.
	//
	// No status is returned from THIS function (it never has been -- see its own header comment);
	// the std::runtime_error clause and the catch-all below deliberately do NOT return early or
	// rethrow (the std::bad_alloc clause rethrows to the API boundary). `seq->live_state` is
	// mutated BY REFERENCE, incrementally, by every (sub-)chunk that completed and finished
	// synchronously BEFORE the one that threw (`SubmitChunkToFullDepthForG5Bridge`'s own multi-
	// sub-chunk loop, superslm_gpu.cpp) -- so falling through to the unchanged copy-back below
	// (which reads `seq->live_state`) derives `*out_derived_count` from exactly how much of the
	// chunk genuinely committed, the SAME partial-consumption contract every other rejection cause
	// on this path already honors (D-SLM3614/D-SLM3622). Both public callers already test
	// `derived_count < admit_count` BEFORE consulting `scan.cause` and return `SSLM_DEVICE_LOST` on
	// that path -- since `admit_count > 0` is already established above (the `admit_count == 0`
	// early return, this function's own top), a caught throw here that leaves `*out_derived_count`
	// short of `admit_count` is guaranteed to resolve to `SSLM_DEVICE_LOST` (or, since 1.8.0,
	// `SSLM_GPU_ALLOCATION_FAILED` for a recorded allocation fault) through that EXISTING logic,
	// with no new status-mapping code duplicated at either call site.
	try {
		// T-2432 (Track A step 2/3): q_width threaded explicitly -- this is the real GPU-side
		// prefill submission path, the direct analog of sslm_abi.cpp's own
		// RunLayerLoopChunkBatched call site.
		// (T-2577 round 2, D-SLM6278): `model_generation`, the model's own `content_hash`
		// (SHA-256, D-SLM3415) reduced to its first 8 bytes -- an opaque, stable-per-load
		// identity, matching the contract every other caller of this parameter now supplies.
		// This 1.0 handle path binds `weights_buf`/`rope_cos_buf`/`rope_sin_buf` directly
		// (the `external_*` arguments below), which already bypasses
		// `g_resident_weights`/`g_resident_kv`/`g_resident_rope` entirely (D-SLM3362) -- so this
		// identity is inert on the path this call actually takes today. Wired anyway, so the
		// identity is already correct if a future change ever routes this handle through the
		// pre-1.0 caches instead.
		uint64_t model_generation = 0;
		std::memcpy(&model_generation, model->content_hash.data(), sizeof(model_generation));
		bool nonfinal_guard_rejected = false;
		const superslm::SslmForwardStatus submit_status =
		    superslm_gpu::SubmitChunkToFullDepthForG5Bridge(
		        seq->live_state, /*layers=*/nullptr, model->num_hidden_layers, model->hidden_size,
		        model->head_dim, model->num_key_value_heads, model->intermediate_size,
		        seq->context_cap, kEmptyManifest, KvHostBase(seq), KvHostBytes(seq),
		        chunk_embedding_bytes, admit_count, kv_resident, kv_latch, model->weights_buf.Get(),
		        model->rope_cos_buf.Get(), model->rope_sin_buf.Get(), model->has_rope_tables,
		        model->rope_cos_elem_count, model->rope_sin_elem_count, adapter_bridge_ptr, &inflight,
		        /*q_width=*/static_cast<size_t>(model->num_attention_heads) * model->head_dim,
		        model_generation, model->has_qk_norm, &nonfinal_guard_rejected, paged_ptr);
		// A sub-chunk that reached the device uploaded the table at its head (§4 item 5).
		if (seq->pool != nullptr && (submit_status == superslm::SslmForwardStatus::Ok || nonfinal_guard_rejected)) {
			seq->table_dirty = false;
		}
		*out_guard_rejected = nonfinal_guard_rejected;
		if (submit_status != superslm::SslmForwardStatus::Ok && !nonfinal_guard_rejected) {
			*out_fault = submit_status;  // 1.8.0 (TE-426): kept, not discarded
		}
		if (submit_status == superslm::SslmForwardStatus::Ok && inflight) {
			// `SubmitChunkToFullDepthForG5Bridge` returns the FINAL (sub-)chunk's own inflight token
			// genuinely unfenced (its own header comment: "the caller's own async contract... only
			// the FINAL (sub-)chunk's own inflight token is handed back unfinished"), so the fence
			// for that last sub-chunk has not yet signaled at this point -- a concurrent caller on
			// another thread (e.g. `sslm_gpu_model_unmap`, `sslm_gpu_seq_save`) still genuinely
			// observes the window while `RunLayerLoopGpuFinish` below blocks on it.
			seq->in_flight = std::move(inflight);
			int32_t ready = 0;
			const superslm::SslmForwardStatus final_finish_status = superslm_gpu::RunLayerLoopGpuFinish(
			    std::move(seq->in_flight), seq->live_state, KvHostBase(seq), /*block=*/1, &ready);
			// A blocking finish always moves from the token, so `seq->in_flight` is empty here.
			// The final sub-chunk's finish: `GpuDeviceRemoved`/`GpuAllocationFailed`/
			// `GpuOperationFailed` come only from its catch or a null token; any other non-`Ok`
			// value is a sticky-tag decode (gpu_port.h, the `out_readback_guard_rejected` comment).
			if (final_finish_status == superslm::SslmForwardStatus::GpuDeviceRemoved ||
			    final_finish_status == superslm::SslmForwardStatus::GpuAllocationFailed ||
			    final_finish_status == superslm::SslmForwardStatus::GpuOperationFailed) {
				*out_fault = final_finish_status;
			} else if (final_finish_status != superslm::SslmForwardStatus::Ok) {
				*out_guard_rejected = true;
			}
		}
	} catch (const std::bad_alloc& e) {
		std::fprintf(stderr,
		             "gpu_1p0: SubmitAdmittedChunkForG5Bridge: bad_alloc escaped the batched-prefill "
		             "submit/finish call, contained at the SslmGpuStatus boundary: %s\n",
		             e.what());
		seq->in_flight = superslm_gpu::GpuInFlightToken();
		*out_guard_rejected = false;
		throw;
	} catch (const std::runtime_error& e) {
		std::fprintf(stderr,
		             "gpu_1p0: SubmitAdmittedChunkForG5Bridge: runtime_error escaped the "
		             "batched-prefill submit/finish call, contained at the SslmGpuStatus boundary: "
		             "%s\n",
		             e.what());
		record_contained_fault();
	} catch (...) {
		// 1.8.0 (TE-435): any other exception type -- a std::length_error, or a non-standard type,
		// which the sub-chunk recording window's catch-all rethrows once it has closed the list.
		// Contained as the std::runtime_error clause above: the fault is classified and recorded
		// (length_error is an allocation failure, a non-standard type a non-allocation one), and the
		// committed sub-chunks are copied back below. Before this clause it left the function with
		// the fault unrecorded and the copy-back skipped.
		std::fprintf(stderr,
		             "gpu_1p0: SubmitAdmittedChunkForG5Bridge: an exception outside std::runtime_error "
		             "escaped the batched-prefill submit/finish call, contained at the SslmGpuStatus "
		             "boundary\n");
		record_contained_fault();
	}
	// `submitted_window_guard`'s destructor closes the window here, unconditionally, on this
	// normal-return path exactly as it would on an exception unwind -- `sslm_gpu_ready`'s own
	// "collapse Submitted -> Idle in one call" symmetry, now a property of the guard's lifetime
	// rather than a second statement a throw could skip. T-2243 (S2): the guard's own
	// constructor/destructor now also owns the bound adapter's `submitted_sequences`/
	// `seq->in_flight_adapter` bookkeeping (above) when `seq->bound_adapter` is non-null; an
	// adapter-less sequence takes the identical no-op path this comment used to describe as
	// unconditional.
	//
	// Copy the (possibly partially-advanced, per the sub-chunk split's own synchronous-finish
	// discipline, D-SLM3596/3649) live_state back into this handle's own canonical fields.
	seq->hidden_scale = seq->live_state.hidden_scale;
	seq->layer_index = seq->live_state.layer_index;
	seq->kv_saturation_count = seq->live_state.kv_saturation_count;
	seq->context_length = seq->live_state.context_length;
	*out_derived_count = seq->context_length - chunk_open_ctxlen;
	if (*out_derived_count < 0) *out_derived_count = 0;  // defensive; not expected by construction.
}

// D-SLM3619: reproduces the shipped per-token loop's own incidental side effect at the
// position-cap boundary -- token[admit_count] is a VALID id (position_admit_count, never embed
// validity, is what capped it, since cause == kPositionCap implies embed_admit_count >
// admit_count), so the shipped loop's own unconditional `sslm_gpu_seq_embed_token(tokens[
// admit_count])` call ALWAYS succeeds before the device-side guard discovers the position-cap
// violation, leaving `hidden_codes`/`layer_index` at that token's own raw (never-dispatched)
// embedding -- exactly the state `SslmGpuSeqFinishTokenForG5Bridge`'s own "not full depth yet"
// check reads to decide whether a subsequent ready_for_logits-shortcut decode step is REJECTED.
// No dispatch is recorded for this token; this is a host-only write, matching
// `sslm_gpu_seq_embed_token`'s own exact effect. Called only when cause == kPositionCap and the
// device did not additionally override the outcome (the device-computed fallback is a distinct,
// untested-by-this-suite mechanism, design Sec6/Sec9's own named gap).
void ReproducePositionCapBoundaryEmbedSideEffect(SslmGpuModelHandle* model,
                                                  SslmGpuSequenceHandle* seq, const int32_t* tokens,
                                                  uint32_t admit_count) {
	const int32_t next_token = tokens[admit_count];
	std::vector<int8_t> embed_codes(model->hidden_size);
	superslm::CarriedScale embed_scale{};
	const superslm::SslmForwardStatus est = superslm::EmbedEntry(
	    next_token, model->vocab_size, model->embed_weights.data(),
	    static_cast<size_t>(model->hidden_size), model->embed_site_constant, embed_codes.data(),
	    &embed_scale);
	if (est != superslm::SslmForwardStatus::Ok) return;  // unreachable per this function's own
	                                                        // header comment; defensive only.
	std::copy(embed_codes.begin(), embed_codes.end(), seq->hidden_codes.begin());
	seq->hidden_scale = embed_scale;
	seq->layer_index = 0;
}

#ifdef SUPERSLM_T2169_CHUNK_RECORDING_FAULT_INJECTION
bool g_prefill_guard_device_removed_query_injection_armed = false;
#endif

// Plan te266 Sec3.6 item 3: whether the device the prefill chunk ran on is reported removed,
// asked at the moment the prompt twin classifies a guard refusal. The chunk is recorded and
// submitted on the process-wide `harness::GetDevice()` singleton
// (`SubmitOneSubChunkToFullDepthForG5Bridge`, superslm_gpu.cpp), not on
// `SslmGpuContext::device`, which serves only the residency uploads at map/create time (see
// `sslm_decode_step_batch_gpuImpl`'s comment) -- so the singleton is the device asked. A
// successful readback does not by itself prove the device alive, which is why the query is made
// at all rather than inferred from the finish having reached its normal path.
bool PrefillGuardDeviceReportedRemoved() {
#ifdef SUPERSLM_T2169_CHUNK_RECORDING_FAULT_INJECTION
	if (g_prefill_guard_device_removed_query_injection_armed) {
		g_prefill_guard_device_removed_query_injection_armed = false;  // single-shot
		return true;
	}
#endif
	// 1.8.0 (TE-426): the classifier's removed query, which never reads a device that was not set
	// up as removed (the `!dev.dev ||` term this used to carry). A guard refusal proves the device
	// was set up, so this GetDevice() call makes no setup attempt.
	return superslm_gpu::harness::DeviceReportedRemoved(superslm_gpu::harness::GetDevice());
}

}  // namespace

#ifdef SUPERSLM_T2169_CHUNK_RECORDING_FAULT_INJECTION
namespace superslm_gpu {
// Declared in gpu_port.h beside the T-2169 seams; consumed by `PrefillGuardDeviceReportedRemoved`.
void ArmPrefillGuardDeviceRemovedQueryInjection() {
	g_prefill_guard_device_removed_query_injection_armed = true;
}
}  // namespace superslm_gpu
#endif

// T-2184 remedy C1 (Brunel fix round 1, D-SLM3662): a bench-only entry point that reproduces the
// GENUINELY shipped (pre-T-2169, `e35edc1`) per-token prefill composition -- `sslm_gpu_seq_embed_
// token` followed by `DriveGpuSeqToFullDepthForG5Bridge`, one submit-and-fence round-trip per
// LAYER (28 for Qwen2.5-1.5B, `dispatch_budget=24`), issued once per token in a plain host loop.
// Body copied verbatim from `e35edc1:src/gpu/gpu_1p0.cpp`'s own
// `SslmGpuSeqPrefillPromptForG5Bridge` (including its own per-iteration busy check, `if (seq->
// state != Idle) return SSLM_BUSY;` inside the loop, and its own `if (count > 0) seq->ready_for_
// logits = true;` tail) -- not reconstructed from a description of it.
//
// WHY THIS EXISTS. The T-2184 review's own C1 finding: after Rungs 3/4 rewrote `SslmGpuSeqPrefill
// PromptForG5Bridge`'s body to the pre-scan-then-batch composition, `tools/t2180_rung6_tokps.cpp`'s
// own "shipped-per-token" bench arm -- N separate single-token calls into that SAME public bridge
// function -- silently became the batched primitive at `chunk_len=1`, not the genuinely shipped
// 28-round-trip-per-token path. `DriveGpuSeqToFullDepthForG5Bridge` (this file, above) still exists
// and is still exactly the shipped shape, but its only remaining production caller is
// `SslmGpuSeqDecodeStepForG5Bridge` -- neither prefill twin can reach it any more. This function is
// the one new caller that can, for bench purposes only.
//
// NOT part of the shipped 1.0 API surface (matching `gpu_1p0_bench_bridge.h`'s own "not part of
// the shipped GPU API surface, and not installed alongside it" convention) -- declared in that
// header, `tools/t2180_rung6_tokps.cpp`'s own intended includer alongside its existing bench
// accessors.
//
// T-2185 remedy N6/observation (Brunel fix round 2, D-SLM3677): this definition, and its own
// declaration in `gpu_1p0_bench_bridge.h`, are now gated behind
// `SUPERSLM_ENABLE_GPU_BENCH_PRE_BATCHING` -- previously defined unconditionally at file scope,
// so the symbol linked into every binary that compiled this translation unit, not only the one
// bench tool that calls it. Gated per this codebase's own established injection-seam convention
// (`SUPERSLM_O11_ALLOC_INJECTION`, `SUPERSLM_T2169_CHUNK_RECORDING_FAULT_INJECTION`, gpu_port.h):
// `build.bat`'s own `tools\t2180_rung6_tokps.cpp` compile line is the only place that defines the
// macro; every other target -- the red-suite cells, every other tool -- links `gpu_1p0.cpp`
// without it, so this symbol is absent from those binaries entirely (verified: `dumpbin
// /symbols` against a build of this file without the macro defined shows no
// `SslmGpuSeqPrefillPromptPreBatchingBenchOnly` entry).
#ifdef SUPERSLM_ENABLE_GPU_BENCH_PRE_BATCHING
SslmGpuStatus SslmGpuSeqPrefillPromptPreBatchingBenchOnly(SslmGpuContext* ctx,
                                                            SslmGpuSequenceHandle* seq,
                                                            const int32_t* tokens, int32_t count,
                                                            uint32_t dispatch_budget) {
	superslm_gpu::inflight::GpuApiCallScope scope(__func__);  // G3 (§4.4): every C ABI entry point
	if (!ctx || !seq || !seq->model || seq->ctx != ctx || (count > 0 && !tokens) || count < 0 ||
	    dispatch_budget < 1) {
		return SSLM_SEQUENCE_KV_BUFFER_MISMATCH;
	}
	// EntrySubmitGuard (§4.4): reaches the singleton; scope by holder; before the loop.
	if (const SslmGpuStatus g = EntrySubmitGuard(ScopeForHolder(seq), GuardDevice::kReachesSingleton);
	    g != SSLM_OK) {
		return g;
	}
	for (int32_t i = 0; i < count; ++i) {
		if (seq->state() != superslm_gpu::SslmSequenceGpuState::Idle) return SSLM_BUSY;
		const SslmGpuStatus st = sslm_gpu_seq_embed_token(ctx, seq, tokens[i]);
		if (st != SSLM_OK) return st;
		const SslmGpuStatus driven = DriveGpuSeqToFullDepthForG5Bridge(ctx, seq, dispatch_budget);
		if (driven != SSLM_OK) return driven;
	}
	if (count > 0) seq->ready_for_logits = true;
	return SSLM_OK;
}
#endif  // SUPERSLM_ENABLE_GPU_BENCH_PRE_BATCHING

// G5-5 session 3 fix (T-2132, Brunel, Claude/Brunel/t2132-g5-build-2026-08-16.md session 3): the
// GPU-1.0 twin of `sslm_prefill(..., SSLM_SPAN_PROMPT, ...)` -- embeds and drives EVERY token in
// `tokens` (including the last) to full depth, exactly `PrefillWholeTokensImpl`'s own
// SSLM_SPAN_PROMPT branch (no walk-state touch, no masking -- design Sec5: "never advances the
// DFA walk, whatever schema is bound"). On success, sets `seq->ready_for_logits = true`,
// mirroring `sslm_prefill`'s own unconditional set on any successful prefill
// (src/sslm_abi.cpp:1607) -- this is the fix: a caller that then calls
// `SslmGpuSeqDecodeStepForG5Bridge` (below) for its first post-prompt decode step gets the SAME
// "reuse the already-computed final residual, do not re-embed" shortcut the CPU ABI already gives
// every real caller, closing the exact duplicate-KV-commit gap session 3 diagnosed in this
// bridge's own prior caller (`tools/t2132_g5_gpu_parity_gpu.cpp`'s hand-rolled `PrefillPrompt`).
SslmGpuStatus SslmGpuSeqPrefillPromptForG5BridgeImpl(SslmGpuContext* ctx, SslmGpuSequenceHandle* seq,
                                                  const int32_t* tokens, int32_t count,
                                                  uint32_t dispatch_budget) {
	if (!ctx || !seq || !seq->model || seq->ctx != ctx || (count > 0 && !tokens) || count < 0 ||
	    dispatch_budget < 1) {
		return SSLM_SEQUENCE_KV_BUFFER_MISMATCH;
	}
	if (seq->state() != superslm_gpu::SslmSequenceGpuState::Idle) return SSLM_BUSY;
	// EntrySubmitGuard (§4.4): reaches the singleton; scope by holder. After the own-handle Busy
	// refusal, before the first state change (`bind_eligible`).
	if (const SslmGpuStatus g = EntrySubmitGuard(ScopeForHolder(seq), GuardDevice::kReachesSingleton);
	    g != SSLM_OK) {
		return g;
	}
	seq->bind_eligible = false; // D-SLM7625 generation entry: gpu prompt prefill
	if (count == 0) return SSLM_OK;  // matches the shipped per-token loop's own vacuous-loop exit

	// T-2169 (Rung 4, design Sec5/Sec8): the pre-scan-then-batch composition -- host-side
	// sequential admission, then ONE call into the chunk-submission primitive for exactly
	// `admit_count` tokens, replacing the per-token DriveGpuSeqToFullDepthForG5Bridge loop above.
	// No DFA pre-scan (the prompt twin never advances or checks the DFA walk, design Sec5).
	SslmGpuModelHandle* model = seq->model;
	seq->live_state.hidden_scale = seq->hidden_scale;
	seq->live_state.layer_index = seq->layer_index;
	seq->live_state.kv_saturation_count = seq->kv_saturation_count;
	seq->live_state.context_length = seq->context_length;
	const int64_t chunk_open_ctxlen = seq->context_length;

	// Prefill snapshot writer: invalid from here until this call's kNone exit, so every exit
	// after the pre-scan other than kNone -- including an exception unwinding a chunk -- leaves
	// the read refusing.
	seq->prefill_hidden_valid = false;
	const ChunkPreScanResult scan = RunChunkAdmissionPreScan(
	    model, chunk_open_ctxlen, seq->context_cap, tokens, count, /*dfa_entry=*/nullptr,
	    /*dfa_walk_state_start=*/0, seq->pool != nullptr ? seq->limit : seq->context_cap);

	// Paged KV (G3, §3.6): the pages the admitted rows land in, before any state moves.
	if (scan.admit_count > 0) {
		const SslmGpuStatus map_st = MapPooledPagesThrough(seq, chunk_open_ctxlen + scan.admit_count);
		if (map_st != SSLM_OK) return map_st;
	}
	int64_t derived_count = 0;
	bool guard_rejected = false;
	superslm::SslmForwardStatus chunk_fault = superslm::SslmForwardStatus::Ok;
	SubmitAdmittedChunkForG5Bridge(model, seq, scan.chunk_embedding_bytes.data(), scan.admit_count,
	                                chunk_open_ctxlen, &derived_count, &guard_rejected, &chunk_fault);

	// D-SLM3622 (design Sec5 step 5): the device-computed fallback -- a rejection the pre-scan
	// could not see, discovered only via the post-chunk readback -- overrides whatever the
	// host-computable cause predicted, regardless of cause. ready_for_logits stays untouched.
	// A guard refusal always lands here: the refused token's last-layer commit never runs, so the
	// derived count is short.
	//
	// D-SLM7282 (plan te266 Sec3.6): a device-side domain guard refusal on a device not reported
	// removed is a per-sequence refusal, SSLM_SEQUENCE_REJECTED, the class the decode path already
	// reports for the identical sticky-tag fact (MapDecodedStatusToGpuStatus).
	//
	// 1.8.0 (TE-426): an allocation failure on a live device with the submission clean
	// (`chunk_fault` GpuAllocationFailed) is SSLM_GPU_ALLOCATION_FAILED. Every other short count --
	// a non-allocation fault, a stranded submission, a removed device -- stays SSLM_DEVICE_LOST.
	if (derived_count < static_cast<int64_t>(scan.admit_count)) {
		if (guard_rejected && !PrefillGuardDeviceReportedRemoved()) {
			return SSLM_SEQUENCE_REJECTED;
		}
		if (chunk_fault == superslm::SslmForwardStatus::GpuAllocationFailed) {
			return SSLM_GPU_ALLOCATION_FAILED;
		}
		return SSLM_DEVICE_LOST;
	}

	switch (scan.cause) {
		case ChunkAdmissionCause::kEmbed:
			// ready_for_logits left untouched -- P2's own disposition (design Sec5's cause table).
			return scan.embed_reject_status;
		case ChunkAdmissionCause::kPositionCap:
			// D-SLM3619: refuses the whole call with the shipped path's exact observables --
			// ready_for_logits left untouched, and the next-rejected token's own incidental
			// host-side embed side effect is reproduced (never a dispatch for it).
			ReproducePositionCapBoundaryEmbedSideEffect(model, seq, tokens, scan.admit_count);
			return SSLM_DEVICE_LOST;
		case ChunkAdmissionCause::kBudget:
			// Paged KV (G3, §3.6, the CPU twin's rule): the admitted tokens landed and the call is
			// refused SSLM_GPU_KV_BUDGET_EXCEEDED; ready_for_logits is set iff any landed, so the
			// holder's one ready token (1 + (limit - c), §3.4) can be taken. No embed side effect:
			// nothing past the limit is embedded.
			if (scan.admit_count > 0) seq->ready_for_logits = true;
			return SSLM_GPU_KV_BUDGET_EXCEEDED;
		case ChunkAdmissionCause::kDfa:
			// Unreachable for the prompt twin (no DFA pre-scan is run for it) -- defensive only.
			return SSLM_SEQUENCE_REJECTED;
		case ChunkAdmissionCause::kNone:
		default:
			seq->ready_for_logits = true;  // count > 0 already established above -- P4's own row.
			// Prefill snapshot writer: this call's completed last-position residual.
			seq->prefill_hidden_codes = seq->hidden_codes;
			seq->prefill_hidden_valid = true;
			return SSLM_OK;
	}
}

// G5-5 session 3 fix (T-2132, Brunel): the recommended one-call-per-step entry point -- the GPU-
// 1.0 twin of `sslm_decode_step`'s own composition (embed-if-needed, layer-loop-to-depth, finish),
// including its `ready_for_logits` shortcut (src/sslm_abi.cpp:1708-1715) verbatim: when a prior
// `SslmGpuSeqPrefillPromptForG5Bridge`/`SslmGpuSeqPrefillSchemaContentForG5Bridge` call left this
// flag set, `token_to_embed_if_needed` is IGNORED and this call jumps straight to finishing the
// already-computed residual -- no embed, no layer loop, matching the CPU ABI's own "this is the
// ONE call that costs no layer work" comment exactly. Otherwise embeds `token_to_embed_if_needed`
// and drives it to full depth first. Either way, finishes via `SslmGpuSeqFinishTokenForG5Bridge`
// (masking if a schema is bound, walk-state advance, plain argmax otherwise). This is the shape a
// future bridge consumer should reach for FIRST -- the granular
// embed/`sslm_decode_step_gpu`/`sslm_gpu_ready`/finish primitives stay available (this bridge/
// `gpu_1p0.h` do not remove them) for a caller that genuinely needs to interleave other GPU work
// between layers, but composing them by hand is exactly what re-trips this session's own bug.
SslmGpuStatus SslmGpuSeqDecodeStepForG5BridgeImpl(SslmGpuContext* ctx, SslmGpuSequenceHandle* seq,
                                               int32_t token_to_embed_if_needed,
                                               uint32_t dispatch_budget, int32_t* out_token) {
	if (!out_token) return SSLM_SEQUENCE_KV_BUFFER_MISMATCH;
	*out_token = -1;
	if (!ctx || !seq || !seq->model || seq->ctx != ctx || dispatch_budget < 1) {
		return SSLM_SEQUENCE_KV_BUFFER_MISMATCH;
	}
	// The same-sequence claim (strike B1), held across the whole call -- the state read, the guard,
	// the ready flag, the embed, the drive and the finish -- so a second thread's decode or finish of
	// this handle returns SSLM_BUSY having changed nothing. The drive's sslm_decode_step_gpu calls and
	// the finish re-enter it on this thread.
	const SeqDecodeClaim claim(seq);
	if (!claim.held()) return SSLM_BUSY;
	if (seq->state() == superslm_gpu::SslmSequenceGpuState::Submitted) return SSLM_BUSY;
	// EntrySubmitGuard (§4.4): the bridge decode reaches both devices (the drive on the singleton,
	// the finish's device logits on the context's own), so the ledger term applies; scope by
	// holder. After the own-handle Busy refusal, before `bind_eligible`. The inner verbs it calls
	// keep their own guards, which then pass.
	if (const SslmGpuStatus g = EntrySubmitGuard(ScopeForHolder(seq), GuardDevice::kReachesSingleton);
	    g != SSLM_OK) {
		return g;
	}
	seq->bind_eligible = false;
	if (seq->ready_for_logits) {
		// This sequence's hidden_codes already hold a fully-computed final hidden state (from a
		// prior prefill call) -- no embed, no layer loop this call; jump straight to finishing,
		// exactly sslm_decode_step's own ready_for_logits branch (src/sslm_abi.cpp:1708-1715).
		seq->ready_for_logits = false;
	} else {
		if (seq->state() != superslm_gpu::SslmSequenceGpuState::Idle) return SSLM_BUSY;
		const SslmGpuStatus st = sslm_gpu_seq_embed_token(ctx, seq, token_to_embed_if_needed);
		if (st != SSLM_OK) return st;
		const SslmGpuStatus driven = DriveGpuSeqToFullDepthForG5Bridge(ctx, seq, dispatch_budget);
		if (driven != SSLM_OK) return driven;  // 1.8.0 (TE-426): the drive's own status, kept
	}
	return SslmGpuSeqFinishTokenForG5Bridge(ctx, seq, out_token);
}

SslmGpuStatus SslmGpuSeqPrefillSchemaContentForG5BridgeImpl(SslmGpuContext* ctx,
                                                          SslmGpuSequenceHandle* seq,
                                                          const int32_t* tokens, int32_t count,
                                                          uint32_t dispatch_budget_per_token,
                                                          int32_t* consumed) {
	if (!consumed) return SSLM_SEQUENCE_KV_BUFFER_MISMATCH;
	*consumed = 0;
	if (!ctx || !seq || !seq->model || seq->ctx != ctx || (count > 0 && !tokens) || count < 0 ||
	    dispatch_budget_per_token < 1) {
		return SSLM_SEQUENCE_KV_BUFFER_MISMATCH;
	}
	if (seq->state() == superslm_gpu::SslmSequenceGpuState::Submitted) return SSLM_BUSY;
	// EntrySubmitGuard (§4.4): reaches the singleton; scope by holder. After the own-handle Busy
	// refusal, before the first state change (`bind_eligible`).
	if (const SslmGpuStatus g = EntrySubmitGuard(ScopeForHolder(seq), GuardDevice::kReachesSingleton);
	    g != SSLM_OK) {
		return g;
	}
	seq->bind_eligible = false;
	if (seq->bound_schema_index < 0) {
		return SSLM_SEQUENCE_REJECTED;  // the GPU twin of SSLM_SCHEMA_SPAN_UNBOUND.
	}
	SslmGpuModelHandle* model = seq->model;
	const superslm::SchemaEntry* entry =
	    model->schemas.ByIndex(static_cast<size_t>(seq->bound_schema_index));
	if (!entry) return SSLM_SEQUENCE_REJECTED;
	if (count == 0) return SSLM_OK;  // matches the shipped per-token loop's own vacuous-loop exit.
	// T-2184 remedy M1 (Brunel fix round 1, D-SLM3662): restores the shipped per-token loop's own
	// ordering -- `Transition(tokens[0])` first, `SSLM_SEQUENCE_REJECTED` on failure, THEN the busy
	// check (`e35edc1`'s own per-token body). The batched composition below had the busy check run
	// FIRST, so a sequence left `Submitted` (reachable single-threaded through
	// `sslm_decode_step_gpu`, `fixture_common.h:174`'s own documented idiom) whose first token is
	// DFA-unreachable returned `SSLM_BUSY` here instead of the shipped `SSLM_SEQUENCE_REJECTED`.
	// `*consumed` stays 0 on this path -- no earlier admitted token exists in THIS call to set
	// `ready_for_logits`, matching the shipped loop's own first-iteration disposition exactly.
	{
		uint32_t reachability_probe_state = 0;
		if (!model->schemas.Transition(*entry, seq->dfa_walk_state, static_cast<uint32_t>(tokens[0]),
		                                &reachability_probe_state)) {
			return SSLM_SEQUENCE_REJECTED;
		}
	}
	// T-2169 (Rung 3, design Sec5/Sec8): the pre-scan-then-batch composition -- host-side
	// sequential admission (DFA reachability, embed validity, position cap, in that priority),
	// then ONE call into the chunk-submission primitive for exactly `admit_count` tokens,
	// replacing the per-token Transition/embed_token/DriveGpuSeqToFullDepthForG5Bridge loop above.
	// design Sec14.3/D-SLM3478's own partial-consumption contract is preserved: tokens admitted
	// before the cause that stops the call already ran their forward pass for real and *consumed
	// already counts them -- what changes is HOW they are dispatched (one chunk call), never
	// which tokens land.
	seq->live_state.hidden_scale = seq->hidden_scale;
	seq->live_state.layer_index = seq->layer_index;
	seq->live_state.kv_saturation_count = seq->kv_saturation_count;
	seq->live_state.context_length = seq->context_length;
	const int64_t chunk_open_ctxlen = seq->context_length;

	// Prefill snapshot writer: invalid from here until this call's kNone exit (the prompt twin's
	// identical rule). The kDfa exit sets ready_for_logits but never the snapshot.
	seq->prefill_hidden_valid = false;
	const ChunkPreScanResult scan = RunChunkAdmissionPreScan(
	    model, chunk_open_ctxlen, seq->context_cap, tokens, count, entry, seq->dfa_walk_state,
	    seq->pool != nullptr ? seq->limit : seq->context_cap);

	// Paged KV (G3, §3.6): the pages the admitted rows land in, before any state moves.
	if (scan.admit_count > 0) {
		const SslmGpuStatus map_st = MapPooledPagesThrough(seq, chunk_open_ctxlen + scan.admit_count);
		if (map_st != SSLM_OK) return map_st;
	}
	int64_t derived_count = 0;
	// The schema twin does not classify its device override (plan te266 Sec3.3, TE266-Q5): a guard
	// refusal here stays SSLM_DEVICE_LOST below, because this twin's SSLM_SEQUENCE_REJECTED already
	// means "tokens landed, carry on decoding" (kDfa). The flag is therefore not read.
	bool guard_rejected_unused = false;
	superslm::SslmForwardStatus chunk_fault = superslm::SslmForwardStatus::Ok;
	SubmitAdmittedChunkForG5Bridge(model, seq, scan.chunk_embedding_bytes.data(), scan.admit_count,
	                                chunk_open_ctxlen, &derived_count, &guard_rejected_unused,
	                                &chunk_fault);

	// D-SLM3622 (design Sec5 step 5): the device-computed fallback overrides whatever the
	// host-computable cause predicted, regardless of cause -- ready_for_logits stays untouched,
	// and *consumed reflects the device's own actually-committed count, not admit_count.
	const bool overridden = derived_count < static_cast<int64_t>(scan.admit_count);
	*consumed = static_cast<int32_t>(overridden ? derived_count
	                                             : static_cast<int64_t>(scan.admit_count));

	// The DFA walk state advances exactly as far as the tokens that actually committed
	// (*consumed, whether that is admit_count or the device-derived shorter count) -- nothing
	// past the actually-committed prefix legitimately advanced the schema's own state either,
	// matching design Sec14.3's "the rejected token's own effects never land" extended to the
	// device-computed fallback.
	if (*consumed > 0) {
		seq->dfa_walk_state = scan.dfa_states_after[static_cast<size_t>(*consumed) - 1];
	}

	if (overridden) {
		// 1.8.0 (TE-426): the prompt twin's rule -- an allocation failure on a live device with the
		// submission clean is SSLM_GPU_ALLOCATION_FAILED; every other override SSLM_DEVICE_LOST.
		// `*consumed` still counts only the committed tokens.
		if (chunk_fault == superslm::SslmForwardStatus::GpuAllocationFailed) {
			return SSLM_GPU_ALLOCATION_FAILED;
		}
		return SSLM_DEVICE_LOST;
	}

	switch (scan.cause) {
		case ChunkAdmissionCause::kDfa:
			// S1's own disposition (design Sec5's cause table): ready_for_logits SET, iff
			// *consumed > 0 -- unlike EMBED/POSITION_CAP, this row is not "left untouched".
			if (*consumed > 0) seq->ready_for_logits = true;
			return SSLM_SEQUENCE_REJECTED;
		case ChunkAdmissionCause::kEmbed:
			// S3's own disposition: ready_for_logits left untouched.
			return scan.embed_reject_status;
		case ChunkAdmissionCause::kBudget:
			// Paged KV (G3, §3.6): the prompt twin's kBudget row -- `*consumed` counts the landed
			// tokens, the walk advanced over them, ready_for_logits set iff any landed.
			if (*consumed > 0) seq->ready_for_logits = true;
			return SSLM_GPU_KV_BUDGET_EXCEEDED;
		case ChunkAdmissionCause::kPositionCap:
			// D-SLM3619: refuses the whole call with the shipped path's exact observables --
			// S4(a)'s own disposition, ready_for_logits left untouched, and the next-rejected
			// token's own incidental host-side embed side effect is reproduced.
			ReproducePositionCapBoundaryEmbedSideEffect(model, seq, tokens, scan.admit_count);
			return SSLM_DEVICE_LOST;
		case ChunkAdmissionCause::kNone:
		default:
			// S5's own disposition: ready_for_logits set iff *consumed > 0 -- always true here
			// since admit_count == count > 0 on the all-admitted path.
			if (*consumed > 0) seq->ready_for_logits = true;
			// Prefill snapshot writer: this call's completed last-position residual.
			seq->prefill_hidden_codes = seq->hidden_codes;
			seq->prefill_hidden_valid = true;
			return SSLM_OK;
	}
}

// gpu_1p0.h: the model's hidden width, the element count the read below writes.
SslmGpuStatus sslm_gpu_model_hidden_sizeImpl(const SslmGpuModelHandle* model,
                                             uint32_t* out_hidden_size) {
	if (!model || !out_hidden_size) {
		return SSLM_SEQUENCE_KV_BUFFER_MISMATCH;
	}
	*out_hidden_size = model->hidden_size;
	return SSLM_OK;
}

// gpu_1p0.h: the post-final_norm hidden state of this sequence's most recent successful prefill,
// read from the prefill snapshot (SslmGpuSequenceHandle::prefill_hidden_codes), never from the
// live residual. Host-side: no command list, no dispatch, and nothing on the sequence is written.
// The checks run in the header's stated order; each refusal before step 4 writes nothing at all.
SslmGpuStatus sslm_gpu_seq_read_prefill_final_hiddenImpl(SslmGpuContext* ctx,
                                                         SslmGpuSequenceHandle* seq,
                                                         int8_t* out_codes, size_t out_capacity,
                                                         size_t* out_required, int64_t* out_scale_m,
                                                         int64_t* out_scale_e) {
	// 1. Malformed handle or output pointer.
	if (!ctx || !seq || !seq->model || seq->ctx != ctx || !out_required || !out_scale_m ||
	    !out_scale_e || (out_codes == nullptr && out_capacity != 0)) {
		return SSLM_SEQUENCE_KV_BUFFER_MISMATCH;
	}
	// 2. Submitted.
	if (seq->state() == superslm_gpu::SslmSequenceGpuState::Submitted) {
		return SSLM_BUSY;
	}
	// 3. No snapshot.
	if (!seq->prefill_hidden_valid) {
		return SSLM_PREFILL_HIDDEN_UNAVAILABLE;
	}
	// 4. Required size, written on every outcome from here on.
	const SslmGpuModelHandle* model = seq->model;
	const size_t hidden = model->hidden_size;
	*out_required = hidden;
	if (out_capacity < hidden) {
		return SSLM_OUTPUT_BUFFER_TOO_SMALL;
	}
	// 5. final_norm over the snapshot: the RmsNormSite call SslmGpuSeqFinishTokenForG5BridgeImpl
	// makes before logits, with the same gain and site constant. The incoming scale is
	// CarriedScale{} because RmsNormSite does not read it (forward_sites.h), which is why the
	// snapshot stores none. Computed into local storage first, so a guard refusal writes nothing.
	std::vector<int8_t> final_codes(hidden);
	superslm::CarriedScale final_scale{};
	const superslm::SslmForwardStatus st = superslm::RmsNormSite(
	    seq->prefill_hidden_codes.data(), model->final_norm_gain.data(), hidden,
	    superslm::CarriedScale{}, model->final_norm_site_constant, final_codes.data(), &final_scale,
	    "final_norm", /*token_index=*/0, /*trace_hook_state=*/nullptr,
	    /*external_wide_scratch=*/nullptr);
	if (st != superslm::SslmForwardStatus::Ok) {
		return SSLM_SEQUENCE_REJECTED;
	}
	// 6. Success.
	std::memcpy(out_codes, final_codes.data(), hidden);
	*out_scale_m = final_scale.m;
	*out_scale_e = final_scale.e;
	return SSLM_OK;
}

// =================================================================================================
// Paged-KV plan (rev 16.2) §4 items 7 and 8, step G4: GPU prefixes, begin_from, adopt and the pool
// upload. Every function of this step is in this block (the wrappers are in the boundary block below).
// =================================================================================================

// §4 item 7: a GPU prefix is a budgeted holder over a pool, built by driving the existing prompt and
// schema-content prefill bridges on its internal pooled sequence (`seq`, adapter unbound: prefixes are
// base-only, as on CPU). The internal sequence is never handed out; it carries the prefix's page holder,
// table buffer, budget, origin and limit, and the forward state the CPU prefix keeps in its own fields
// (hidden codes and scale, layer index, context length, the five saturation counts, the schema binding
// and walk state, the ready flag and the prefill snapshot). It counts as one of the model's live
// sequences, one of the context's live handles and one of the pool's live references, so the model,
// the pool and the context refuse to go away under a live prefix, exactly as under a live sequence.
// `frozen` is the CPU prefix's flag: set by the first freeze, never cleared.
struct SslmGpuPrefixHandle {
	SslmGpuContext* ctx = nullptr;
	SslmGpuModelHandle* model = nullptr;
	SslmGpuKvPoolHandle* pool = nullptr;
	SslmGpuSequenceHandle* seq = nullptr;
	bool frozen = false;
	// The last token prefilled (the CPU prefix's current_token): -1 while empty; copied by begin_from,
	// written by prefill, read by save, set by restore.
	int32_t current_token = -1;

	SslmGpuPrefixHandle() = default;
	SslmGpuPrefixHandle(const SslmGpuPrefixHandle&) = delete;
	SslmGpuPrefixHandle& operator=(const SslmGpuPrefixHandle&) = delete;
};

namespace superslm_gpu {
// §4 item 8's pool upload regions: `bytes` bytes of pool page `page` at byte `offset` within it, taken
// from the pool's host mirror. The plan names the call, not the region type; this is the shape the GPU
// red suite declares (g1d_cell_3_6e.cpp, unless G1D_POOL_UPLOAD_DECLARED). No public header declares
// either: the pool upload is internal (§4.4's derived table, "new at G4").
struct PoolUploadRegion {
	uint32_t page = 0;
	uint64_t offset = 0;
	uint64_t bytes = 0;
};
void UploadPoolRegionsSync(harness::Device& dev, SslmGpuKvPoolHandle* pool, const std::vector<PoolUploadRegion>& regions);
}  // namespace superslm_gpu

namespace {

// The pool upload's phase A (§4 item 8), the zero-fill's split: the UPLOAD staging buffer holding each
// region's bytes back to back, copied from `sources[i]` (regions[i].bytes bytes each). Records and resets
// nothing, so a throw leaves the singleton's list Closed. Throws as Device::Upload does (SSLM_GPU_HR);
// the caller classifies.
Microsoft::WRL::ComPtr<ID3D12Resource> AllocatePoolUploadStaging(
    superslm_gpu::harness::Device& dev, const std::vector<superslm_gpu::PoolUploadRegion>& regions,
    const std::vector<const uint8_t*>& sources) {
	if (!dev.available.load()) throw std::runtime_error("pool upload: the process's D3D12 device is not available");
	uint64_t total = 0;
	for (const superslm_gpu::PoolUploadRegion& r : regions) total += r.bytes;
	Microsoft::WRL::ComPtr<ID3D12Resource> staging =
	    dev.MakeBuffer(total, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_GENERIC_READ);
	void* mapped = nullptr;
	D3D12_RANGE none{0, 0};
	SSLM_GPU_HR(staging->Map(0, &none, &mapped));
	uint64_t at = 0;
	for (size_t i = 0; i < regions.size(); ++i) {
		std::memcpy(static_cast<uint8_t*>(mapped) + at, sources[i], static_cast<size_t>(regions[i].bytes));
		at += regions[i].bytes;
	}
	staging->Unmap(0, nullptr);
	return staging;
}

// The pool upload's phase B (§4 item 8): the eighth reset site, on the process's device. Its rule:
//   - it runs only after the verb's entry guard (scope "any sequence"), so no command list submitted
//     through the C ABI references the pool buffer while it runs;
//   - it reads the pool latch: set (the last readback left the buffer in COPY_SOURCE) records
//     COPY_SOURCE -> COPY_DEST, otherwise UNORDERED_ACCESS -> COPY_DEST; it copies each region from the
//     staging buffer, records COPY_DEST -> UNORDERED_ACCESS, executes, waits, and clears the latch, so
//     the buffer is back in the state the next pooled dispatch expects;
//   - it uploads only the regions it is given (the pages the verb wrote).
// INVARIANT (as the zero-fill's): nothing between the list's Reset and CloseListOrThrow allocates or
// throws. The latch is cleared only once the wait has returned; a throw leaves it as it was.
void SubmitPoolUploadSync(superslm_gpu::harness::Device& dev, SslmGpuKvPoolHandle* pool,
                          const std::vector<superslm_gpu::PoolUploadRegion>& regions,
                          const Microsoft::WRL::ComPtr<ID3D12Resource>& staging) {
	// The ledger's reset (G3, plan §4.4): the eighth reset site (§4 item 8). BeginRecording() runs the
	// stranded check, the boundary guard check and PoolSubmitAssert before it.
	dev.BeginRecording();
	// ---- no allocation and no throw from here to CloseListOrThrow ----
	superslm_gpu::harness::RecordKvTransitionFromLatch(dev.list.Get(), pool->pool_buf.Get(), pool->kv_needs_resume_barrier,
	                                                  D3D12_RESOURCE_STATE_COPY_DEST);
	const uint64_t page_bytes = pool->geo.page_bytes;
	uint64_t at = 0;
	for (const superslm_gpu::PoolUploadRegion& r : regions) {
		superslm_gpu::harness::RecordCopyBufferRegion(dev.list.Get(), pool->pool_buf.Get(),
		                                              uint64_t{r.page} * page_bytes + r.offset, staging.Get(), at, r.bytes);
		at += r.bytes;
	}
	superslm_gpu::harness::RecordTransition(dev.list.Get(), pool->pool_buf.Get(), D3D12_RESOURCE_STATE_COPY_DEST,
	                                        D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
	dev.CloseListOrThrow();
	// ---- end of the recording window ----
	// The ledger's submission and wait (G3), as the zero-fill's: Submit(no retry), then its Wait(); the
	// staging buffer and the pool buffer are kept alive if it strands (review-g23 F1, as the other
	// synchronous sites). Every pool upload -- the prefix tail copy, adopt's tail copy, and
	// restore_into_pool through UploadPoolRegionsSync -- submits here.
	dev.SubmitAndWaitKeepingAlive(/*retry=*/false, staging, pool->pool_buf);
	pool->kv_needs_resume_barrier = false;
}

// Throws std::out_of_range for a region outside the pool (a caller's defect: every library caller
// passes a page its own table maps, whole).
void CheckPoolUploadRegions(const SslmGpuKvPoolHandle* pool, const std::vector<superslm_gpu::PoolUploadRegion>& regions) {
	if (pool == nullptr || pool->pool_buf == nullptr) throw std::out_of_range("pool upload: no pool");
	const uint64_t page_bytes = pool->geo.page_bytes;
	for (const superslm_gpu::PoolUploadRegion& r : regions) {
		if (r.page >= pool->page_count || r.bytes == 0 || r.offset > page_bytes || r.bytes > page_bytes - r.offset) {
			throw std::out_of_range("pool upload: a region outside the pool");
		}
	}
}

// A pooled sequence handle of `budget` over `pool`, R(budget) pages (or `reserve` when nonzero) drawn
// into its reserve and its table buffer allocated, NOT yet published: no count moved, so a caller that
// fails after this releases `(*out)->holder` and lets the handle go. The budgeted create's body
// (sslm_gpu_seq_create_budgetedImpl), for the prefix verbs. Refusals: SSLM_GPU_KV_POOL_EXHAUSTED with
// nothing drawn; a table-buffer failure returns the draw and is classified as every handle create's.
//
// G5: `extra_pages` more pages in the same draw (restore-into-pool's E materialized pages, §3.7), so the
// draw of R(budget) + E is refused whole, with nothing drawn.
// Precached prefixes: with `reserve_exact`, the draw is exactly `extra_pages` (the prefix restore's
// ceil(L/B) - s), and `budget` only sets the handle's budget.
SslmGpuStatus MakePooledSeqUnpublished(SslmGpuContext* ctx, SslmGpuModelHandle* model, SslmGpuKvPoolHandle* pool,
                                       int32_t budget, std::unique_ptr<SslmGpuSequenceHandle>* out,
                                       uint32_t extra_pages = 0, bool reserve_exact = false) {
	const uint32_t reserve =
	    reserve_exact ? extra_pages
	                  : superslm_gpu::kv_pool::ReservationPages(budget, pool->geo, model->context_cap) + extra_pages;
	std::unique_ptr<SslmGpuSequenceHandle> h(new SslmGpuSequenceHandle());
	h->hidden_codes.assign(model->hidden_size, 0);
	h->prefill_hidden_codes.assign(model->hidden_size, 0);
	superslm::kv_pages::PageHolder* holder = nullptr;
	const superslm::kv_pages::PageStatus ps =
	    superslm::kv_pages::HolderCreate(pool->pages, pool->geo.cap_pages, reserve, &holder);
	if (ps == superslm::kv_pages::PageStatus::kExhausted) return SSLM_GPU_KV_POOL_EXHAUSTED;
	if (ps == superslm::kv_pages::PageStatus::kAllocationFailed) return SSLM_GPU_ALLOCATION_FAILED;
	if (ps != superslm::kv_pages::PageStatus::kOk) return SSLM_SEQUENCE_KV_BUFFER_MISMATCH;
	try {
		h->table_buf = ctx->device.MakeBuffer(static_cast<UINT64>(pool->geo.cap_pages) * sizeof(uint32_t),
		                                      D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_FLAG_NONE,
		                                      D3D12_RESOURCE_STATE_COMMON);
		superslm_gpu::harness::NameBuffer(h->table_buf.Get(), L"SuperSLM K/V page table");
	} catch (...) {
		superslm::kv_pages::HolderRelease(holder);
		return ClassifyHandleFault(&ctx->device);
	}
	h->ctx = ctx;
	h->model = model;
	h->pool = pool;
	h->holder = holder;
	h->table_dirty = true;
	h->budget = budget;
	h->origin = 0;
	h->limit = superslm_gpu::kv_pool::HolderLimit(0, budget, model->context_cap);
	h->context_cap = model->context_cap;
	h->kv_bytes = 0;
	h->kv_needs_resume_barrier = false;
	h->prefill_hidden_valid = false;
	h->hidden_scale = superslm::CarriedScale{};
	h->layer_index = 0;
	h->kv_saturation_count = 0;
	h->context_length = 0;
	h->bind_eligible = true;
	h->live_state.hidden_codes = h->hidden_codes.data();
	*out = std::move(h);
	return SSLM_OK;
}

// Publishes a handle made by MakePooledSeqUnpublished: the counts the budgeted create moves.
SslmGpuSequenceHandle* PublishPooledSeq(std::unique_ptr<SslmGpuSequenceHandle> h) {
	h->pool->live_refs += 1;
	h->model->live_sequences += 1;
	h->ctx->live_handles += 1;
	return h.release();
}

// The forward state a prefix's internal sequence carries into an adopter or a begin_from child, copied
// verbatim (§3.5's begin_from list, and the CPU adopt's field rules applied to the GPU handle's fields):
// hidden codes and scale, layer index, the aggregate saturation count and the four per-site counts,
// context length, the ready flag (gap 4b/4d: the GPU twin of the CPU's `current_token >= 0`) and the
// prefill snapshot. The schema fields, origin, limit and bind permission are the callers'.
void CopyPrefixForwardState(SslmGpuSequenceHandle* dst, const SslmGpuSequenceHandle* src) {
	std::copy(src->hidden_codes.begin(), src->hidden_codes.end(), dst->hidden_codes.begin());
	dst->hidden_scale = src->hidden_scale;
	dst->layer_index = src->layer_index;
	dst->kv_saturation_count = src->kv_saturation_count;
	dst->context_length = src->context_length;
	dst->live_state.hidden_scale = src->hidden_scale;
	dst->live_state.layer_index = src->layer_index;
	dst->live_state.kv_saturation_count = src->kv_saturation_count;
	dst->live_state.context_length = src->context_length;
	dst->live_state.kv_landing_saturation_count = src->live_state.kv_landing_saturation_count;
	dst->live_state.k_channel_landing_saturation_count = src->live_state.k_channel_landing_saturation_count;
	dst->live_state.rope_q_saturation_count = src->live_state.rope_q_saturation_count;
	dst->live_state.rope_k_saturation_count = src->live_state.rope_k_saturation_count;
	dst->ready_for_logits = src->ready_for_logits;
	std::copy(src->prefill_hidden_codes.begin(), src->prefill_hidden_codes.end(), dst->prefill_hidden_codes.begin());
	dst->prefill_hidden_valid = src->prefill_hidden_valid;
}

// The dispatches one token costs at full depth: the unit of sslm_gpu_prefix_prefill's dispatch_budget
// (a call admits floor(dispatch_budget / this) tokens, as a decode call with this budget runs one token
// to full depth; cell 5.8's "a dispatch_budget that stops mid-span").
uint32_t PrefixPrefillDispatchesPerToken(const SslmGpuModelHandle* model) {
	return model->dispatches_per_layer * model->num_hidden_layers;
}

}  // namespace

namespace superslm_gpu {
// §4 item 8: the pool upload. Writes `regions` of the shared pool buffer from the pool's host mirror on
// `dev`, which every library caller passes as harness::GetDevice(): the latch it reads is set by
// submissions on the singleton queue. Phase A (the staging buffer) and phase B (the eighth reset site,
// PoolSubmitAssert in front of it) as above. Throws on any failure; the verb classifies. An empty
// region list submits nothing.
void UploadPoolRegionsSync(harness::Device& dev, SslmGpuKvPoolHandle* pool, const std::vector<PoolUploadRegion>& regions) {
	CheckPoolUploadRegions(pool, regions);
	if (regions.empty()) return;
	std::vector<const uint8_t*> sources(regions.size());
	for (size_t i = 0; i < regions.size(); ++i) {
		sources[i] = pool->host_mirror.data() + size_t{regions[i].page} * pool->geo.page_bytes +
		             static_cast<size_t>(regions[i].offset);
	}
	const Microsoft::WRL::ComPtr<ID3D12Resource> staging = AllocatePoolUploadStaging(dev, regions, sources);
	SubmitPoolUploadSync(dev, pool, regions, staging);
}
}  // namespace superslm_gpu

// §4 item 7: a budgeted prefix over `pool` at origin 0, as the budgeted create: the same refusals in the
// same order (SSLM_SEQUENCE_KV_BUFFER_MISMATCH for a null out-handle, a context, model or pool mismatch,
// or a budget outside [1, cap]; then SSLM_GPU_KV_POOL_EXHAUSTED with nothing drawn). Submits nothing,
// so it is not refused Busy (§4.4: "does not submit").
SslmGpuStatus sslm_gpu_prefix_beginImpl(SslmGpuContext* ctx, SslmGpuModelHandle* model, SslmGpuKvPoolHandle* pool,
                                        int32_t budget, SslmGpuPrefixHandle** out_prefix) {
	if (!out_prefix) return SSLM_SEQUENCE_KV_BUFFER_MISMATCH;
	*out_prefix = nullptr;
	if (!ctx || !model || !pool || model->ctx != ctx || pool->ctx != ctx || pool->model != model) {
		return SSLM_SEQUENCE_KV_BUFFER_MISMATCH;
	}
	if (!superslm_gpu::kv_pool::BudgetInDomain(budget, model->context_cap)) return SSLM_SEQUENCE_KV_BUFFER_MISMATCH;
	std::unique_ptr<SslmGpuPrefixHandle> p(new SslmGpuPrefixHandle());
	std::unique_ptr<SslmGpuSequenceHandle> h;
	const SslmGpuStatus st = MakePooledSeqUnpublished(ctx, model, pool, budget, &h);
	if (st != SSLM_OK) return st;
	p->ctx = ctx;
	p->model = model;
	p->pool = pool;
	p->seq = PublishPooledSeq(std::move(h));
	*out_prefix = p.release();
	return SSLM_OK;
}

// §3.5/§4 items 7 and 8: a budgeted child of the frozen `parent`, of the parent's pool (begin_from takes
// no pool, so it cannot cross pools, and no refusal is written for it). Refusals, in §4.4's order, each
// changing nothing: a null out-handle, a null or another context's parent, a budget outside [1, cap]
// (SSLM_SEQUENCE_KV_BUFFER_MISMATCH); a parent that is not frozen (SSLM_GPU_PREFIX_REJECTED); then the
// entry guard (EntrySubmitGuard) (before the draw); then SSLM_GPU_KV_POOL_EXHAUSTED with nothing drawn. It shares the
// parent's floor(len/B) full pages, copies the partial tail page through the host mirror into a page of
// its own reserve and uploads that page (the pool upload, on the process's device), and copies the
// parent's state verbatim -- the bind permission and the ready flag included (gap 4b) -- with origin =
// the parent's length. A failure of the upload's staging buffer returns the draw before anything is
// shared; a failure of its submission releases the holder (its pages back to the pool) and is
// classified, rules 0-3, as every handle create's.
SslmGpuStatus sslm_gpu_prefix_begin_fromImpl(SslmGpuContext* ctx, SslmGpuPrefixHandle* parent, int32_t budget,
                                             SslmGpuPrefixHandle** out_prefix) {
	if (!out_prefix) return SSLM_SEQUENCE_KV_BUFFER_MISMATCH;
	*out_prefix = nullptr;
	if (!ctx || !parent || parent->ctx != ctx || parent->seq == nullptr) return SSLM_SEQUENCE_KV_BUFFER_MISMATCH;
	if (!parent->frozen) return SSLM_GPU_PREFIX_REJECTED;  // §4 item 7, A19
	SslmGpuModelHandle* model = parent->model;
	SslmGpuKvPoolHandle* pool = parent->pool;
	const SslmGpuSequenceHandle* src = parent->seq;
	if (!superslm_gpu::kv_pool::BudgetInDomain(budget, model->context_cap)) return SSLM_SEQUENCE_KV_BUFFER_MISMATCH;
	const int64_t B = pool->geo.page_positions;
	const int64_t len = src->context_length;
	const uint32_t full = static_cast<uint32_t>(len / B);
	const uint32_t tail = (len % B) != 0 ? 1u : 0u;
	const uint32_t reserve = superslm_gpu::kv_pool::ReservationPages(budget, pool->geo, model->context_cap);
	if (full + tail > pool->geo.cap_pages || full + tail > superslm::kv_pages::HolderMapped(src->holder) ||
	    tail > reserve) {
		return SSLM_SEQUENCE_KV_BUFFER_MISMATCH;  // unreachable for a frozen parent of this pool
	}
	// EntrySubmitGuard (§4.4): begin_from reaches the singleton (the pool upload); scope any sequence;
	// after the last refusal that changes nothing, before the draw.
	if (const SslmGpuStatus g = EntrySubmitGuard(GuardScope::kAny, GuardDevice::kReachesSingleton); g != SSLM_OK) {
		return g;
	}

	std::unique_ptr<SslmGpuPrefixHandle> p(new SslmGpuPrefixHandle());
	std::unique_ptr<SslmGpuSequenceHandle> h;
	const SslmGpuStatus st = MakePooledSeqUnpublished(ctx, model, pool, budget, &h);
	if (st != SSLM_OK) return st;
	superslm_gpu::harness::Device& dev = superslm_gpu::harness::GetDevice();
	std::vector<superslm_gpu::PoolUploadRegion> regions;
	Microsoft::WRL::ComPtr<ID3D12Resource> staging;
	if (tail != 0) {
		// Phase A before anything is shared: the staging buffer holds the parent's tail page, which is
		// exactly what the child's tail page holds once the host copy below has run.
		try {
			const uint32_t src_page = superslm::kv_pages::HolderTableEntryForAddress(src->holder, full);
			const std::vector<const uint8_t*> sources{superslm::kv_pages::PageBase(pool->pages, src_page)};
			regions.push_back(superslm_gpu::PoolUploadRegion{0u, 0u, pool->geo.page_bytes});
			staging = AllocatePoolUploadStaging(dev, regions, sources);
		} catch (...) {
			superslm::kv_pages::HolderRelease(h->holder);
			return ClassifyHandleFault(&dev);
		}
	}
	if (full > 0) {
		const superslm::kv_pages::PageStatus ps = superslm::kv_pages::HolderShareLeading(h->holder, src->holder, full);
		if (ps != superslm::kv_pages::PageStatus::kOk) {
			superslm::kv_pages::HolderRelease(h->holder);
			return SSLM_SEQUENCE_KV_BUFFER_MISMATCH;
		}
	}
	if (tail != 0) {
		const superslm::kv_pages::PageStatus ps = superslm::kv_pages::HolderMapNext(h->holder);
		if (ps != superslm::kv_pages::PageStatus::kOk) {
			superslm::kv_pages::HolderRelease(h->holder);
			return SSLM_SEQUENCE_KV_BUFFER_MISMATCH;  // unreachable: the reserve holds the tail page
		}
		const uint32_t src_page = superslm::kv_pages::HolderTableEntryForAddress(src->holder, full);
		const uint32_t dst_page = superslm::kv_pages::HolderTableEntryForAddress(h->holder, full);
		std::memcpy(superslm::kv_pages::PageBase(pool->pages, dst_page),
		            superslm::kv_pages::PageBase(pool->pages, src_page), pool->geo.page_bytes);
		regions[0].page = dst_page;
		try {
			SubmitPoolUploadSync(dev, pool, regions, staging);
		} catch (...) {
			superslm::kv_pages::HolderRelease(h->holder);
			return ClassifyHandleFault(&dev);
		}
	}
	h->table_dirty = true;
	CopyPrefixForwardState(h.get(), src);
	h->bound_schema_index = src->bound_schema_index;
	h->dfa_walk_state = src->dfa_walk_state;
	h->bind_eligible = src->bind_eligible;
	h->origin = len;
	h->limit = superslm_gpu::kv_pool::HolderLimit(len, budget, model->context_cap);
	p->ctx = ctx;
	p->model = model;
	p->pool = pool;
	p->current_token = parent->current_token;
	p->seq = PublishPooledSeq(std::move(h));
	*out_prefix = p.release();
	return SSLM_OK;
}

// §4 item 7: binds `schema_index` through the prefix's internal sequence (SslmGpuSeqSetSchemaForG5Bridge's
// rules), and returns SSLM_GPU_PREFIX_REJECTED whenever that sequence cannot bind: after content, after a
// call that revoked its bind permission (3.6b), for an index the model does not have, and on a frozen
// prefix (the CPU twin refuses both "frozen" and "after content"). Submits nothing.
SslmGpuStatus sslm_gpu_prefix_set_schemaImpl(SslmGpuContext* ctx, SslmGpuPrefixHandle* prefix, int32_t schema_index) {
	if (!ctx || !prefix || prefix->ctx != ctx || prefix->seq == nullptr) return SSLM_SEQUENCE_KV_BUFFER_MISMATCH;
	if (prefix->frozen) return SSLM_GPU_PREFIX_REJECTED;
	const SslmGpuStatus st = SslmGpuSeqSetSchemaForG5BridgeImpl(ctx, prefix->seq, schema_index);
	return st == SSLM_SEQUENCE_REJECTED ? SSLM_GPU_PREFIX_REJECTED : st;
}

// §4 item 7: prefills up to `count` tokens into the prefix through the prompt (`kind` 0) or the
// schema-content (`kind` 1) prefill bridge on its internal sequence, adapter unbound. Refusals before the
// guard (§4.4's table): the arguments (SSLM_SEQUENCE_KV_BUFFER_MISMATCH: a null `consumed`, context or
// prefix, another context's prefix, a negative count, null tokens, a zero budget, a kind other than 0 or
// 1); after freeze, SSLM_GPU_PREFIX_REJECTED; and, as the decode call's budget refusal, a dispatch_budget
// below one token at full depth while count > 0: SSLM_DISPATCH_BUDGET_TOO_SMALL. Then the entry guard (EntrySubmitGuard)
// (after the freeze check). The call admits n = min(count, floor(dispatch_budget / (dispatches per layer
// * layers))) tokens and hands them to the bridge, whose statuses it returns unchanged; `*consumed` counts
// the tokens that landed (the schema bridge's own count, or the prompt bridge's context-length advance).
// A call that admitted n < count tokens and landed all n returns SSLM_OK with `*consumed` = n, resumable,
// as the CPU twin's chunk_budget (cell 5.8). Repeated calls of any count are allowed (gap 4c).
SslmGpuStatus sslm_gpu_prefix_prefillImpl(SslmGpuContext* ctx, SslmGpuPrefixHandle* prefix, const int32_t* tokens,
                                          int32_t count, int32_t kind, uint32_t dispatch_budget, int32_t* consumed) {
	if (!consumed) return SSLM_SEQUENCE_KV_BUFFER_MISMATCH;
	*consumed = 0;
	if (!ctx || !prefix || prefix->ctx != ctx || prefix->seq == nullptr) return SSLM_SEQUENCE_KV_BUFFER_MISMATCH;
	if (count < 0 || (count > 0 && !tokens) || dispatch_budget < 1 || (kind != 0 && kind != 1)) {
		return SSLM_SEQUENCE_KV_BUFFER_MISMATCH;
	}
	if (prefix->frozen) return SSLM_GPU_PREFIX_REJECTED;
	const uint32_t per_token = PrefixPrefillDispatchesPerToken(prefix->model);
	if (count > 0 && dispatch_budget < per_token) return SSLM_DISPATCH_BUDGET_TOO_SMALL;
	// EntrySubmitGuard (§4.4): the prefill bridges reach the singleton; scope any sequence; after the
	// freeze check and the budget refusal.
	if (const SslmGpuStatus g = EntrySubmitGuard(GuardScope::kAny, GuardDevice::kReachesSingleton); g != SSLM_OK) {
		return g;
	}
	const uint32_t admissible = dispatch_budget / per_token;
	const int32_t n = static_cast<uint32_t>(count) < admissible ? count : static_cast<int32_t>(admissible);
	SslmGpuSequenceHandle* seq = prefix->seq;
	if (kind == 0) {
		const int64_t before = seq->context_length;
		const SslmGpuStatus st = SslmGpuSeqPrefillPromptForG5BridgeImpl(ctx, seq, tokens, n, per_token);
		const int64_t landed = seq->context_length - before;
		*consumed = landed <= 0 ? 0 : (landed >= n ? n : static_cast<int32_t>(landed));
		if (*consumed > 0) prefix->current_token = tokens[*consumed - 1];
		return st;
	}
	int32_t landed = 0;
	const SslmGpuStatus st = SslmGpuSeqPrefillSchemaContentForG5BridgeImpl(ctx, seq, tokens, n, per_token, &landed);
	*consumed = landed;
	if (landed > 0) prefix->current_token = tokens[landed - 1];
	return st;
}

// §3.4's freeze row: the first call returns the prefix's unmapped reserve pages to the pool, so the frozen
// prefix keeps exactly ceil(len/B) pages; later calls return SSLM_OK with no pool effect (G33). Submits
// nothing.
// The first freeze also sets the state restore derives from L (§4 items 2 and 7; code review 1.14 round 1,
// findings 1 and 2): SPX1 carries no ready flag, snapshot flag or bind permission, so sslm_gpu_prefix_restore
// sets the prefill snapshot to the saved residual and valid iff L > 0, and the bind permission iff L == 0.
// On the original those were functions of the call history (a later failed or zero-count prefill call
// invalidates the snapshot or clears the permission). Every reader of either -- an adopt and begin_from --
// takes a frozen prefix only, so normalising here makes the original and its restored copy agree whatever
// the history, and replaces the per-exit snapshot refresh the prefill verb used to make.
SslmGpuStatus sslm_gpu_prefix_freezeImpl(SslmGpuContext* ctx, SslmGpuPrefixHandle* prefix) {
	if (!ctx || !prefix || prefix->ctx != ctx || prefix->seq == nullptr) return SSLM_SEQUENCE_KV_BUFFER_MISMATCH;
	if (prefix->frozen) return SSLM_OK;
	const superslm::kv_pages::PageStatus ps = superslm::kv_pages::HolderReturnReserve(prefix->seq->holder, nullptr);
	if (ps != superslm::kv_pages::PageStatus::kOk) return SSLM_SEQUENCE_KV_BUFFER_MISMATCH;
	prefix->frozen = true;  // only once the reserve is back: a refused freeze leaves the prefix unfrozen
	SslmGpuSequenceHandle* seq = prefix->seq;
	const bool ready = seq->context_length > 0 && seq->ready_for_logits;  // restore: ready = L > 0 (a savable prefix)
	seq->prefill_hidden_codes = seq->hidden_codes;
	seq->prefill_hidden_valid = ready;
	seq->bind_eligible = seq->context_length == 0;                      // restore: bind_eligible = !(L > 0)
	return SSLM_OK;
}

// §3.3's release rows: the internal sequence's release (its private pages and reserve back to the pool,
// shared pages decremented, so pages an adopter or a child still maps stay live), then the handle. A null
// prefix is a no-op (as every GPU release); another context's prefix is SSLM_SEQUENCE_KV_BUFFER_MISMATCH.
// Submits nothing.
SslmGpuStatus sslm_gpu_prefix_releaseImpl(SslmGpuContext* ctx, SslmGpuPrefixHandle* prefix) {
	if (!prefix) return SSLM_OK;
	if (prefix->ctx != ctx) return SSLM_SEQUENCE_KV_BUFFER_MISMATCH;
	if (prefix->seq != nullptr) {
		// Only SSLM_BUSY leaves the internal sequence alive; every other status (a refcount underflow's
		// SSLM_SEQUENCE_KV_BUFFER_MISMATCH included) has already consumed and deleted it, so the handle
		// is finished too, as the sequence release finishes (G4 review F2).
		const SslmGpuStatus st = sslm_gpu_seq_releaseImpl(ctx, prefix->seq);
		if (st == SSLM_BUSY) return st;
		prefix->seq = nullptr;
		delete prefix;
		return st;
	}
	delete prefix;
	return SSLM_OK;
}

// §4 items 7 and 8, §3.9 A15-A18: adopts the frozen `prefix` onto the pooled sequence `seq` -- a share
// adopt (every pooled GPU sequence is a budget holder): the prefix's floor(len/B) full pages mapped shared
// and its partial tail page copied through the host mirror into a page of the sequence's own reserve,
// then uploaded (the pool upload, on the process's device). It never draws from the pool. Every check
// runs before any state change, the bind permission included (§4 item 7, A18's GPU twin), in this order:
// a null or another context's handle, or a prefix of another model (SSLM_SEQUENCE_KV_BUFFER_MISMATCH); a
// dedicated `seq` (SSLM_GPU_HOLDER_MODE_MISMATCH, A15); a prefix that is not frozen, or of another pool
// (SSLM_GPU_PREFIX_REJECTED, A16/A17); the schema rule (SSLM_GPU_PREFIX_SCHEMA_MISMATCH, A18); then the
// entry guard (EntrySubmitGuard), which also refuses an adopt onto a Submitted `seq` (SSLM_BUSY). On success the CPU adopt's
// field rules: the forward state copied, the walk state transferred iff the prefix carries real schema
// progress (else reset to the bound schema's start), the binding kept, origin = the prefix's length, and
// the bind permission revoked (gap 4d); a refusal revokes nothing. A failure of the upload's staging buffer
// changes nothing; a failure of its submission leaves the sequence reset (its table emptied, as
// sslm_gpu_seq_reset leaves it) and is classified, rules 0-3, as every handle create's.
SslmGpuStatus sslm_gpu_seq_adopt_prefixImpl(SslmGpuContext* ctx, SslmGpuSequenceHandle* seq, SslmGpuPrefixHandle* prefix) {
	if (!ctx || !seq || !prefix || prefix->seq == nullptr) return SSLM_SEQUENCE_KV_BUFFER_MISMATCH;
	if (seq->ctx != ctx || prefix->ctx != ctx) return SSLM_SEQUENCE_KV_BUFFER_MISMATCH;  // another context
	if (seq->model == nullptr || seq->model != prefix->model) return SSLM_SEQUENCE_KV_BUFFER_MISMATCH;  // another model
	if (seq->pool == nullptr || seq->holder == nullptr) return SSLM_GPU_HOLDER_MODE_MISMATCH;  // A15
	if (!prefix->frozen) return SSLM_GPU_PREFIX_REJECTED;                                     // A17
	if (prefix->pool != seq->pool) return SSLM_GPU_PREFIX_REJECTED;                           // A16
	const SslmGpuSequenceHandle* src = prefix->seq;
	// The CPU adopt's schema rule (src/sslm_abi.cpp, sslm_seq_adopt_prefix): the prefix's walk transfers
	// iff the sequence's binding matches the prefix's, or the prefix carries no real schema-content
	// progress (unbound, or bound and still at its start state). Every other case is a mismatch.
	const bool prefix_has_real_progress = src->bound_schema_index >= 0 &&
	                                      src->dfa_walk_state != kSslmGpuDfaWalkStateUnused &&
	                                      src->dfa_walk_state != 0u;
	if (prefix_has_real_progress && seq->bound_schema_index != src->bound_schema_index) {
		return SSLM_GPU_PREFIX_SCHEMA_MISMATCH;  // A18's GPU twin: checked before any state change
	}
	SslmGpuKvPoolHandle* pool = seq->pool;
	const int64_t B = pool->geo.page_positions;
	const int64_t len = src->context_length;
	const uint32_t adopt_pages = superslm_gpu::kv_pool::PagesToCover(len, B);
	const uint32_t share_pages = static_cast<uint32_t>(len / B);
	const uint32_t tail = adopt_pages - share_pages;
	{
		const superslm::kv_pages::PageHolder* kv = seq->holder;
		const uint64_t after_unmap = kv->reserve.size() + (kv->mapped - kv->shared - kv->materialized);
		if (adopt_pages > kv->table_entries || adopt_pages > superslm::kv_pages::HolderMapped(src->holder)) {
			return SSLM_SEQUENCE_KV_BUFFER_MISMATCH;  // unreachable for a frozen prefix of this pool
		}
		if (tail > after_unmap) return SSLM_GPU_KV_POOL_EXHAUSTED;  // unreachable: R(budget) >= 1 (§3.4)
	}
	// EntrySubmitGuard (§4.4): adopt reaches the singleton (the pool upload); scope any sequence; after
	// the last check, before any state change. An adopt onto a Submitted sequence is in that scope.
	if (const SslmGpuStatus g = EntrySubmitGuard(GuardScope::kAny, GuardDevice::kReachesSingleton); g != SSLM_OK) {
		return g;
	}

	superslm_gpu::harness::Device& dev = superslm_gpu::harness::GetDevice();
	std::vector<superslm_gpu::PoolUploadRegion> regions;
	Microsoft::WRL::ComPtr<ID3D12Resource> staging;
	if (tail != 0) {
		// Phase A before any state change: the staging buffer holds the prefix's tail page, which is
		// exactly what the sequence's tail page holds once the host copy below has run.
		try {
			const uint32_t src_page = superslm::kv_pages::HolderTableEntryForAddress(src->holder, share_pages);
			const std::vector<const uint8_t*> sources{superslm::kv_pages::PageBase(pool->pages, src_page)};
			regions.push_back(superslm_gpu::PoolUploadRegion{0u, 0u, pool->geo.page_bytes});
			staging = AllocatePoolUploadStaging(dev, regions, sources);
		} catch (...) {
			return ClassifyHandleFault(&dev);
		}
	}
	// The state change: the unmap and the share in one pool-mutex section (§3.3), then the tail.
	const superslm::kv_pages::PageStatus share_ps =
	    superslm::kv_pages::HolderAdoptShare(seq->holder, src->holder, share_pages);
	if (share_ps != superslm::kv_pages::PageStatus::kOk) return SSLM_SEQUENCE_KV_BUFFER_MISMATCH;
	seq->table_dirty = true;
	if (tail != 0) {
		const superslm::kv_pages::PageStatus map_ps = superslm::kv_pages::HolderMapNext(seq->holder);
		if (map_ps != superslm::kv_pages::PageStatus::kOk) {
			(void)sslm_gpu_seq_resetImpl(ctx, seq);
			return SSLM_GPU_KV_POOL_EXHAUSTED;  // unreachable: checked above
		}
		const uint32_t src_page = superslm::kv_pages::HolderTableEntryForAddress(src->holder, share_pages);
		const uint32_t dst_page = superslm::kv_pages::HolderTableEntryForAddress(seq->holder, share_pages);
		std::memcpy(superslm::kv_pages::PageBase(pool->pages, dst_page),
		            superslm::kv_pages::PageBase(pool->pages, src_page), pool->geo.page_bytes);
		regions[0].page = dst_page;
		try {
			SubmitPoolUploadSync(dev, pool, regions, staging);
		} catch (...) {
			const SslmGpuStatus fault = ClassifyHandleFault(&dev);
			(void)sslm_gpu_seq_resetImpl(ctx, seq);
			return fault;
		}
	}
	if (prefix_has_real_progress) {
		seq->dfa_walk_state = src->dfa_walk_state;
	} else {
		seq->dfa_walk_state = seq->bound_schema_index >= 0 ? 0u : kSslmGpuDfaWalkStateUnused;
	}
	CopyPrefixForwardState(seq, src);
	seq->origin = len;
	seq->limit = superslm_gpu::kv_pool::HolderLimit(len, seq->budget, seq->context_cap);
	seq->bind_eligible = false;  // gap 4d: a successful adopt revokes it, as the CPU adopt does
	return SSLM_OK;
}

// §4 item 10, §3.7 and §3.9 A12-A14 and A21, step G5: restores an 'SLM6' budget blob into a fresh pooled
// sequence of `pool`, keeping its budget and origin (the same holder as before the save: the same limit,
// and R(budget) private pages after its first reset or adopt). `prefix_or_null` is a sharing hint, never
// a checked argument. §4.10's order, each refusal changing nothing:
//   1. the arguments: a null out-handle, context, model, pool or blob, a model or pool of another
//      context, a pool of another model (A21) -> SSLM_SEQUENCE_KV_BUFFER_MISMATCH, null out-handle;
//   2. EntrySubmitGuard: any sequence in the process Submitted, or the singleton's ledger unobserved ->
//      SSLM_BUSY (stranded -> SSLM_DEVICE_LOST), as the sibling restore's guard (§4.4's row: singleton,
//      any sequence);
//   3. the magic: anything that is not 'SLM6' ('SLM5', 'SLM4', any other magic, or a blob too short to
//      carry one) -> SSLM_GPU_HOLDER_MODE_MISMATCH (A14);
//   4. the body (ParseSlm6Body): SSLM_SEQUENCE_KV_BUFFER_MISMATCH, or SSLM_RESTORE_MODEL_MISMATCH for
//      another model's blob;
//   5. the draw: SSLM_GPU_KV_POOL_EXHAUSTED with nothing drawn. R(budget) and, on the private path, the E
//      materialized pages are one kv_pages draw (HolderCreate), so a refusal of either part draws nothing
//      and the first is rolled back by construction.
// Re-sharing (A12) only under §3.7's identity conditions: (a) the blob is budget mode (every valid SLM6
// is), (b) the prefix is frozen, (c) of the destination model, (d) of the destination pool (which also
// makes it of this context), (e) its context_length == origin, and (f) its HOST-MIRROR rows at positions
// [0, floor(origin/B)*B), every layer, K and V, equal the blob's own. Then its floor(origin/B) full pages
// are shared (+1 each under the pool mutex) and only [floor(origin/B)*B, L') is scattered. Otherwise
// (A13: no prefix, a mismatch, a prefix of another context, model or pool) the restore is private:
// R(budget) + E pages, E = min(floor(origin/B), ceil(cap/B) - R(budget)), the E pages mapped as
// materialized (back to the pool at the first reset, adopt or release, §3.3), *out_shared_pages = 0.
// The scatter writes the pool's host mirror, and only the pages it wrote are uploaded, whole, through the
// pool upload (UploadPoolRegionsSync on the process's device, after the guard). A failure of that upload
// releases the holder (its pages back to the pool; nothing was published) and is classified, rules 0-3,
// as every handle create's.
SslmGpuStatus sslm_gpu_seq_restore_into_poolImpl(SslmGpuContext* ctx, SslmGpuModelHandle* model, SslmGpuKvPoolHandle* pool,
                                                 const void* blob, size_t blob_size, SslmGpuPrefixHandle* prefix_or_null,
                                                 SslmGpuSequenceHandle** out_seq, uint32_t* out_shared_pages) {
	if (out_shared_pages) *out_shared_pages = 0;
	if (!out_seq) return SSLM_SEQUENCE_KV_BUFFER_MISMATCH;
	*out_seq = nullptr;
	// 1. The checked arguments (A21). The prefix is not one of them (A13, decision 64).
	if (!ctx || !model || !pool || !blob || model->ctx != ctx || pool->ctx != ctx || pool->model != model ||
	    pool->pages == nullptr) {
		return SSLM_SEQUENCE_KV_BUFFER_MISMATCH;
	}
	// 2. EntrySubmitGuard (§4.4): the pool upload reaches the singleton; scope any sequence; after the
	// call-level refusals, before the magic (§4.10 step 2).
	if (const SslmGpuStatus g = EntrySubmitGuard(GuardScope::kAny, GuardDevice::kReachesSingleton); g != SSLM_OK) {
		return g;
	}
	// 3. The magic (A14): only a pooled sequence's 'SLM6' restores into a pool.
	if (GpuSeqBlobMagic(blob, blob_size) != kGpuSeqBlobMagicSlm6) return SSLM_GPU_HOLDER_MODE_MISMATCH;
	// 4. The body.
	Slm6Restore b;
	if (const SslmGpuStatus st = ParseSlm6Body(model, blob, blob_size, &b); st != SSLM_OK) return st;

	const superslm_gpu::kv_pool::GpuPageGeometry& geo = pool->geo;
	const int64_t B = geo.page_positions;
	const size_t D = model->head_dim;
	const size_t runs = GpuKvRuns(model);
	const uint32_t full = static_cast<uint32_t>(b.origin / B);
	const uint32_t reserve = superslm_gpu::kv_pool::ReservationPages(b.budget, geo, model->context_cap);
	// The identity conditions (a)-(f), read before anything is drawn. (a) holds for every valid SLM6.
	const SslmGpuSequenceHandle* src = (prefix_or_null != nullptr) ? prefix_or_null->seq : nullptr;
	bool share = full > 0 && src != nullptr && prefix_or_null->frozen &&                 // (b)
	             prefix_or_null->model == model && src->model == model &&               // (c)
	             prefix_or_null->pool == pool && src->pool == pool && src->holder != nullptr &&  // (d)
	             src->context_length == b.origin &&                                     // (e)
	             full <= superslm::kv_pages::HolderMapped(src->holder);
	if (share) {  // (f): the prefix's host-mirror rows against the blob's, [0, full * B), every run
		const size_t span_bytes = static_cast<size_t>(B) * D;
		for (size_t r = 0; share && r < runs; ++r) {
			const uint8_t* run_src = b.rows + r * static_cast<size_t>(b.lprime) * D;
			for (uint32_t pg = 0; pg < full; ++pg) {
				const int64_t p0 = static_cast<int64_t>(pg) * B;
				if (std::memcmp(PooledMirrorRow(pool, src->holder, r, p0, D), run_src + static_cast<size_t>(p0) * D,
				                span_bytes) != 0) {
					share = false;
					break;
				}
			}
		}
	}
	const uint32_t share_pages = share ? full : 0u;
	const uint32_t materialized = share ? 0u : std::min<uint32_t>(full, geo.cap_pages - reserve);
	const uint32_t needed = superslm_gpu::kv_pool::PagesToCover(b.lprime, B);
	if (needed > geo.cap_pages || share_pages > needed || materialized > needed) {
		return SSLM_SEQUENCE_KV_BUFFER_MISMATCH;  // unreachable: L' <= cap and floor(origin/B) <= ceil(L'/B)
	}
	std::vector<superslm_gpu::PoolUploadRegion> regions;
	regions.reserve(needed - share_pages);  // before the draw: nothing below it allocates but the upload

	// 5. The draw: R(budget) + E in one kv_pages draw, and the table buffer (MakePooledSeqUnpublished).
	std::unique_ptr<SslmGpuSequenceHandle> h;
	if (const SslmGpuStatus st = MakePooledSeqUnpublished(ctx, model, pool, b.budget, &h, materialized); st != SSLM_OK) {
		return st;
	}
	superslm::kv_pages::PageHolder* kv = h->holder;
	superslm::kv_pages::PageStatus ps = superslm::kv_pages::PageStatus::kOk;
	if (share_pages > 0) {
		ps = superslm::kv_pages::HolderShareLeading(kv, src->holder, share_pages);
	} else if (materialized > 0) {
		ps = superslm::kv_pages::HolderMapMaterialized(kv, materialized);
	}
	while (ps == superslm::kv_pages::PageStatus::kOk && superslm::kv_pages::HolderMapped(kv) < needed) {
		ps = superslm::kv_pages::HolderMapNext(kv);  // R(budget) covers the rest (§3.4/§3.7)
	}
	if (ps != superslm::kv_pages::PageStatus::kOk) {
		superslm::kv_pages::HolderRelease(kv);
		return ps == superslm::kv_pages::PageStatus::kExhausted ? SSLM_GPU_KV_POOL_EXHAUSTED
		                                                        : SSLM_SEQUENCE_KV_BUFFER_MISMATCH;
	}
	// The scatter into the host mirror: rows [share_pages * B, L') of every run.
	for (size_t r = 0; r < runs; ++r) {
		const uint8_t* run_src = b.rows + r * static_cast<size_t>(b.lprime) * D;
		for (int64_t p0 = static_cast<int64_t>(share_pages) * B; p0 < b.lprime; p0 += B) {
			const int64_t n = std::min<int64_t>(B, b.lprime - p0);
			std::memcpy(PooledMirrorRow(pool, kv, r, p0, D), run_src + static_cast<size_t>(p0) * D,
			            static_cast<size_t>(n) * D);
		}
	}
	for (uint32_t i = share_pages; i < needed; ++i) {
		regions.push_back(superslm_gpu::PoolUploadRegion{superslm::kv_pages::HolderTableEntryForAddress(kv, i), 0u,
		                                                 static_cast<uint64_t>(geo.page_bytes)});
	}
	// The pool upload of the pages written (phase A and phase B), on the process's device.
	superslm_gpu::harness::Device& dev = superslm_gpu::harness::GetDevice();
	try {
		superslm_gpu::UploadPoolRegionsSync(dev, pool, regions);
	} catch (...) {
		const SslmGpuStatus fault = ClassifyHandleFault(&dev);
		superslm::kv_pages::HolderRelease(kv);
		return fault;
	}
	// The holder the blob was saved from (§3.7): its budget, origin and limit, its forward state verbatim,
	// its schema binding and walk state (validated above), the ready flag; no adapter, no bind permission.
	h->table_dirty = true;
	h->budget = b.budget;
	h->origin = b.origin;
	h->limit = superslm_gpu::kv_pool::HolderLimit(b.origin, b.budget, model->context_cap);
	std::memcpy(h->hidden_codes.data(), b.residual, h->hidden_codes.size());
	h->hidden_scale = b.hidden_scale;
	h->layer_index = b.layer_index;
	h->kv_saturation_count = b.kv_saturation_count;
	h->context_length = b.context_length;
	h->live_state.hidden_scale = b.hidden_scale;
	h->live_state.layer_index = b.layer_index;
	h->live_state.kv_saturation_count = b.kv_saturation_count;
	h->live_state.kv_landing_saturation_count = b.kv_landing_saturation_count;
	h->live_state.k_channel_landing_saturation_count = b.k_channel_landing_saturation_count;
	h->live_state.rope_q_saturation_count = b.rope_q_saturation_count;
	h->live_state.rope_k_saturation_count = b.rope_k_saturation_count;
	h->live_state.context_length = b.context_length;
	h->bound_schema_index = b.bound_schema_index;
	h->dfa_walk_state = b.dfa_walk_state;
	h->ready_for_logits = b.ready_for_logits;
	h->prefill_hidden_valid = false;
	h->bind_eligible = false;
	SetGpuSeqState(h.get(), superslm_gpu::SslmSequenceGpuState::Idle, nullptr);  // a fresh handle: Idle, no count
	if (out_shared_pages) *out_shared_pages = share_pages;
	*out_seq = PublishPooledSeq(std::move(h));
	return SSLM_OK;
}

// Precached prefixes (1.16.0; plan §4): the GPU twin of sslm_prefix_save / sslm_prefix_restore over the
// same 'SPX1' layout (§3.2): a 184-byte header, the residual, then rows [0, L) in canonical order
// [layer][K|V][kv_head][pos < L][d]. backend = 1 and the producer adapter's VendorId/DeviceId at 20/24. The
// bytes and the reader's check order are src/spx1_format's, the same writer and reader the CPU verbs use.
namespace {

// The context's adapter identity (the producer field): DXGI_ADAPTER_DESC1's VendorId and DeviceId.
bool ContextAdapterIds(SslmGpuContext* ctx, uint32_t* vendor, uint32_t* device) {
	if (ctx->device.adapter == nullptr) return false;
	DXGI_ADAPTER_DESC1 desc{};
	if (FAILED(ctx->device.adapter->GetDesc1(&desc))) return false;
	*vendor = desc.VendorId;
	*device = desc.DeviceId;
	return true;
}
// The same, as the SPX1 reader's producer callback (`user` is the context).
bool ContextAdapterIdsForReader(void* user, uint32_t* vendor, uint32_t* device) {
	return ContextAdapterIds(static_cast<SslmGpuContext*>(user), vendor, device);
}

#if defined(SUPERSLM_GPU_ALLOC_FAULT_INJECTION)
// The restore fault slot (cells 5.11 and 5.7, GPU halves), per calling thread: position 1 replaces the
// share with a refusal (2: kInvalidArgument, 3: kRefcountUnderflow); position 2 returns the reserve just
// before the private pages are mapped; position 3 fails the restore with SSLM_GPU_ALLOCATION_FAILED once
// everything is drawn, shared and mapped. Each records the holder's shared-entry count when it fires.
struct GpuPrefixRestoreFault {
	int position = 0;
	int share_refusal = 0;
	bool fired = false;
	uint32_t shared_at_fire = 0;
};
thread_local GpuPrefixRestoreFault g_gpu_prefix_restore_fault;
bool FireGpuPrefixRestoreFault(int position, const superslm::kv_pages::PageHolder* holder) {
	if (g_gpu_prefix_restore_fault.position != position) return false;
	g_gpu_prefix_restore_fault.position = 0;
	g_gpu_prefix_restore_fault.fired = true;
	g_gpu_prefix_restore_fault.shared_at_fire = holder->shared;
	return true;
}
#endif

}  // namespace

// §4 item 1. Refusals, each writing nothing: a null size slot, a null context or prefix, or a prefix of
// another context (SSLM_SEQUENCE_KV_BUFFER_MISMATCH); a prefix that is not frozen, or not at rest on a
// token boundary (SSLM_GPU_PREFIX_REJECTED); a null or short buffer (SSLM_OUTPUT_BUFFER_TOO_SMALL, the
// size written). Rows come from the pool's host mirror; nothing is submitted, so there is no guard.
SslmGpuStatus sslm_gpu_prefix_saveImpl(SslmGpuContext* ctx, const SslmGpuPrefixHandle* prefix, void* out_blob,
                                       size_t* out_blob_size) {
	if (!out_blob_size) return SSLM_SEQUENCE_KV_BUFFER_MISMATCH;
	if (!ctx || !prefix || prefix->ctx != ctx || prefix->seq == nullptr) return SSLM_SEQUENCE_KV_BUFFER_MISMATCH;
	if (!prefix->frozen) return SSLM_GPU_PREFIX_REJECTED;
	const SslmGpuSequenceHandle* seq = prefix->seq;
	const SslmGpuModelHandle* model = prefix->model;
	const SslmGpuKvPoolHandle* pool = prefix->pool;
	const superslm::SequenceLayerState& st = seq->live_state;
	const int64_t L = st.context_length;
	const int32_t token = prefix->current_token;
	// The rest a prefix reaches: empty (layer 0, not ready, no token) or after a completed token (full
	// depth, ready, a token of the vocabulary). The blob carries no layer index, so nothing else saves.
	const bool rest = L == 0 ? (st.layer_index == 0 && !seq->ready_for_logits && token == -1)
	                         : (L > 0 && L <= model->context_cap && st.layer_index == model->num_hidden_layers &&
	                            seq->ready_for_logits && token >= 0 && token < model->vocab_size);
	if (!rest) return SSLM_GPU_PREFIX_REJECTED;
	size_t required = 0;
	if (!superslm::spx1::BlobSize(GpuKvRuns(model), model->head_dim, model->hidden_size, static_cast<uint64_t>(L),
	                              &required)) {
		return SSLM_GPU_PREFIX_REJECTED;
	}
	if (!out_blob || *out_blob_size < required) {
		*out_blob_size = required;
		return SSLM_OUTPUT_BUFFER_TOO_SMALL;
	}
	const int64_t B = pool->geo.page_positions;
	if (seq->holder == nullptr ||
	    superslm_gpu::kv_pool::PagesToCover(L, B) > superslm::kv_pages::HolderMapped(seq->holder)) {
		return SSLM_SEQUENCE_KV_BUFFER_MISMATCH;  // unreachable: a frozen prefix maps ceil(L/B) pages
	}
	uint64_t schema_hash = 0;
	if (seq->bound_schema_index >= 0) {
		const superslm::SchemaEntry* entry = model->schemas.ByIndex(static_cast<size_t>(seq->bound_schema_index));
		if (entry == nullptr) return SSLM_SEQUENCE_KV_BUFFER_MISMATCH;  // unreachable: bound through the table
		schema_hash = entry->name_hash;
	}
	uint32_t vendor = 0, device = 0;
	if (!ContextAdapterIds(ctx, &vendor, &device)) return SSLM_DEVICE_LOST;
	uint8_t* p = static_cast<uint8_t*>(out_blob);
	superslm::spx1::Header hdr;
	hdr.backend = superslm::spx1::kBackendGpu;
	hdr.adapter_vendor = vendor;
	hdr.adapter_device = device;
	std::memcpy(hdr.model_hash.data(), model->content_hash.data(), 32);
	hdr.kv_precision = model->kv_precision;
	hdr.schema_hash = schema_hash;
	hdr.walk = seq->dfa_walk_state;
	hdr.current_token = token;
	hdr.length = L;
	hdr.origin = seq->origin;
	hdr.scale_m = st.hidden_scale.m;
	hdr.scale_e = st.hidden_scale.e;
	hdr.kv_saturation_count = st.kv_saturation_count;
	hdr.kv_landing_saturation_count = st.kv_landing_saturation_count;
	hdr.k_channel_landing_saturation_count = st.k_channel_landing_saturation_count;
	hdr.rope_q_saturation_count = st.rope_q_saturation_count;
	hdr.rope_k_saturation_count = st.rope_k_saturation_count;
	superslm::spx1::WriteHeader(p, hdr);
	const size_t residual = model->hidden_size;
	if (residual) std::memcpy(superslm::spx1::Residual(p), seq->hidden_codes.data(), residual);
	uint8_t* rows = superslm::spx1::Rows(p, residual);
	const size_t D = model->head_dim;
	for (size_t r = 0; r < GpuKvRuns(model); ++r) {
		uint8_t* run_dst = rows + r * static_cast<size_t>(L) * D;
		for (int64_t p0 = 0; p0 < L; p0 += B) {
			const int64_t n = std::min<int64_t>(B, L - p0);
			std::memcpy(run_dst + static_cast<size_t>(p0) * D, PooledMirrorRow(pool, seq->holder, r, p0, D),
			            static_cast<size_t>(n) * D);
		}
	}
	superslm::spx1::Seal(p, required);
	*out_blob_size = required;
	return SSLM_OK;
}

// §4 items 2 and 3: validate everything in §4 item 3's order, then draw ceil(L/B) - s pages, share the
// hint's s = floor(origin/B) leading pages when conditions (a)-(f) hold (else s = 0), map the rest,
// scatter rows [s*B, L) into the host mirror and upload only those pages (the pool upload on the
// process's device, after the guard). The returned prefix is frozen and rests as the saved one did.
// A share refusal after the draw returns the draw (SSLM_SEQUENCE_KV_BUFFER_MISMATCH, as begin_from's);
// an upload failure returns every drawn page and shared refcount and is classified as every handle create's.
SslmGpuStatus sslm_gpu_prefix_restoreImpl(SslmGpuContext* ctx, SslmGpuModelHandle* model, SslmGpuKvPoolHandle* pool,
                                          const void* blob, size_t n, SslmGpuPrefixHandle* parent_or_null,
                                          SslmGpuPrefixHandle** out_prefix, uint32_t* out_shared_pages) {
	if (out_shared_pages) *out_shared_pages = 0;
	if (!out_prefix) return SSLM_SEQUENCE_KV_BUFFER_MISMATCH;
	*out_prefix = nullptr;
	// 1. The checked arguments. The hint is not one of them.
	if (!ctx || !model || !pool || !blob || model->ctx != ctx || pool->ctx != ctx || pool->model != model ||
	    pool->pages == nullptr) {
		return SSLM_SEQUENCE_KV_BUFFER_MISMATCH;
	}
	// 2. EntrySubmitGuard (§4.4): the pool upload reaches the singleton; scope any sequence; after the
	// argument refusals, before the blob is read.
	if (const SslmGpuStatus g = EntrySubmitGuard(GuardScope::kAny, GuardDevice::kReachesSingleton); g != SSLM_OK) {
		return g;
	}
	// 3-10. The blob, through the SPX1 reader, in §4 item 3's order: size and magic, the digest, the backend,
	// the producer (engine version, adapter vendor and device, so a blob restores only on the device that wrote
	// it), the model, kv_precision, the schema (by name hash; the walk within its states, or the unbound value
	// when no schema is named), then the domains in §3.4 step 8's order.
	superslm::spx1::Expect expect;
	expect.backend = superslm::spx1::kBackendGpu;
	expect.adapter_ids = &ContextAdapterIdsForReader;
	expect.adapter_user = ctx;
	expect.model_hash = model->content_hash.data();
	expect.kv_precision = model->kv_precision;
	expect.schemas = &model->schemas;
	expect.context_cap = model->context_cap;
	expect.vocab_size = static_cast<uint64_t>(model->vocab_size);
	expect.runs = GpuKvRuns(model);
	expect.head_dim = model->head_dim;
	expect.hidden_size = model->hidden_size;
	superslm::spx1::Fields f;
	switch (superslm::spx1::Read(static_cast<const uint8_t*>(blob), n, expect, &f)) {
		case superslm::spx1::Verdict::kOk: break;
		case superslm::spx1::Verdict::kAdapterUnavailable: return SSLM_DEVICE_LOST;
		case superslm::spx1::Verdict::kProducer: return SSLM_RESTORE_PRODUCER_MISMATCH;
		case superslm::spx1::Verdict::kModel: return SSLM_RESTORE_MODEL_MISMATCH;
		default: return SSLM_SEQUENCE_KV_BUFFER_MISMATCH;  // size, magic, digest, backend, kv_precision, schema, a domain
	}
	const int64_t L = f.header.length;
	const int64_t origin = f.header.origin;
	const int32_t token = f.header.current_token;
	const uint32_t walk = f.header.walk;
	const int32_t schema_index = f.schema_index;
	const int64_t m = f.header.scale_m;
	const int64_t e = f.header.scale_e;

	const superslm_gpu::kv_pool::GpuPageGeometry& geo = pool->geo;
	const int64_t B = geo.page_positions;
	const size_t D = model->head_dim;
	const size_t runs = GpuKvRuns(model);
	const size_t residual = model->hidden_size;
	const uint8_t* rows = f.rows;
	const uint32_t needed = superslm_gpu::kv_pool::PagesToCover(L, B);
	const uint32_t full = static_cast<uint32_t>(origin / B);
	if (needed > geo.cap_pages || full > needed) return SSLM_SEQUENCE_KV_BUFFER_MISMATCH;  // unreachable: L <= cap
	// The sharing hint, conditions (a)-(f), read before anything is drawn: (a) a nested blob (origin > 0),
	// (b) a frozen hint, (c) of this model, (d) of this pool, (e) holding exactly `origin` positions, and
	// (f) its host-mirror rows [0, full * B), every run, equal to the blob's. (e) precedes (f), so a
	// shorter hint's table is never read past what it maps.
	const SslmGpuSequenceHandle* src = parent_or_null != nullptr ? parent_or_null->seq : nullptr;
	bool share = full > 0 && src != nullptr && parent_or_null->frozen &&                    // (a), (b)
	             parent_or_null->ctx == ctx && parent_or_null->model == model &&             // (c)
	             parent_or_null->pool == pool && src->holder != nullptr &&                   // (d)
	             src->context_length == origin &&                                            // (e)
	             full <= superslm::kv_pages::HolderMapped(src->holder);
	if (share) {  // (f)
		const size_t span_bytes = static_cast<size_t>(B) * D;
		for (size_t r = 0; share && r < runs; ++r) {
			const uint8_t* run_src = rows + r * static_cast<size_t>(L) * D;
			for (uint32_t pg = 0; pg < full; ++pg) {
				const int64_t p0 = static_cast<int64_t>(pg) * B;
				if (std::memcmp(PooledMirrorRow(pool, src->holder, r, p0, D), run_src + static_cast<size_t>(p0) * D,
				                span_bytes) != 0) {
					share = false;
					break;
				}
			}
		}
	}
	const uint32_t s = share ? full : 0u;
	const int32_t budget = static_cast<int32_t>(model->context_cap);
	// Everything the success path allocates on the host, before the draw.
	std::unique_ptr<SslmGpuPrefixHandle> out(new SslmGpuPrefixHandle());
	std::vector<superslm_gpu::PoolUploadRegion> regions;
	regions.reserve(needed - s);

	// 11. The draw: exactly ceil(L/B) - s pages and the table buffer; SSLM_GPU_KV_POOL_EXHAUSTED with
	// nothing drawn.
	std::unique_ptr<SslmGpuSequenceHandle> h;
	if (const SslmGpuStatus st = MakePooledSeqUnpublished(ctx, model, pool, budget, &h, needed - s, true); st != SSLM_OK) {
		return st;
	}
	superslm::kv_pages::PageHolder* kv = h->holder;
	// The share, in its own pool-mutex section after the draw.
	if (s > 0) {
		superslm::kv_pages::PageStatus ps = superslm::kv_pages::PageStatus::kOk;
#if defined(SUPERSLM_GPU_ALLOC_FAULT_INJECTION)
		if (FireGpuPrefixRestoreFault(1, kv)) {
			ps = g_gpu_prefix_restore_fault.share_refusal == 3 ? superslm::kv_pages::PageStatus::kRefcountUnderflow
			                                                   : superslm::kv_pages::PageStatus::kInvalidArgument;
		} else
#endif
		{
			ps = superslm::kv_pages::HolderShareLeading(kv, src->holder, s);
		}
		if (ps != superslm::kv_pages::PageStatus::kOk) {
			superslm::kv_pages::HolderRelease(kv);
			return SSLM_SEQUENCE_KV_BUFFER_MISMATCH;
		}
	}
#if defined(SUPERSLM_GPU_ALLOC_FAULT_INJECTION)
	if (g_gpu_prefix_restore_fault.position == 2) {
		superslm::kv_pages::HolderReturnReserve(kv, nullptr);
		FireGpuPrefixRestoreFault(2, kv);
	}
#endif
	while (superslm::kv_pages::HolderMapped(kv) < needed) {
		const superslm::kv_pages::PageStatus ps = superslm::kv_pages::HolderMapNext(kv);
		if (ps != superslm::kv_pages::PageStatus::kOk) {
			superslm::kv_pages::HolderRelease(kv);
			return ps == superslm::kv_pages::PageStatus::kExhausted ? SSLM_GPU_KV_POOL_EXHAUSTED
			                                                        : SSLM_SEQUENCE_KV_BUFFER_MISMATCH;
		}
	}
#if defined(SUPERSLM_GPU_ALLOC_FAULT_INJECTION)
	if (FireGpuPrefixRestoreFault(3, kv)) {
		superslm::kv_pages::HolderRelease(kv);
		return SSLM_GPU_ALLOCATION_FAILED;
	}
#endif
	// The scatter into the host mirror: rows [s*B, L) of every run.
	for (size_t r = 0; r < runs; ++r) {
		const uint8_t* run_src = rows + r * static_cast<size_t>(L) * D;
		for (int64_t p0 = static_cast<int64_t>(s) * B; p0 < L; p0 += B) {
			const int64_t c = std::min<int64_t>(B, L - p0);
			std::memcpy(PooledMirrorRow(pool, kv, r, p0, D), run_src + static_cast<size_t>(p0) * D,
			            static_cast<size_t>(c) * D);
		}
	}
	for (uint32_t i = s; i < needed; ++i) {
		regions.push_back(superslm_gpu::PoolUploadRegion{superslm::kv_pages::HolderTableEntryForAddress(kv, i), 0u,
		                                                 static_cast<uint64_t>(geo.page_bytes)});
	}
	// 12. The pool upload of the pages written, on the process's device.
	superslm_gpu::harness::Device& dev = superslm_gpu::harness::GetDevice();
	try {
		superslm_gpu::UploadPoolRegionsSync(dev, pool, regions);
	} catch (...) {
		const SslmGpuStatus fault = ClassifyHandleFault(&dev);
		superslm::kv_pages::HolderRelease(kv);
		return fault;
	}
	// The prefix as saved: at rest (full depth and ready after a token, or empty), its prefill snapshot
	// the residual, its schema binding and walk, frozen; bind permission only while empty.
	const bool ready = L > 0;
	h->table_dirty = true;
	h->budget = budget;
	h->origin = origin;
	h->limit = superslm_gpu::kv_pool::HolderLimit(origin, budget, model->context_cap);
	if (residual) std::memcpy(h->hidden_codes.data(), f.residual, residual);
	const superslm::CarriedScale scale{m, e};
	const uint32_t layer_index = ready ? model->num_hidden_layers : 0u;
	h->hidden_scale = scale;
	h->layer_index = layer_index;
	h->kv_saturation_count = f.header.kv_saturation_count;
	h->context_length = L;
	h->live_state.hidden_scale = scale;
	h->live_state.layer_index = layer_index;
	h->live_state.kv_saturation_count = f.header.kv_saturation_count;
	h->live_state.kv_landing_saturation_count = f.header.kv_landing_saturation_count;
	h->live_state.k_channel_landing_saturation_count = f.header.k_channel_landing_saturation_count;
	h->live_state.rope_q_saturation_count = f.header.rope_q_saturation_count;
	h->live_state.rope_k_saturation_count = f.header.rope_k_saturation_count;
	h->live_state.context_length = L;
	h->bound_schema_index = schema_index;
	h->dfa_walk_state = walk;
	h->ready_for_logits = ready;
	h->prefill_hidden_codes = h->hidden_codes;
	h->prefill_hidden_valid = ready;
	h->bind_eligible = !ready;
	SetGpuSeqState(h.get(), superslm_gpu::SslmSequenceGpuState::Idle, nullptr);  // a fresh handle: Idle, no count
	out->ctx = ctx;
	out->model = model;
	out->pool = pool;
	out->frozen = true;
	out->current_token = token;
	out->seq = PublishPooledSeq(std::move(h));
	if (out_shared_pages) *out_shared_pages = s;
	*out_prefix = out.release();
	return SSLM_OK;
}

// Every installed status-returning GPU entry point terminates exceptions here. Allocation
// failures are recoverable and distinguishable; all other unexpected failures retain the
// existing device-failure disposition. No C++ exception crosses the public API boundary.
namespace {
#if defined(SUPERSLM_ENABLE_GPU_API_FAILURE_INJECTION)
std::atomic<int> g_gpu_api_failure_point{-1};
char g_gpu_api_failure_name[96] = {};

void MaybeThrowGpuApiFailure(const char* name, SslmGpuApiFailureInjectionPoint point) {
	if (g_gpu_api_failure_point.load() == static_cast<int>(point) &&
	    std::strcmp(g_gpu_api_failure_name, name) == 0) {
		g_gpu_api_failure_point.store(-1);
		throw std::bad_alloc();
	}
}
#else
void MaybeThrowGpuApiFailure(const char*, int) {}
#endif

// The containment, shared by both boundary forms below.
template <typename Fn>
SslmGpuStatus ContainGpuApiCall(const char* name, Fn&& fn) noexcept {
	try {
#if defined(SUPERSLM_ENABLE_GPU_API_FAILURE_INJECTION)
		MaybeThrowGpuApiFailure(name, SslmGpuApiFailureInjectionPoint::BeforeHandler);
		const SslmGpuStatus status = fn();
		MaybeThrowGpuApiFailure(name, SslmGpuApiFailureInjectionPoint::AfterHandler);
		return status;
#else
		return fn();
#endif
	} catch (const std::bad_alloc&) {
		return SSLM_GPU_ALLOCATION_FAILED;
	} catch (const std::length_error&) {
		return SSLM_GPU_ALLOCATION_FAILED;
	} catch (const superslm_gpu::harness::GpuAllocationError&) {
		// 1.8.0 (TE-426): E_OUTOFMEMORY from a D3D12 call. The boundary sees neither a device nor a
		// command list: every site that can hold a list open or work in flight catches first, so
		// what reaches here had nothing open.
		return SSLM_GPU_ALLOCATION_FAILED;
	} catch (const std::exception& e) {
		std::fprintf(stderr, "%s: contained exception: %s\n", name, e.what());
		return SSLM_DEVICE_LOST;
	} catch (...) {
		std::fprintf(stderr, "%s: contained non-standard exception\n", name);
		return SSLM_DEVICE_LOST;
	}
}

// Paged-KV plan §4.4 (rev 11, rev 12), step G3: every C ABI entry point opens the call scope
// (inflight::GpuApiCallScope) as its first statement, so the boundary token check's walk and the
// boundary guard check run in every build. The scope is declared before the containment, so it
// closes after every local of the call has been destroyed: the walk at its close visits every
// in-flight record this call registered and drains and reports any not held by a Submitted
// handle's `in_flight`.
template <typename Fn>
SslmGpuStatus InvokeGpuApiBoundary(const char* name, Fn&& fn) noexcept {
	superslm_gpu::inflight::GpuApiCallScope scope(name);
	return ContainGpuApiCall(name, std::forward<Fn>(fn));
}

// The two gpu_port.h test entry points (SslmGpuDeviceLogitsBundleCreateForTest /
// SslmGpuDeviceLogitsBundleRunForTest) are internal-layer functions, not C ABI verbs: they keep the
// containment but open no call scope, so they stay at depth 0 and the boundary guard check (which
// refuses a reset inside a C ABI call that passed no EntrySubmitGuard) does not apply to them --
// the S1 row: 3.6e expects SSLM_OK from them while a pooled sequence is Submitted.
template <typename Fn>
SslmGpuStatus InvokeGpuInternalBoundary(const char* name, Fn&& fn) noexcept {
	return ContainGpuApiCall(name, std::forward<Fn>(fn));
}

// Every public entry point with an output handle clears a non-null slot here, before the
// boundary runs, so every refusal -- including SSLM_GPU_ALLOCATION_FAILED produced by the
// boundary itself from a contained exception -- leaves the handle null. A null slot is left
// undereferenced, and the implementation reports its existing status for it.
template <typename Handle>
void ClearOutputHandle(Handle** out) noexcept {
	if (out) {
		*out = nullptr;
	}
}
}  // namespace

#if defined(SUPERSLM_ENABLE_GPU_API_FAILURE_INJECTION)
void ArmSslmGpuApiFailureInjection(const char* api_name,
                                   SslmGpuApiFailureInjectionPoint point) {
	std::snprintf(g_gpu_api_failure_name, sizeof(g_gpu_api_failure_name), "%s",
	              api_name ? api_name : "");
	g_gpu_api_failure_point.store(static_cast<int>(point));
}
void ClearSslmGpuApiFailureInjection() {
	g_gpu_api_failure_point.store(-1);
	g_gpu_api_failure_name[0] = '\0';
}
#endif

SslmGpuStatus sslm_gpu_context_create(GpuContextConfig cfg, SslmGpuContext** out_ctx) noexcept {
	ClearOutputHandle(out_ctx);
	return InvokeGpuApiBoundary(__func__, [&] { return sslm_gpu_context_createImpl(cfg, out_ctx); });
}
SslmGpuStatus sslm_gpu_context_destroy(SslmGpuContext* ctx) noexcept {
	return InvokeGpuApiBoundary(__func__, [&] { return sslm_gpu_context_destroyImpl(ctx); });
}
SslmGpuStatus sslm_gpu_context_set_host_parallel_for(SslmGpuContext* ctx,
                                                     const sslm_parallel_for* pf) noexcept {
	return InvokeGpuApiBoundary(
	    __func__, [&] { return sslm_gpu_context_set_host_parallel_forImpl(ctx, pf); });
}
SslmGpuStatus sslm_gpu_model_map(SslmGpuContext* ctx, const SslmModelView* base,
                                  GpuResidencyConfig cfg,
                                  SslmGpuModelHandle** out_model) noexcept {
	ClearOutputHandle(out_model);
	return InvokeGpuApiBoundary(__func__, [&] { return sslm_gpu_model_mapImpl(ctx, base, cfg, out_model); });
}
SslmGpuStatus sslm_gpu_model_unmap(SslmGpuContext* ctx, SslmGpuModelHandle* model) noexcept {
	return InvokeGpuApiBoundary(__func__, [&] { return sslm_gpu_model_unmapImpl(ctx, model); });
}
SslmGpuStatus sslm_gpu_adapter_map(SslmGpuContext* ctx, SslmGpuModelHandle* model,
                                    const SslmModelView* artifact,
                                    SslmGpuAdapterHandle** out_adapter) noexcept {
	ClearOutputHandle(out_adapter);
	return InvokeGpuApiBoundary(__func__, [&] {
		return sslm_gpu_adapter_mapImpl(ctx, model, artifact, out_adapter);
	});
}
SslmGpuStatus sslm_gpu_adapter_unmap(SslmGpuContext* ctx,
                                      SslmGpuAdapterHandle* adapter) noexcept {
	return InvokeGpuApiBoundary(__func__, [&] { return sslm_gpu_adapter_unmapImpl(ctx, adapter); });
}
SslmGpuStatus sslm_gpu_seq_create(SslmGpuContext* ctx, SslmGpuModelHandle* model,
                                   int64_t cap, SslmGpuSequenceHandle** out_seq) noexcept {
	ClearOutputHandle(out_seq);
	return InvokeGpuApiBoundary(__func__, [&] { return sslm_gpu_seq_createImpl(ctx, model, cap, out_seq); });
}
SslmGpuStatus sslm_gpu_seq_release(SslmGpuContext* ctx, SslmGpuSequenceHandle* seq) noexcept {
	return InvokeGpuApiBoundary(__func__, [&] { return sslm_gpu_seq_releaseImpl(ctx, seq); });
}
SslmGpuStatus sslm_gpu_seq_bind_adapter(SslmGpuContext* ctx, SslmGpuSequenceHandle* seq,
                                         const SslmGpuAdapterHandle* adapter) noexcept {
	return InvokeGpuApiBoundary(__func__, [&] { return sslm_gpu_seq_bind_adapterImpl(ctx, seq, adapter); });
}
SslmGpuStatus sslm_gpu_seq_embed_token(SslmGpuContext* ctx, SslmGpuSequenceHandle* seq,
                                        int32_t token) noexcept {
	return InvokeGpuApiBoundary(__func__, [&] { return sslm_gpu_seq_embed_tokenImpl(ctx, seq, token); });
}
SslmGpuStatus sslm_gpu_seq_save(SslmGpuContext* ctx, const SslmGpuSequenceHandle* seq,
                                 void* blob, size_t* size) noexcept {
	return InvokeGpuApiBoundary(__func__, [&] { return sslm_gpu_seq_saveImpl(ctx, seq, blob, size); });
}
SslmGpuStatus sslm_gpu_seq_restore(SslmGpuContext* ctx, SslmGpuModelHandle* model,
                                    const void* blob, size_t size,
                                    SslmGpuSequenceHandle** out_seq) noexcept {
	ClearOutputHandle(out_seq);
	return InvokeGpuApiBoundary(__func__, [&] {
		return sslm_gpu_seq_restoreImpl(ctx, model, blob, size, out_seq);
	});
}
SslmGpuStatus sslm_gpu_seq_reset(SslmGpuContext* ctx, SslmGpuSequenceHandle* seq) noexcept {
	return InvokeGpuApiBoundary(__func__, [&] { return sslm_gpu_seq_resetImpl(ctx, seq); });
}
SslmGpuStatus sslm_decode_step_gpu(SslmGpuContext* ctx, SslmGpuSequenceHandle* seq,
                                    const SslmGpuAdapterHandle* adapter,
                                    uint32_t budget) noexcept {
	return InvokeGpuApiBoundary(__func__, [&] { return sslm_decode_step_gpuImpl(ctx, seq, adapter, budget); });
}
SslmGpuStatus sslm_decode_step_batch_gpu(SslmGpuContext* ctx,
                                          SslmGpuSequenceHandle* const* seqs,
                                          const SslmGpuAdapterHandle* const* adapters,
                                          uint32_t count, uint32_t budget,
                                          SslmGpuStatus* statuses) noexcept {
	return InvokeGpuApiBoundary(__func__, [&] {
		return sslm_decode_step_batch_gpuImpl(ctx, seqs, adapters, count, budget, statuses);
	});
}
SslmGpuStatus sslm_gpu_ready(SslmGpuContext* ctx, SslmGpuSequenceHandle* seq, int32_t block,
                              int32_t* ready, SslmGpuStatus* status) noexcept {
	return InvokeGpuApiBoundary(__func__, [&] { return sslm_gpu_readyImpl(ctx, seq, block, ready, status); });
}
SslmGpuStatus SslmGpuSeqSetSchemaForG5Bridge(SslmGpuContext* ctx,
                                              SslmGpuSequenceHandle* seq,
                                              int32_t schema) noexcept {
	return InvokeGpuApiBoundary(__func__, [&] { return SslmGpuSeqSetSchemaForG5BridgeImpl(ctx, seq, schema); });
}
SslmGpuStatus SslmGpuSeqSchemaAcceptingForG5Bridge(
    SslmGpuContext* ctx, SslmGpuSequenceHandle* seq, int32_t* out_schema_accepting) noexcept {
	return InvokeGpuApiBoundary(__func__, [&] {
		return SslmGpuSeqSchemaAcceptingForG5BridgeImpl(ctx, seq, out_schema_accepting);
	});
}
SslmGpuStatus SslmGpuSeqSchemaBoundForG5Bridge(
    SslmGpuContext* ctx, SslmGpuSequenceHandle* seq, int32_t* out_schema_bound) noexcept {
	return InvokeGpuApiBoundary(__func__, [&] {
		return SslmGpuSeqSchemaBoundForG5BridgeImpl(ctx, seq, out_schema_bound);
	});
}
SslmGpuStatus SslmGpuSeqFinishTokenForG5Bridge(SslmGpuContext* ctx,
                                                SslmGpuSequenceHandle* seq,
                                                int32_t* token) noexcept {
	return InvokeGpuApiBoundary(__func__, [&] { return SslmGpuSeqFinishTokenForG5BridgeImpl(ctx, seq, token); });
}
SslmGpuStatus SslmGpuSeqPrefillPromptForG5Bridge(SslmGpuContext* ctx,
                                                  SslmGpuSequenceHandle* seq,
                                                  const int32_t* tokens, int32_t count,
                                                  uint32_t budget) noexcept {
	return InvokeGpuApiBoundary(__func__, [&] {
		return SslmGpuSeqPrefillPromptForG5BridgeImpl(ctx, seq, tokens, count, budget);
	});
}
SslmGpuStatus SslmGpuSeqDecodeStepForG5Bridge(SslmGpuContext* ctx,
                                               SslmGpuSequenceHandle* seq, int32_t token,
                                               uint32_t budget, int32_t* out_token) noexcept {
	return InvokeGpuApiBoundary(__func__, [&] {
		return SslmGpuSeqDecodeStepForG5BridgeImpl(ctx, seq, token, budget, out_token);
	});
}
SslmGpuStatus SslmGpuSeqPrefillSchemaContentForG5Bridge(
    SslmGpuContext* ctx, SslmGpuSequenceHandle* seq, const int32_t* tokens, int32_t count,
    uint32_t budget, int32_t* consumed) noexcept {
	return InvokeGpuApiBoundary(__func__, [&] {
		return SslmGpuSeqPrefillSchemaContentForG5BridgeImpl(
		    ctx, seq, tokens, count, budget, consumed);
	});
}
SslmGpuStatus sslm_gpu_model_hidden_size(const SslmGpuModelHandle* model,
                                         uint32_t* out_hidden_size) noexcept {
	return InvokeGpuApiBoundary(__func__, [&] {
		return sslm_gpu_model_hidden_sizeImpl(model, out_hidden_size);
	});
}
SslmGpuStatus sslm_gpu_seq_read_prefill_final_hidden(SslmGpuContext* ctx, SslmGpuSequenceHandle* seq,
                                                     int8_t* out_codes, size_t out_capacity,
                                                     size_t* out_required, int64_t* out_scale_m,
                                                     int64_t* out_scale_e) noexcept {
	return InvokeGpuApiBoundary(__func__, [&] {
		return sslm_gpu_seq_read_prefill_final_hiddenImpl(ctx, seq, out_codes, out_capacity,
		                                                   out_required, out_scale_m, out_scale_e);
	});
}

// -------------------------------------------------------------------------------------------------
// Paged KV (1.13.0), step G3: the pool, the budgeted create and the two diagnostics/seams of
// gpu_1p0.h's page-pool block, each through the boundary and with its out-handle cleared first.
// -------------------------------------------------------------------------------------------------
SslmGpuStatus sslm_gpu_kv_pool_create(SslmGpuContext* ctx, SslmGpuModelHandle* model, uint32_t page_count,
                                      SslmGpuKvPoolHandle** out_pool) noexcept {
	ClearOutputHandle(out_pool);
	return InvokeGpuApiBoundary(__func__, [&] { return sslm_gpu_kv_pool_createImpl(ctx, model, page_count, out_pool); });
}
SslmGpuStatus sslm_gpu_kv_pool_destroy(SslmGpuContext* ctx, SslmGpuKvPoolHandle* pool) noexcept {
	return InvokeGpuApiBoundary(__func__, [&] { return sslm_gpu_kv_pool_destroyImpl(ctx, pool); });
}
SslmGpuStatus sslm_gpu_kv_pool_max_page_count(SslmGpuContext* ctx, SslmGpuModelHandle* model, uint32_t* out_pages) noexcept {
	if (out_pages) *out_pages = 0;
	return InvokeGpuApiBoundary(__func__, [&] { return sslm_gpu_kv_pool_max_page_countImpl(ctx, model, out_pages); });
}
SslmGpuStatus sslm_gpu_seq_create_budgeted(SslmGpuContext* ctx, SslmGpuModelHandle* model, SslmGpuKvPoolHandle* pool,
                                           int32_t budget, SslmGpuSequenceHandle** out_seq) noexcept {
	ClearOutputHandle(out_seq);
	return InvokeGpuApiBoundary(
	    __func__, [&] { return sslm_gpu_seq_create_budgetedImpl(ctx, model, pool, budget, out_seq); });
}
SslmGpuStatus sslm_gpu_kv_pool_stats(const SslmGpuKvPoolHandle* pool, SslmGpuKvPoolStats* out) noexcept {
	return InvokeGpuApiBoundary(__func__, [&] { return sslm_gpu_kv_pool_statsImpl(pool, out); });
}
SslmGpuStatus sslm_gpu_test_only_peek_pool_mirror_bytes(const SslmGpuKvPoolHandle* pool, uint32_t page,
                                                        size_t offset, size_t len, uint8_t* out) noexcept {
	return InvokeGpuApiBoundary(
	    __func__, [&] { return sslm_gpu_test_only_peek_pool_mirror_bytesImpl(pool, page, offset, len, out); });
}

SslmGpuStatus sslm_gpu_test_only_wait_in_flight(SslmGpuContext* ctx, SslmGpuSequenceHandle* seq) noexcept {
	return InvokeGpuApiBoundary(__func__, [&] { return sslm_gpu_test_only_wait_in_flightImpl(ctx, seq); });
}

// -------------------------------------------------------------------------------------------------
// Paged KV (1.13.0), step G4: the prefix verbs and adopt, each through the boundary and with its
// out-handle cleared first.
// -------------------------------------------------------------------------------------------------
SslmGpuStatus sslm_gpu_prefix_begin(SslmGpuContext* ctx, SslmGpuModelHandle* model, SslmGpuKvPoolHandle* pool,
                                    int32_t budget, SslmGpuPrefixHandle** out_prefix) noexcept {
	ClearOutputHandle(out_prefix);
	return InvokeGpuApiBoundary(__func__, [&] { return sslm_gpu_prefix_beginImpl(ctx, model, pool, budget, out_prefix); });
}
SslmGpuStatus sslm_gpu_prefix_begin_from(SslmGpuContext* ctx, SslmGpuPrefixHandle* parent, int32_t budget,
                                         SslmGpuPrefixHandle** out_prefix) noexcept {
	ClearOutputHandle(out_prefix);
	return InvokeGpuApiBoundary(__func__, [&] { return sslm_gpu_prefix_begin_fromImpl(ctx, parent, budget, out_prefix); });
}
SslmGpuStatus sslm_gpu_prefix_set_schema(SslmGpuContext* ctx, SslmGpuPrefixHandle* prefix, int32_t schema_index) noexcept {
	return InvokeGpuApiBoundary(__func__, [&] { return sslm_gpu_prefix_set_schemaImpl(ctx, prefix, schema_index); });
}
SslmGpuStatus sslm_gpu_prefix_prefill(SslmGpuContext* ctx, SslmGpuPrefixHandle* prefix, const int32_t* tokens,
                                      int32_t count, int32_t kind, uint32_t dispatch_budget, int32_t* consumed) noexcept {
	if (consumed) *consumed = 0;
	return InvokeGpuApiBoundary(__func__, [&] {
		return sslm_gpu_prefix_prefillImpl(ctx, prefix, tokens, count, kind, dispatch_budget, consumed);
	});
}
SslmGpuStatus sslm_gpu_prefix_freeze(SslmGpuContext* ctx, SslmGpuPrefixHandle* prefix) noexcept {
	return InvokeGpuApiBoundary(__func__, [&] { return sslm_gpu_prefix_freezeImpl(ctx, prefix); });
}
SslmGpuStatus sslm_gpu_prefix_release(SslmGpuContext* ctx, SslmGpuPrefixHandle* prefix) noexcept {
	return InvokeGpuApiBoundary(__func__, [&] { return sslm_gpu_prefix_releaseImpl(ctx, prefix); });
}
SslmGpuStatus sslm_gpu_seq_adopt_prefix(SslmGpuContext* ctx, SslmGpuSequenceHandle* seq,
                                        SslmGpuPrefixHandle* prefix) noexcept {
	return InvokeGpuApiBoundary(__func__, [&] { return sslm_gpu_seq_adopt_prefixImpl(ctx, seq, prefix); });
}

// Paged KV (1.13.0), step G5: restore into a pool, through the boundary, its out-handle and shared-page
// count cleared first.
SslmGpuStatus sslm_gpu_seq_restore_into_pool(SslmGpuContext* ctx, SslmGpuModelHandle* model, SslmGpuKvPoolHandle* pool,
                                             const void* blob, size_t blob_size, SslmGpuPrefixHandle* prefix_or_null,
                                             SslmGpuSequenceHandle** out_seq, uint32_t* out_shared_pages) noexcept {
	ClearOutputHandle(out_seq);
	if (out_shared_pages) *out_shared_pages = 0;
	return InvokeGpuApiBoundary(__func__, [&] {
		return sslm_gpu_seq_restore_into_poolImpl(ctx, model, pool, blob, blob_size, prefix_or_null, out_seq,
		                                          out_shared_pages);
	});
}

// Precached prefixes (1.16.0): save and restore a frozen GPU prefix ('SPX1'), through the boundary.
SslmGpuStatus sslm_gpu_prefix_save(SslmGpuContext* ctx, const SslmGpuPrefixHandle* prefix, void* out_blob,
                                   size_t* out_blob_size) noexcept {
	return InvokeGpuApiBoundary(__func__, [&] { return sslm_gpu_prefix_saveImpl(ctx, prefix, out_blob, out_blob_size); });
}
SslmGpuStatus sslm_gpu_prefix_restore(SslmGpuContext* ctx, SslmGpuModelHandle* model, SslmGpuKvPoolHandle* pool,
                                      const void* blob, size_t blob_size, SslmGpuPrefixHandle* parent_or_null,
                                      SslmGpuPrefixHandle** out_prefix, uint32_t* out_shared_pages) noexcept {
	ClearOutputHandle(out_prefix);
	if (out_shared_pages) *out_shared_pages = 0;
	return InvokeGpuApiBoundary(__func__, [&] {
		return sslm_gpu_prefix_restoreImpl(ctx, model, pool, blob, blob_size, parent_or_null, out_prefix,
		                                   out_shared_pages);
	});
}
SslmGpuStatus sslm_gpu_test_only_prefix_table_entry(const SslmGpuPrefixHandle* prefix, uint32_t i, uint32_t* out_page,
                                                    uint32_t* out_mapped) noexcept {
	if (!prefix || prefix->seq == nullptr) return SSLM_SEQUENCE_KV_BUFFER_MISMATCH;
	return sslm_gpu_test_only_seq_table_entry(prefix->seq, i, out_page, out_mapped);
}

// G3 test seams the GPU red suite names (gpu_1p0_bench_bridge.h). Host-only reads and one flag;
// none submits, allocates or throws.
SslmGpuStatus sslm_gpu_test_only_seq_table_entry(const SslmGpuSequenceHandle* seq, uint32_t i, uint32_t* out_page,
                                                 uint32_t* out_mapped) noexcept {
	superslm_gpu::inflight::GpuApiCallScope scope(__func__);  // G3 (§4.4): every C ABI entry point
	if (!seq || !out_page || seq->pool == nullptr || seq->holder == nullptr) return SSLM_SEQUENCE_KV_BUFFER_MISMATCH;
	const uint32_t mapped = superslm::kv_pages::HolderMapped(seq->holder);
	if (out_mapped) *out_mapped = mapped;
	if (i >= mapped) return SSLM_SEQUENCE_KV_BUFFER_MISMATCH;
	*out_page = seq->holder->table[i];
	return SSLM_OK;
}
SslmGpuStatus sslm_gpu_test_only_arm_view_fault(SslmGpuSequenceHandle* seq, uint32_t fault) noexcept {
	superslm_gpu::inflight::GpuApiCallScope scope(__func__);  // G3 (§4.4): every C ABI entry point
	if (!seq || seq->pool == nullptr || fault > 4) return SSLM_SEQUENCE_KV_BUFFER_MISMATCH;
	seq->armed_view_fault = fault;
	return SSLM_OK;
}
SslmGpuStatus sslm_gpu_test_only_last_ladder_status(const SslmGpuSequenceHandle* seq,
                                                    int32_t* out_forward_status) noexcept {
	superslm_gpu::inflight::GpuApiCallScope scope(__func__);  // G3 (§4.4): every C ABI entry point
	if (!seq || !out_forward_status) return SSLM_SEQUENCE_KV_BUFFER_MISMATCH;
	*out_forward_status = static_cast<int32_t>(seq->last_ladder_status);
	return SSLM_OK;
}

// -------------------------------------------------------------------------------------------------
// T-2851 test-build seams (SUPERSLM_GPU_ALLOC_FAULT_INJECTION only; design Sec4.6 items 2 and 4,
// Sec4.9). Global scope and C++ linkage, matching the red suite's own local declarations
// (tests/t2956-token-finish-red-suite/) and gpu_port.h's declarations. None is in a product build.
// -------------------------------------------------------------------------------------------------
#if defined(SUPERSLM_GPU_ALLOC_FAULT_INJECTION)

void SslmGpuAllocCounterResetForTest() noexcept {
	superslm_gpu::harness::GpuTestSeamState& seam = superslm_gpu::harness::TestSeamState();
	std::lock_guard<std::mutex> lock(seam.mutex);
	seam.alloc_count = 0;
	seam.alloc_in_bundle.clear();
	seam.alloc_bytes.clear();
	seam.alloc_heap.clear();
}

// Paged KV (G3, cell 11.4): the k-th counted allocation's size and heap.
bool SslmGpuAllocRecordForTest(uint32_t k, uint64_t* out_bytes, uint32_t* out_heap_type) noexcept {
	superslm_gpu::harness::GpuTestSeamState& seam = superslm_gpu::harness::TestSeamState();
	std::lock_guard<std::mutex> lock(seam.mutex);
	if (k < 1 || k > seam.alloc_bytes.size()) return false;
	if (out_bytes) *out_bytes = seam.alloc_bytes[k - 1];
	if (out_heap_type) *out_heap_type = seam.alloc_heap[k - 1];
	return true;
}

// Paged KV (G3, cell 11.4 leg 3): Close faults inside the pool create's zero-fill.
void ArmGpuPoolZeroFillCloseFaultForTest(uint32_t consecutive_failures) noexcept {
	superslm_gpu::harness::GpuTestSeamState& seam = superslm_gpu::harness::TestSeamState();
	std::lock_guard<std::mutex> lock(seam.mutex);
	seam.pool_zero_fill_close_failures = consecutive_failures;  // 0 disarms
}

// Precached prefixes: the restore fault slot of the calling thread (cells 5.11 and 5.7, GPU halves).
// Arming clears the fired flag; position 0 disarms.
void ArmGpuPrefixRestoreFaultForTest(int position, int share_refusal) noexcept {
	g_gpu_prefix_restore_fault = GpuPrefixRestoreFault{};
	g_gpu_prefix_restore_fault.position = position;
	g_gpu_prefix_restore_fault.share_refusal = share_refusal;
}
bool GpuPrefixRestoreFaultFiredForTest(uint32_t* out_shared_at_fire) noexcept {
	if (out_shared_at_fire) *out_shared_at_fire = g_gpu_prefix_restore_fault.shared_at_fire;
	return g_gpu_prefix_restore_fault.fired;
}
// review-g23 F2's pin: a refcount underflow at the pooled sequence's release.
bool ZeroGpuPooledSeqReservePageRefcountForTest(SslmGpuSequenceHandle* seq) noexcept {
	if (seq == nullptr || seq->pool == nullptr || seq->holder == nullptr || seq->holder->reserve.empty()) return false;
	superslm::kv_pages::PagePool* pages = seq->pool->pages;
	std::lock_guard<superslm::kv_pages::CountingMutex> lock(pages->mutex);
	pages->refcount[seq->holder->reserve.front()] = 0;
	return true;
}

uint32_t SslmGpuAllocCountForTest() noexcept {
	superslm_gpu::harness::GpuTestSeamState& seam = superslm_gpu::harness::TestSeamState();
	std::lock_guard<std::mutex> lock(seam.mutex);
	return seam.alloc_count;
}

void ArmGpuAllocFaultAtOccurrence(uint32_t k, HRESULT hr) noexcept {
	superslm_gpu::harness::GpuTestSeamState& seam = superslm_gpu::harness::TestSeamState();
	std::lock_guard<std::mutex> lock(seam.mutex);
	seam.alloc_fault_countdown = k;  // 0 disarms
	seam.alloc_fault_hr = hr;
}

bool SslmGpuAllocInBundleForTest(uint32_t k) noexcept {
	superslm_gpu::harness::GpuTestSeamState& seam = superslm_gpu::harness::TestSeamState();
	std::lock_guard<std::mutex> lock(seam.mutex);
	return k >= 1 && k <= seam.alloc_in_bundle.size() && seam.alloc_in_bundle[k - 1];
}

void ArmGpuMapDeviceRemovedQueryInjection() noexcept {
	superslm_gpu::harness::GpuTestSeamState& seam = superslm_gpu::harness::TestSeamState();
	std::lock_guard<std::mutex> lock(seam.mutex);
	seam.map_removed_query = true;
}

void ArmGpuDeviceLogitsReadbackOverride(int32_t row, int64_t value) noexcept {
	superslm_gpu::harness::GpuTestSeamState& seam = superslm_gpu::harness::TestSeamState();
	std::lock_guard<std::mutex> lock(seam.mutex);
	seam.readback_override_armed = true;
	seam.readback_override_row = row;
	seam.readback_override_value = value;
}

void ArmGpuDeviceLogitsCloseFaultForTest(uint32_t consecutive_failures) noexcept {
	superslm_gpu::harness::GpuTestSeamState& seam = superslm_gpu::harness::TestSeamState();
	std::lock_guard<std::mutex> lock(seam.mutex);
	seam.close_failures = consecutive_failures;  // 0 disarms
}

size_t SslmGpuHeadWeightCopySizeForTest(const SslmGpuModelHandle* model) noexcept {
	return model ? model->head_weights.size() : 0;
}

struct SslmGpuDeviceLogitsBundleForTest {
	DeviceLogitsBuffers buffers;
};

SslmGpuStatus SslmGpuDeviceLogitsBundleCreateForTest(
    SslmGpuContext* ctx, const int8_t* head_rows, uint32_t V, uint32_t H,
    SslmGpuDeviceLogitsBundleForTest** out_bundle) noexcept {
	if (!out_bundle) return SSLM_SEQUENCE_KV_BUFFER_MISMATCH;
	*out_bundle = nullptr;
	if (!ctx || !head_rows || V == 0 || H == 0) return SSLM_SEQUENCE_KV_BUFFER_MISMATCH;
	return InvokeGpuInternalBoundary(__func__, [&]() -> SslmGpuStatus {
		auto bundle = std::make_unique<SslmGpuDeviceLogitsBundleForTest>();
		const SslmGpuStatus st = CreateDeviceLogitsBuffers(ctx, head_rows, V, H, &bundle->buffers);
		if (st == SSLM_OK) *out_bundle = bundle.release();
		return st;
	});
}

SslmGpuStatus SslmGpuDeviceLogitsRunForTest(SslmGpuContext* ctx,
                                            SslmGpuDeviceLogitsBundleForTest* bundle,
                                            const int8_t* x_codes, int64_t* wide_out) noexcept {
	if (!ctx || !bundle || !x_codes || !wide_out) return SSLM_SEQUENCE_KV_BUFFER_MISMATCH;
	return InvokeGpuInternalBoundary(
	    __func__, [&] { return RunDeviceLogits(ctx, bundle->buffers, x_codes, wide_out); });
}

void SslmGpuDeviceLogitsBundleResourcesForTest(const SslmGpuDeviceLogitsBundleForTest* bundle,
                                               uint64_t out_gpu_vas[5]) noexcept {
	const DeviceLogitsBuffers& b = bundle->buffers;
	ID3D12Resource* resources[5] = {b.head.Get(), b.x_upload.Get(), b.x.Get(), b.out.Get(),
	                                b.readback.Get()};
	for (int i = 0; i < 5; ++i) {
		out_gpu_vas[i] = resources[i] ? resources[i]->GetGPUVirtualAddress() : 0;
	}
}

void SslmGpuDeviceLogitsBundleDestroyForTest(SslmGpuDeviceLogitsBundleForTest* bundle) noexcept {
	delete bundle;
}

#endif  // SUPERSLM_GPU_ALLOC_FAULT_INJECTION
