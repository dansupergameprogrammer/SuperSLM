// Paged KV, GPU slice (SuperSLM 1.13.0), plan step G1 -- group g1d's shared harness.
//
// Cells: plan.md Sec7 3.6e, 3.6f, 3.6g, 3.12, and the stranded pins of Sec4.4's fault-path table
// (*The ledger on the fault paths*), as far as they are new cells. 3.6h is not this group's.
//
// Conventions reused, not invented: CHECK/CHECK_MSG/SKIP_MSG, FinishSuite, StatusName, GpuModelFixture,
// Drain and CaptureState are tests/t2791-gpu-prefill-read-red-suite/fixture_common.h's (included
// below); the phase watchdog is t2791's cell_prompt_guard_removal.cpp:20/:41 (the precedent Sec7 3.6e
// names); the D3D12 call hooks are te425's (te425_harness.h, InstallHooks at te425_cells.cpp:312) and
// are pulled in only by the cells that need them (G1D_WITH_TE425_HOOKS).
//
// HOW THESE CELLS ARE RED BEFORE G3. Every cell calls at least one name that G3 (or G4/G5) defines and
// that has no definition at the 63e7b0e base: the 14 arc verbs declared in include/superslm/gpu_1p0.h
// (link-red, t2791's documented convention), and the G3 internal-layer seams Sec4.4 names, declared
// below exactly as Sec4.4 spells them (ArmQueueGateForTest, ReleaseQueueGateForTest, DrainSubmissionsGpu).
// Each cell's header names its own unresolved symbols. Once a G3 header declares a seam, the
// redeclaration here must match it; a drift is a compile error, which is the intended pin (define
// G1D_G3_SEAMS_DECLARED_BY_HEADER to drop the redeclarations if G3 moves them into gpu_port.h with a
// different exception specification, and record why).
//
// GRADING (Sec7's three rules, the grading rule's seven sources). Every verdict here reads only:
// public statuses; tokens compared with an unsaved twin; the debug-layer message count read from
// the device's info queue (harness::Device::DrainDebugLayerMessages); the status of a public verb
// called afterwards; the type of an exception that reaches an internal-layer caller; and, for the
// hooked legs, te425's test-owned hook counters (g_arm.fired, g_calls[]), which the test's own hook
// increments (3.6d's first-touch control reads g_arm.fired the same way). No ledger or count value
// is read.
#ifndef SSLM_G1D_HELPERS_H
#define SSLM_G1D_HELPERS_H

#include "fixture_common.h"  // tests/t2791-gpu-prefill-read-red-suite (CHECK, StatusName, GpuModelFixture, ...)
#include "d3d12_harness.h"   // superslm_gpu::harness::GetDevice(), GpuSubmissionStrandedError

#include <atomic>
#include <chrono>
#include <cstdarg>
#include <cstdlib>
#include <functional>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <tuple>
#include <type_traits>
#include <typeinfo>
#include <utility>
#include <vector>

#if defined(G1D_WITH_TE425_HOOKS)
#include "te425_harness.h"  // tests/te425-gpu-alloc-status-red-suite: HookArmCall, g_arm, g_calls, M_SIGNAL, ...
#endif

// ---------------------------------------------------------------------------------------------
// The G3 internal-layer seams, as Sec4.4 names them (*The submission ledger*, *The queue gate*).
// ---------------------------------------------------------------------------------------------
#if !defined(G1D_G3_SEAMS_DECLARED_BY_HEADER)
namespace superslm_gpu {
// "Held and released through ArmQueueGateForTest() and ReleaseQueueGateForTest() in gpu_port.h, beside
// the tree's other arming seams" (whose spelling is `void ArmXxx();` in this namespace). Fault-injection
// builds only (both: the allocation-fault build and the T2169 chunk-recording build). The singleton only.
// Release is idempotent.
void ArmQueueGateForTest();
void ReleaseQueueGateForTest();
// "Its signature is `void DrainSubmissionsGpu();`, in namespace superslm_gpu".
void DrainSubmissionsGpu();
}  // namespace superslm_gpu
#endif

namespace g1d {

// ---------------------------------------------------------------------------------------------
// Options: --key=value, read once.
// ---------------------------------------------------------------------------------------------
inline std::vector<std::pair<std::string, std::string>>& Opts() {
	static std::vector<std::pair<std::string, std::string>> o;
	return o;
}
inline void ParseOpts(int argc, char** argv) {
	ParseFixtureArgs(argc, argv);  // t2791's own flags (--allow-skip among them)
	for (int i = 1; i < argc; ++i) {
		const std::string a = argv[i];
		if (a.rfind("--", 0) != 0) continue;
		const size_t eq = a.find('=');
		if (eq == std::string::npos) Opts().emplace_back(a.substr(2), "1");
		else Opts().emplace_back(a.substr(2, eq - 2), a.substr(eq + 1));
	}
}
inline std::string Opt(const char* k, const char* d = "") {
	for (const auto& kv : Opts())
		if (kv.first == k) return kv.second;
	return d;
}
inline bool HasOpt(const char* k) {
	for (const auto& kv : Opts())
		if (kv.first == k) return true;
	return false;
}

// ---------------------------------------------------------------------------------------------
// The watchdog (cell_prompt_guard_removal.cpp:20, :41). Sec7 3.6e: "Every run has a wall-clock timeout,
// and a timeout fails the cell." A phase that does not return within the limit exits 7. Before the
// exit it releases the queue gate (Sec4.4 *The queue gate*: "no process exits with the gate held").
// ---------------------------------------------------------------------------------------------
inline std::atomic<int>& Phase() {
	static std::atomic<int> p{0};
	return p;
}
inline std::atomic<const char*>& PhaseName() {
	static std::atomic<const char*> n{"setup"};
	return n;
}
inline std::atomic<bool>& GateMayBeHeld() {
	static std::atomic<bool> g{false};
	return g;
}
constexpr int kPhaseDone = -1;
inline void SetPhase(const char* name) {
	PhaseName().store(name);
	Phase().fetch_add(1);
	std::printf("  -- phase: %s\n", name);
	std::fflush(stdout);
}
inline void PhaseDone() { Phase().store(kPhaseDone); }

// Under the debug layer (SSLM_GPU_ENABLE_DEBUG_LAYER=1, which also turns on GPU-based validation) every
// submission runs many times slower, so a limit sized for a plain run (3.12 asks for 120 s) fires on a run
// that is only slow. The limit is scaled by kDebugLayerWatchdogScale there and kept as given with the layer
// off, so a real hang is still caught quickly on a plain run.
constexpr int kDebugLayerWatchdogScale = 10;
inline int WatchdogSeconds(int seconds) {
	const char* v = std::getenv("SSLM_GPU_ENABLE_DEBUG_LAYER");
	return (v != nullptr && v[0] == '1') ? seconds * kDebugLayerWatchdogScale : seconds;
}

inline void StartWatchdog(int requested_seconds) {
	const int seconds = WatchdogSeconds(requested_seconds);
	std::thread([seconds] {
		int last = Phase().load();
		auto t0 = std::chrono::steady_clock::now();
		for (;;) {
			std::this_thread::sleep_for(std::chrono::milliseconds(200));
			const int p = Phase().load();
			if (p == kPhaseDone) return;
			if (p != last) {
				last = p;
				t0 = std::chrono::steady_clock::now();
			}
			if (std::chrono::steady_clock::now() - t0 > std::chrono::seconds(seconds)) {
				std::printf("FAIL: HANG -- phase '%s' did not return within %d s (a timeout fails the cell)\n",
				            PhaseName().load(), seconds);
				std::fflush(stdout);
#if !defined(G1D_NO_QUEUE_GATE)
				if (GateMayBeHeld().load()) superslm_gpu::ReleaseQueueGateForTest();
#endif
				std::_Exit(7);
			}
		}
	}).detach();
}

// ---------------------------------------------------------------------------------------------
// The queue gate, RAII. Released in the destructor on every exit (stranded runs included), and
// releasable early. Release is idempotent at the seam (Sec4.4), and here too.
// ---------------------------------------------------------------------------------------------
#if !defined(G1D_NO_QUEUE_GATE)
class QueueGate {
public:
	QueueGate() = default;
	QueueGate(const QueueGate&) = delete;
	QueueGate& operator=(const QueueGate&) = delete;
	~QueueGate() { Release(); }
	void Arm() {
		GateMayBeHeld().store(true);
		superslm_gpu::ArmQueueGateForTest();
		held_ = true;
	}
	void Release() {
		if (held_) superslm_gpu::ReleaseQueueGateForTest();
		held_ = false;
	}
	bool Held() const { return held_; }

private:
	bool held_ = false;
};
#endif

// ---------------------------------------------------------------------------------------------
// The exception channel of an internal-layer call (Sec7's grading source "the type of an exception that
// reaches an internal-layer caller"). PoolSubmitContractError is a G3 type (a std::logic_error, Sec4.4
// *The reset-site assert*) whose header the plan does not name, so it is matched by its dynamic type's
// name under a std::logic_error catch: both its base and its name are pinned, and the cell compiles
// before G3. GpuSubmissionStrandedError exists at the base (d3d12_harness.h:84) and is caught by type.
// ---------------------------------------------------------------------------------------------
enum class Thrown { None, PoolSubmitContract, Stranded, OtherLogic, OtherException, NonStd };
inline const char* ThrownName(Thrown t) {
	switch (t) {
		case Thrown::None: return "no exception";
		case Thrown::PoolSubmitContract: return "PoolSubmitContractError";
		case Thrown::Stranded: return "GpuSubmissionStrandedError";
		case Thrown::OtherLogic: return "another std::logic_error";
		case Thrown::OtherException: return "another std::exception";
		case Thrown::NonStd: return "a non-std exception";
	}
	return "?";
}
template <class F>
Thrown CallInternal(F&& f, std::string* what = nullptr) {
	try {
		f();
		return Thrown::None;
	} catch (const superslm_gpu::harness::GpuSubmissionStrandedError& e) {
		if (what) *what = e.what();
		return Thrown::Stranded;
	} catch (const std::logic_error& e) {
		if (what) *what = std::string(typeid(e).name()) + ": " + e.what();
		return std::string(typeid(e).name()).find("PoolSubmitContractError") != std::string::npos
		           ? Thrown::PoolSubmitContract
		           : Thrown::OtherLogic;
	} catch (const std::exception& e) {
		if (what) *what = std::string(typeid(e).name()) + ": " + e.what();
		return Thrown::OtherException;
	} catch (...) {
		if (what) *what = "(non-std)";
		return Thrown::NonStd;
	}
}

// ---------------------------------------------------------------------------------------------
// The debug layer (Sec7 grading source: "the debug-layer message count, read by the test from the
// device's info queue"). A cell whose verdict rests on "the debug layer is clean" refuses to run
// without the layer: a disabled layer returns 0 for everything and would pass vacuously.
//
// A state-tracking build (SUPERSLM_GPU_STATE_TRACKING) counts the host-side resource-state tracker's
// violations (src/gpu/gpu_resource_state_tracker.h) with the layer's messages and runs without the layer:
// on the software GPU, which has no D3D12 validation, the tracker stands in for the layer's resource-state
// class (barrier StateBefore, copy and dispatch states), which is the class 3.12 grades, and nothing else.
// ---------------------------------------------------------------------------------------------
inline bool RequireDebugLayer() {
	const bool layer = superslm_gpu::harness::GetDevice().debug_layer_enabled;
	const bool on = layer || superslm_gpu::harness::kStateTrackingBuild;
	if (!layer && on) std::printf("  note: no D3D12 debug layer; the host resource-state tracker is the count\n");
	CHECK_MSG(on, "SETUP: the D3D12 debug layer is off -- run with SSLM_GPU_ENABLE_DEBUG_LAYER=1; a clean-debug-layer "
	              "verdict without it is vacuous");
	return on;
}
inline size_t DebugMessages() {
	static uint64_t tracked_seen = 0;
	const uint64_t tracked_total = superslm_gpu::harness::StateTrackingViolations().Total();
	const size_t tracked = static_cast<size_t>(tracked_total - tracked_seen);
	tracked_seen = tracked_total;
	return superslm_gpu::harness::GetDevice().DrainDebugLayerMessages() + tracked;
}

// ---------------------------------------------------------------------------------------------
// The in-flight token's type, read off RunLayerLoopGpuSubmit's out-parameter (its 15th parameter), so
// one source compiles at the base (GpuLayerLoopInFlight*, a raw pointer) and at G3 (GpuInFlightToken, the
// draining move-only token of Sec4.4 *The finish made structural*).
// ---------------------------------------------------------------------------------------------
template <class F>
struct FnArgs;
template <class R, class... A>
struct FnArgs<R (*)(A...)> {
	using tuple = std::tuple<A...>;
};
template <class R, class... A>
struct FnArgs<R (*)(A...) noexcept> {
	using tuple = std::tuple<A...>;
};
using SubmitOutParam = std::tuple_element_t<14, FnArgs<decltype(&superslm_gpu::RunLayerLoopGpuSubmit)>::tuple>;
using InFlightToken = std::remove_pointer_t<SubmitOutParam>;

// ---------------------------------------------------------------------------------------------
// One context with one mapped model (t2791's GpuModelFixture), plus the sequences a cell makes.
// ---------------------------------------------------------------------------------------------
struct Bench {
	SslmGpuContext* ctx = nullptr;
	GpuModelFixture fx;
	std::vector<SslmGpuSequenceHandle*> seqs;
	std::vector<SslmGpuKvPoolHandle*> pools;
	std::vector<SslmGpuPrefixHandle*> prefixes;

	bool Open(const std::string& model_path, uint32_t residency_flags = 0) {
		if (sslm_gpu_context_create(GpuContextConfig{}, &ctx) != SSLM_OK || !ctx) {
			std::printf("SETUP: sslm_gpu_context_create failed\n");
			return false;
		}
		if (residency_flags == 0) return fx.Open(model_path, ctx);
		// GpuModelFixture maps with flags 0; a device-resident head maps here, the same way.
		if (!fx.Open(model_path, ctx)) return false;
		sslm_gpu_model_unmap(ctx, fx.model);
		fx.model = nullptr;
		GpuResidencyConfig rc{};
		rc.flags = residency_flags;
		if (sslm_gpu_model_map(ctx, &fx.view, rc, &fx.model) != SSLM_OK || !fx.model) {
			std::printf("SETUP: sslm_gpu_model_map(flags=0x%x) failed\n", residency_flags);
			return false;
		}
		return true;
	}
	// Teardown reaches the pool and prefix verbs only in a cell that opts in, so a [G3] cell that makes no
	// pool does not link against a G3 or G4 verb it never calls (G1D_USES_POOLS: G3's pool destroy;
	// G1D_USES_PREFIXES: G4's prefix release).
	void Close() {
		for (auto* s : seqs) sslm_gpu_seq_release(ctx, s);
#if defined(G1D_USES_PREFIXES)
		for (auto* p : prefixes) sslm_gpu_prefix_release(ctx, p);
#endif
#if defined(G1D_USES_POOLS) || defined(G1D_USES_PREFIXES)
		for (auto* p : pools) sslm_gpu_kv_pool_destroy(ctx, p);
#endif
		seqs.clear();
		prefixes.clear();
		pools.clear();
		fx.Close();
		if (ctx) sslm_gpu_context_destroy(ctx);
		ctx = nullptr;
	}
	SslmGpuSequenceHandle* Dedicated(int64_t cap = 64) {
		SslmGpuSequenceHandle* s = nullptr;
		if (sslm_gpu_seq_create(ctx, fx.model, cap, &s) != SSLM_OK || !s) return nullptr;
		seqs.push_back(s);
		return s;
	}
	SslmGpuKvPoolHandle* Pool(uint32_t pages) {
		SslmGpuKvPoolHandle* p = nullptr;
		if (sslm_gpu_kv_pool_create(ctx, fx.model, pages, &p) != SSLM_OK || !p) return nullptr;
		pools.push_back(p);
		return p;
	}
	SslmGpuSequenceHandle* Budgeted(SslmGpuKvPoolHandle* pool, int32_t budget) {
		SslmGpuSequenceHandle* s = nullptr;
		if (sslm_gpu_seq_create_budgeted(ctx, fx.model, pool, budget, &s) != SSLM_OK || !s) return nullptr;
		seqs.push_back(s);
		return s;
	}
	void Forget(SslmGpuSequenceHandle* s) {
		for (size_t i = 0; i < seqs.size(); ++i)
			if (seqs[i] == s) {
				seqs.erase(seqs.begin() + static_cast<std::ptrdiff_t>(i));
				return;
			}
	}
	uint32_t FullDepthBudget() const { return 0xFFFFFFFFu; }  // te425's DecodeStep: one call, full depth
};

// The two-sided ready: the call's own status, then the decoded out_status.
struct ReadyOut {
	SslmGpuStatus ret = SSLM_DEVICE_LOST;
	int32_t ready = -1;
	SslmGpuStatus out = SSLM_DEVICE_LOST;
	SslmGpuStatus First() const { return ret != SSLM_OK ? ret : out; }
};
inline ReadyOut Ready(SslmGpuContext* ctx, SslmGpuSequenceHandle* s, int32_t block) {
	ReadyOut r;
	r.out = SSLM_OK;
	r.ret = sslm_gpu_ready(ctx, s, block, &r.ready, &r.out);
	return r;
}

// Embed `token` and submit one full-depth decode step (te425's DecodeStep, without its ready).
inline SslmGpuStatus SubmitDecode(Bench& b, SslmGpuSequenceHandle* s, int32_t token) {
	const SslmGpuStatus e = sslm_gpu_seq_embed_token(b.ctx, s, token);
	if (e != SSLM_OK) return e;
	return sslm_decode_step_gpu(b.ctx, s, nullptr, b.FullDepthBudget());
}

// n greedy tokens through the recommended one-call entry point, starting from `first`. A token list is
// the byte-equality operand (Sec7 grading: tokens compared with an unsaved twin).
inline std::vector<int32_t> NextTokens(Bench& b, SslmGpuSequenceHandle* s, int32_t first, int n,
                                       SslmGpuStatus* bad = nullptr) {
	std::vector<int32_t> out;
	int32_t tok = first;
	for (int i = 0; i < n; ++i) {
		int32_t next = -1;
		const SslmGpuStatus st = SslmGpuSeqDecodeStepForG5Bridge(b.ctx, s, tok, b.FullDepthBudget(), &next);
		if (st != SSLM_OK) {
			if (bad) *bad = st;
			out.push_back(-1000 - static_cast<int32_t>(st));
			return out;
		}
		out.push_back(next);
		tok = next;
	}
	return out;
}
inline std::string Join(const std::vector<int32_t>& v) {
	std::string s;
	for (size_t i = 0; i < v.size(); ++i) s += (i ? "," : "") + std::to_string(v[i]);
	return s;
}

inline std::vector<uint8_t> SaveBlob(SslmGpuContext* ctx, const SslmGpuSequenceHandle* s) {
	size_t n = 0;
	(void)sslm_gpu_seq_save(ctx, s, nullptr, &n);
	std::vector<uint8_t> b(n);
	if (!n || sslm_gpu_seq_save(ctx, s, b.data(), &n) != SSLM_OK) b.clear();
	return b;
}

// ---------------------------------------------------------------------------------------------
// The internal layer, driven directly (gpu_port.h): the four functions Sec7 3.6e names, each reaching
// one singleton reset site, and RunLayerLoopGpuSubmit/Finish for the token legs. A caller's own
// SequenceLayerState and workspace, as tools/t2113_b3_sequence_smoke.cpp drives them. RunLayerLoopGpu
// has no q_width parameter (its ~40 callers are square), so the model must be square
// (hidden == heads * head_dim): the R15 artifact (Qwen2.5-1.5B) is; Qwen3-0.6B is not.
// ---------------------------------------------------------------------------------------------
struct InternalLoop {
	const superslm::SslmModelView* view = nullptr;
	std::vector<superslm_marshal::LayerBacking> backings;
	std::vector<superslm::LayerWeights> layers;
	std::vector<int8_t> embed_codes, codes;
	superslm::CarriedScale embed_scale{};
	superslm::SequenceLayerState seq;
	std::vector<uint8_t> ws;
	uint32_t L = 0;
	size_t hidden = 0, head_dim = 0, kvh = 0, inter = 0;
	int64_t cap = 64;
	bool qk_norm = false;

	bool Build(const superslm::SslmModelView& v, bool has_qk_norm) {
		view = &v;
		qk_norm = has_qk_norm;
		const uint32_t heads = v.config.num_attention_heads;
		kvh = v.config.num_key_value_heads;
		L = v.config.num_hidden_layers;
		hidden = v.config.hidden_size;
		head_dim = v.config.head_dim;
		inter = v.config.intermediate_size;
		if (static_cast<size_t>(heads) * head_dim != hidden) {
			std::printf("SETUP: the internal-layer legs need a square model (hidden %zu != heads %u * head_dim %zu); "
			            "pass the R15 artifact\n",
			            hidden, heads, head_dim);
			return false;
		}
		backings.resize(L);
		layers.resize(L);
		for (uint32_t l = 0; l < L; ++l) {
			std::string err;
			if (!superslm_marshal::MarshalLayer(v, l, heads, static_cast<uint32_t>(kvh), backings[l], layers[l], &err)) {
				std::printf("SETUP: MarshalLayer(%u): %s\n", l, err.c_str());
				return false;
			}
		}
		const superslm::SslmTensorView* e = v.weights.Tensor("embed");
		bool ok = e != nullptr;
		const superslm::CarriedScale k = superslm_marshal::ReadCarriedScale(v.composition_constants, "embed", &ok);
		if (!ok) return false;
		embed_codes.assign(hidden, 0);
		if (superslm::EmbedEntry(0, v.config.vocab_size, reinterpret_cast<const int8_t*>(e->data), hidden, k,
		                         embed_codes.data(), &embed_scale) != superslm::SslmForwardStatus::Ok)
			return false;
		codes.assign(hidden, 0);
		ws.assign(static_cast<size_t>(L) * static_cast<size_t>(cap) * kvh * head_dim * 2, 0);
		Fresh();
		return true;
	}
	void Fresh() {
		std::copy(embed_codes.begin(), embed_codes.end(), codes.begin());
		seq = superslm::SequenceLayerState{};
		seq.hidden_codes = codes.data();
		seq.hidden_scale = embed_scale;
	}
	// One synchronous layer loop: the chunk-open reset (superslm_gpu.cpp:2544 on the tree, G0 Sec1a).
	superslm::SslmForwardStatus RunSync() {
		Fresh();
		return superslm_gpu::RunLayerLoopGpu(seq, layers.data(), L, L, hidden, head_dim, kvh, inter, cap,
		                                     view->rope_tables, ws.data(), ws.size(), nullptr, nullptr,
		                                     /*model_generation=*/1, qk_norm);
	}
	// The asynchronous submit: the token is handed out, unfenced.
	superslm::SslmForwardStatus Submit(InFlightToken* out) {
		Fresh();
		return superslm_gpu::RunLayerLoopGpuSubmit(seq, layers.data(), L, L, hidden, head_dim, kvh, inter, cap,
		                                           view->rope_tables, ws.data(), ws.size(), nullptr, nullptr, out,
		                                           nullptr, nullptr, nullptr, false, 0, 0, nullptr, 0, nullptr, 0,
		                                           /*model_generation=*/1, qk_norm);
	}
	superslm::SslmForwardStatus Finish(InFlightToken&& tok, int32_t block, int32_t* ready) {
		return superslm_gpu::RunLayerLoopGpuFinish(std::move(tok), seq, ws.data(), block, ready);
	}
};

// DispatchOne's reset (d3d12_harness.h): one primitive dispatch on the singleton.
inline void CallDispatchOneSite() { (void)superslm_gpu::DynamicScaleReciprocalGpu(12345); }
// The descriptor-bind reset (superslm_gpu.cpp:4895 on the tree): a synthetic two-array fixture.
inline void CallBindSite() {
	static const int8_t a0[16] = {1, -2, 3, -4, 5, -6, 7, -8, 9, -10, 11, -12, 13, -14, 15, -16};
	static const int8_t a1[16] = {-1, 2, -3, 4, -5, 6, -7, 8, -9, 10, -11, 12, -13, 14, -15, 16};
	const int8_t* arrays[2] = {a0, a1};
	const size_t counts[2] = {16, 16};
	int8_t out[32] = {};
	(void)superslm_gpu::BindDescriptorTableAndReadback(arrays, counts, 2, out, sizeof out);
}
// The restore reset (superslm_gpu.cpp:4750 on the tree): a dedicated sequence's own saved blob.
inline void CallRestoreSite(const std::vector<uint8_t>& blob, size_t hidden) {
	uint64_t ws_size = 0;
	if (!superslm_gpu::PeekGpuSeqBlobWorkspaceSize(blob.data(), blob.size(), &ws_size))
		throw std::runtime_error("g1d: restore leg's blob has no readable workspace size");
	std::vector<int8_t> codes(hidden, 0);
	std::vector<uint8_t> ws(static_cast<size_t>(ws_size), 0);
	superslm::SequenceLayerState out;
	out.hidden_codes = codes.data();
	int32_t schema = -1;
	uint32_t walk = 0;
	bool ready = false;
	(void)superslm_gpu::RestoreGpuSequenceState(blob.data(), blob.size(), &out, hidden, ws.data(), ws.size(), &schema,
	                                            &walk, &ready);
}

// A token run of `n` ids valid for this vocabulary, starting at `base` (t2791's GpuModelFixture::Run).
inline std::vector<int32_t> Tokens(const GpuModelFixture& fx, size_t n, int32_t base) { return fx.Run(n, base); }

}  // namespace g1d

#endif  // SSLM_G1D_HELPERS_H
