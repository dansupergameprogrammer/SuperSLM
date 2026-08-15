// T-2113 (B8): the bench proof for design Sec10 B8's own delivery -- thread-safety across
// disjoint sequences (design Sec5.4): two threads, two sequence handles, one context,
// concurrent sslm_decode_step_gpu calls, externally serialized only at the queue-submit
// boundary the design documents.
//
// Design Sec10 B8's own gate: "a concurrent two-thread run against two independent sequences
// produces the same per-sequence output as the equivalent serial run, run under a thread
// sanitizer where the toolchain supports it (Sec11 dim 3)." Realized here as the STRONGER claim
// every other B-section's own bench tool uses: per-step bit-equality against the CPU oracle
// (RunLayerLoop), not merely GPU self-consistency -- since the CPU oracle IS what a serial GPU
// run already matches (every prior B-section's own Gate 1), matching it under concurrency
// subsumes "matches the equivalent serial run."
//
// THE MUTEX, AND WHY IT IS HERE AND NOT IN PRODUCTION CODE: design Sec5.4/Sec13 (see
// src/gpu/gpu_1p0.cpp's own SslmGpuContext comment for the full, grounded account) deliberately
// does NOT build an internal queue-level lock -- "a caller driving two threads' decode calls
// through one context's queue concurrently must serialize the submit call itself." Grounded by
// execution, not merely quoted: the decode dispatch path (RunLayerLoopGpuSubmit/Finish) still
// routes through the pre-1.0 substrate's PROCESS-WIDE harness::GetDevice() singleton, not this
// context's own `device` -- so two genuinely concurrent, unsynchronized calls into it race the
// identical `ID3D12CommandAllocator::Reset()`-while-executing hazard the B7 undrained-batch hang
// already proved by execution (D-SLM3384, Claude/Brunel/t2113-1p0-core-build-2026-08-15.md
// Sec12.2 -- that hang IS this hazard's sequential-but-undrained shape; two real OS threads
// racing the same Reset() is the stronger, genuinely-concurrent form of it). This tool does NOT
// re-run an unlocked reproduction (real risk: an indefinite device-level stall, exactly what
// D-SLM3384 measured at 12+ CPU-seconds motionless) -- that finding is already-executed,
// git-tracked evidence for why the lock below is necessary, satisfying StandardsDocument.md
// Sec5.4's instrument-commissioning rule without re-triggering a hang this session does not need
// to reproduce a second time. Instead: a std::mutex EXTERNAL to the 1.0 API (this file's own
// g_submit_mutex) wraps EXACTLY the submit-and-drain call pair, proving that with the documented
// external-serialization discipline followed, two threads driving disjoint sequences produce
// output bit-identical to the CPU oracle -- the design's own claim, verified rather than assumed.
//
// Also proves the dim1 x dim3 crossed cell (design Sec11 dim 8, cell 5; the T-2112 suite's own
// dim8_composition_red.cpp TestDim8_5_HandleChurnConcurrentWithLiveDecode, "the precise D-SLM3311
// shape"): thread A decodes one long-lived sequence continuously for >=64 steps (mutex-protected,
// per-step CPU-oracle-compared); thread B concurrently churns short-lived sequence handles
// against the SAME model handle in a tight loop, forcing address reuse. Thread B's own calls
// (sslm_gpu_seq_create/sslm_gpu_seq_release) upload through THIS CONTEXT's own `ctx->device` --
// a device object distinct from the process-wide singleton thread A's decode calls use (grepped,
// src/gpu/gpu_1p0.cpp: every create/release/restore call site in that file uses `ctx->device`
// directly, never `harness::GetDevice()`) -- so thread B's own churn needs no mutex against
// thread A's decode loop; the two threads touch disjoint device objects. Thread B is also the
// SOLE mutator of ctx->live_handles/model->live_sequences during the concurrent window (thread
// A's own long-lived sequence is created before, and released after, the two threads run), so
// those plain int64_t counters see no concurrent writers either.
//
// Usage: t2113_b8_thread_smoke <model1p5b.sslm>  (exits 0 on pass, 1 on fail)
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "superslm/forward_sites.h"
#include "superslm/gpu_1p0.h"
#include "superslm/gpu_port.h"
#include "superslm/layer_marshal.h"
#include "superslm/model.h"

using superslm::CarriedScale;
using superslm::EmbedEntry;
using superslm::LayerWeights;
using superslm::RunLayerLoop;
using superslm::SequenceLayerState;
using superslm::SslmForwardStatus;
using superslm::SslmModelStatus;
using superslm::SslmModelView;
using superslm_marshal::LayerBacking;
using superslm_marshal::MarshalLayer;
using superslm_marshal::ReadCarriedScale;

static int g_checks = 0;
static int g_failures = 0;

#define CHECK(cond, msg)                                                    \
	do {                                                                       \
		++g_checks;                                                             \
		if (!(cond)) {                                                          \
			++g_failures;                                                         \
			std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, msg);      \
		}                                                                        \
	} while (0)

extern int8_t* SslmGpuSeqHandleHiddenCodesForBench(SslmGpuSequenceHandle*);
extern superslm::CarriedScale* SslmGpuSeqHandleHiddenScaleForBench(SslmGpuSequenceHandle*);
extern uint32_t* SslmGpuSeqHandleLayerIndexForBench(SslmGpuSequenceHandle*);
extern uint64_t* SslmGpuSeqHandleKvSaturationForBench(SslmGpuSequenceHandle*);
extern int64_t* SslmGpuSeqHandleContextLengthForBench(SslmGpuSequenceHandle*);
extern size_t SslmGpuSeqHandleHiddenSizeForBench(SslmGpuSequenceHandle*);

static bool LoadModel(const std::string& path, SslmModelView* out_view, std::vector<uint8_t>* out_bytes) {
	std::FILE* f = std::fopen(path.c_str(), "rb");
	if (!f) {
		std::fprintf(stderr, "could not open %s\n", path.c_str());
		return false;
	}
	std::fseek(f, 0, SEEK_END);
	const long sz = std::ftell(f);
	std::fseek(f, 0, SEEK_SET);
	out_bytes->resize(sz > 0 ? static_cast<size_t>(sz) : 0);
	if (sz > 0) {
		const size_t n = std::fread(out_bytes->data(), 1, static_cast<size_t>(sz), f);
		std::fclose(f);
		if (n != static_cast<size_t>(sz)) {
			std::fprintf(stderr, "short read on %s\n", path.c_str());
			return false;
		}
	} else {
		std::fclose(f);
	}
	std::string err;
	const SslmModelStatus st = superslm::SslmModel::Load(out_bytes->data(), out_bytes->size(), *out_view, &err);
	if (st != SslmModelStatus::Ok) {
		std::fprintf(stderr, "model load failed for %s: %s\n", path.c_str(), err.c_str());
		return false;
	}
	return true;
}

struct SeqSnapshot {
	std::vector<int8_t> hidden_codes;
	CarriedScale hidden_scale{};
	uint32_t layer_index = 0;
	uint64_t kv_saturation_count = 0;
	int64_t context_length = 0;
};
static SeqSnapshot Snapshot(SslmGpuSequenceHandle* seq) {
	SeqSnapshot s;
	const size_t hs = SslmGpuSeqHandleHiddenSizeForBench(seq);
	s.hidden_codes.assign(SslmGpuSeqHandleHiddenCodesForBench(seq),
	                       SslmGpuSeqHandleHiddenCodesForBench(seq) + hs);
	s.hidden_scale = *SslmGpuSeqHandleHiddenScaleForBench(seq);
	s.layer_index = *SslmGpuSeqHandleLayerIndexForBench(seq);
	s.kv_saturation_count = *SslmGpuSeqHandleKvSaturationForBench(seq);
	s.context_length = *SslmGpuSeqHandleContextLengthForBench(seq);
	return s;
}

// One CPU decode step (the oracle): EmbedEntry(token) then RunLayerLoop over every layer.
static SslmForwardStatus StepCpu(SequenceLayerState& seq, int32_t token, int32_t vocab_size,
                                  const int8_t* embed_weights, size_t hidden_size,
                                  const CarriedScale& embed_site_constant, const LayerWeights* layers,
                                  uint32_t num_hidden_layers, size_t head_dim, size_t num_kv_heads,
                                  size_t intermediate_size, int64_t context_cap,
                                  const superslm::SslmTensorManifest& rope_tables, uint8_t* ws,
                                  size_t ws_size) {
	std::vector<int8_t> embed_codes(hidden_size);
	CarriedScale embed_scale{};
	const SslmForwardStatus est = EmbedEntry(token, vocab_size, embed_weights, hidden_size,
	                                          embed_site_constant, embed_codes.data(), &embed_scale);
	if (est != SslmForwardStatus::Ok) return est;
	std::memcpy(seq.hidden_codes, embed_codes.data(), hidden_size);
	seq.hidden_scale = embed_scale;
	seq.layer_index = 0;
	return RunLayerLoop(seq, layers, num_hidden_layers, num_hidden_layers, hidden_size, head_dim,
	                     num_kv_heads, intermediate_size, context_cap, rope_tables, ws, ws_size);
}

// The mutex external to the 1.0 API this whole tool's own thesis rests on -- see the file
// header comment. Scoped to EXACTLY the two calls that touch the shared device singleton:
// sslm_decode_step_gpu (the submit) and sslm_gpu_ready(block=1) (the drain).
static std::mutex g_submit_mutex;

static SslmGpuStatus SubmitAndDrainSerialized(SslmGpuContext* ctx, SslmGpuSequenceHandle* seq,
                                               const SslmGpuAdapterHandle* adapter_or_null,
                                               uint32_t dispatch_budget) {
	std::lock_guard<std::mutex> lock(g_submit_mutex);
	const SslmGpuStatus submit_status = sslm_decode_step_gpu(ctx, seq, adapter_or_null, dispatch_budget);
	if (submit_status != SSLM_OK) return submit_status;
	int32_t ready = 0;
	SslmGpuStatus out_status = SSLM_OK;
	SslmGpuStatus st = SSLM_OK;
	while (!ready) st = sslm_gpu_ready(ctx, seq, /*block=*/1, &ready, &out_status);
	return st != SSLM_OK ? st : out_status;
}

int main(int argc, char** argv) {
	if (argc < 2) {
		std::fprintf(stderr, "usage: %s <model1p5b.sslm>\n", argv[0]);
		return 2;
	}

	std::vector<uint8_t> model_bytes;
	SslmModelView view;
	if (!LoadModel(argv[1], &view, &model_bytes)) return 1;

	const uint32_t num_heads = view.config.num_attention_heads;
	const uint32_t num_kv_heads = view.config.num_key_value_heads;
	const uint32_t num_hidden_layers = view.config.num_hidden_layers;
	const size_t hidden_size = view.config.hidden_size;
	const size_t head_dim = view.config.head_dim;
	const size_t intermediate_size = view.config.intermediate_size;
	const int64_t context_cap = static_cast<int64_t>(view.config.context_cap);
	const int32_t vocab_size = view.config.vocab_size;
	constexpr uint32_t kDispatchesPerLayer = 24;
	const uint32_t kFullToken = kDispatchesPerLayer * num_hidden_layers;

	std::vector<LayerBacking> backings(num_hidden_layers);
	std::vector<LayerWeights> layers(num_hidden_layers);
	for (uint32_t l = 0; l < num_hidden_layers; ++l) {
		std::string err;
		if (!MarshalLayer(view, l, num_heads, num_kv_heads, backings[l], layers[l], &err)) {
			std::fprintf(stderr, "FAILED at stage=layer_weights_marshal: layer=%u \"%s\"\n", l, err.c_str());
			return 1;
		}
	}
	const superslm::SslmTensorView* embed_w = view.weights.Tensor("embed");
	CHECK(embed_w != nullptr, "artifact has no embed tensor");
	if (!embed_w) return 1;
	bool ok = true;
	const CarriedScale embed_site_constant = ReadCarriedScale(view.composition_constants, "embed", &ok);
	CHECK(ok, "artifact has no embed site constant");
	const int8_t* embed_weights = reinterpret_cast<const int8_t*>(embed_w->data);
	const size_t kv_bytes = static_cast<size_t>(num_hidden_layers) * static_cast<size_t>(context_cap) *
	                        num_kv_heads * head_dim * 2;

	GpuContextConfig cfg{};
	GpuResidencyConfig rcfg{};
	SslmGpuContext* ctx = nullptr;
	CHECK(sslm_gpu_context_create(cfg, &ctx) == SSLM_OK && ctx != nullptr, "sslm_gpu_context_create failed");
	if (!ctx) return g_failures ? 1 : 0;

	SslmGpuModelHandle* model = nullptr;
	CHECK(sslm_gpu_model_map(ctx, &view, rcfg, &model) == SSLM_OK && model != nullptr,
	      "sslm_gpu_model_map(1.5B) failed");
	if (!model) {
		sslm_gpu_context_destroy(ctx);
		return g_failures ? 1 : 0;
	}

	// ============================================================================
	// Gate 1 -- the design's own B8 gate, realized as the stronger claim: two threads, two
	// disjoint sequences, one context, concurrent sslm_decode_step_gpu calls (externally
	// serialized only at the submit boundary), per-step bit-equality against the CPU oracle,
	// for 64 steps each (the design Sec11 dim3 product cell's own "real 1.5B artifact... 64
	// steps each" shape).
	// ============================================================================
	{
		constexpr int kSteps = 64;
		const int32_t tokens[] = {5, 128, 1000, 42};  // cycled -- real, non-degenerate content

		SslmGpuSequenceHandle* seq_a = nullptr;
		SslmGpuSequenceHandle* seq_b = nullptr;
		CHECK(sslm_gpu_seq_create(ctx, model, context_cap, &seq_a) == SSLM_OK && seq_a != nullptr,
		      "Gate1 seq_a create failed");
		CHECK(sslm_gpu_seq_create(ctx, model, context_cap, &seq_b) == SSLM_OK && seq_b != nullptr,
		      "Gate1 seq_b create failed");

		bool thread_a_ok = true, thread_b_ok = true;
		std::vector<SeqSnapshot> gpu_a(kSteps), gpu_b(kSteps);
		std::thread thread_a([&] {
			for (int i = 0; i < kSteps; ++i) {
				if (sslm_gpu_seq_embed_token(ctx, seq_a, tokens[i % 4]) != SSLM_OK) thread_a_ok = false;
				if (SubmitAndDrainSerialized(ctx, seq_a, nullptr, kFullToken) != SSLM_OK) thread_a_ok = false;
				gpu_a[i] = Snapshot(seq_a);
			}
		});
		std::thread thread_b([&] {
			for (int i = 0; i < kSteps; ++i) {
				if (sslm_gpu_seq_embed_token(ctx, seq_b, tokens[i % 4]) != SSLM_OK) thread_b_ok = false;
				if (SubmitAndDrainSerialized(ctx, seq_b, nullptr, kFullToken) != SSLM_OK) thread_b_ok = false;
				gpu_b[i] = Snapshot(seq_b);
			}
		});
		thread_a.join();
		thread_b.join();
		CHECK(thread_a_ok, "Gate1 thread A observed a non-Ok status during concurrent decode");
		CHECK(thread_b_ok, "Gate1 thread B observed a non-Ok status during concurrent decode");
		sslm_gpu_seq_release(ctx, seq_a);
		sslm_gpu_seq_release(ctx, seq_b);

		// CPU oracle, run AFTER the concurrent GPU run completes (order does not matter -- the
		// CPU oracle is independent of the GPU run entirely), one instance per sequence, over the
		// identical token sequence each thread fed its own GPU sequence.
		bool all_match = thread_a_ok && thread_b_ok;
		for (int who = 0; who < 2 && all_match; ++who) {
			const std::vector<SeqSnapshot>& gpu_snaps = who == 0 ? gpu_a : gpu_b;
			SequenceLayerState cpu_seq{};
			std::vector<int8_t> cpu_codes(hidden_size, 0);
			cpu_seq.hidden_codes = cpu_codes.data();
			std::vector<uint8_t> cpu_ws(kv_bytes, 0);
			for (int i = 0; i < kSteps && all_match; ++i) {
				const SslmForwardStatus cpu_st =
				    StepCpu(cpu_seq, tokens[i % 4], vocab_size, embed_weights, hidden_size,
				            embed_site_constant, layers.data(), num_hidden_layers, head_dim, num_kv_heads,
				            intermediate_size, context_cap, view.rope_tables, cpu_ws.data(), cpu_ws.size());
				bool step_ok = cpu_st == SslmForwardStatus::Ok &&
				               cpu_seq.kv_saturation_count == gpu_snaps[i].kv_saturation_count &&
				               cpu_seq.context_length == gpu_snaps[i].context_length;
				for (size_t j = 0; step_ok && j < hidden_size; ++j) {
					if (cpu_codes[j] != gpu_snaps[i].hidden_codes[j]) step_ok = false;
				}
				if (!step_ok) {
					std::fprintf(stderr, "  DIVERGENCE: who=%d step=%d GPU-concurrent != CPU-oracle\n", who, i);
					all_match = false;
				}
			}
		}
		CHECK(all_match, "Gate 1: concurrent two-thread decode (disjoint sequences) not bit-identical to CPU oracle");
		std::fprintf(stderr, "  Gate 1 (2 threads, disjoint sequences, %d steps each, vs CPU oracle): %s\n",
		             kSteps, all_match ? "OK, bit-identical every step" : "FAIL");
	}

	// ============================================================================
	// Gate 2 -- design Sec11 dim8 cell 5 / dim1 x dim3 crossed cell (the T-2112 suite's own
	// TestDim8_5_HandleChurnConcurrentWithLiveDecode, "the precise D-SLM3311 shape"): thread A
	// decodes one long-lived sequence continuously for 64 steps (mutex-protected submit,
	// per-step CPU-oracle-compared); thread B concurrently churns short-lived sequence handles
	// against the SAME model handle, forcing address reuse, for the same duration.
	// ============================================================================
	{
		constexpr int kSteps = 64;
		constexpr int kChurnIterations = 200;
		const int32_t tokens[] = {5, 128, 1000, 42};

		SslmGpuSequenceHandle* long_lived = nullptr;
		CHECK(sslm_gpu_seq_create(ctx, model, context_cap, &long_lived) == SSLM_OK && long_lived != nullptr,
		      "Gate2 long_lived seq create failed");

		bool thread_a_ok = true;
		std::vector<SeqSnapshot> gpu_snaps(kSteps);
		std::thread thread_a([&] {
			for (int i = 0; i < kSteps; ++i) {
				if (sslm_gpu_seq_embed_token(ctx, long_lived, tokens[i % 4]) != SSLM_OK) thread_a_ok = false;
				if (SubmitAndDrainSerialized(ctx, long_lived, nullptr, kFullToken) != SSLM_OK) thread_a_ok = false;
				gpu_snaps[i] = Snapshot(long_lived);
			}
		});
		int churn_creates_ok = 0, churn_releases_ok = 0;
		std::thread thread_b([&] {
			for (int i = 0; i < kChurnIterations; ++i) {
				SslmGpuSequenceHandle* churn = nullptr;
				if (sslm_gpu_seq_create(ctx, model, context_cap, &churn) == SSLM_OK && churn != nullptr) {
					++churn_creates_ok;
					if (sslm_gpu_seq_release(ctx, churn) == SSLM_OK) ++churn_releases_ok;
				}
			}
		});
		thread_a.join();
		thread_b.join();
		CHECK(thread_a_ok, "Gate2 thread A observed a non-Ok status while thread B churned handles concurrently");
		CHECK(churn_creates_ok == kChurnIterations, "Gate2 thread B: not every churn create succeeded");
		CHECK(churn_releases_ok == kChurnIterations, "Gate2 thread B: not every churn release succeeded");

		bool all_match = thread_a_ok;
		SequenceLayerState cpu_seq{};
		std::vector<int8_t> cpu_codes(hidden_size, 0);
		cpu_seq.hidden_codes = cpu_codes.data();
		std::vector<uint8_t> cpu_ws(kv_bytes, 0);
		for (int i = 0; i < kSteps && all_match; ++i) {
			const SslmForwardStatus cpu_st =
			    StepCpu(cpu_seq, tokens[i % 4], vocab_size, embed_weights, hidden_size, embed_site_constant,
			            layers.data(), num_hidden_layers, head_dim, num_kv_heads, intermediate_size,
			            context_cap, view.rope_tables, cpu_ws.data(), cpu_ws.size());
			bool step_ok = cpu_st == SslmForwardStatus::Ok &&
			               cpu_seq.kv_saturation_count == gpu_snaps[i].kv_saturation_count &&
			               cpu_seq.context_length == gpu_snaps[i].context_length;
			for (size_t j = 0; step_ok && j < hidden_size; ++j) {
				if (cpu_codes[j] != gpu_snaps[i].hidden_codes[j]) step_ok = false;
			}
			if (!step_ok) {
				std::fprintf(stderr, "  DIVERGENCE: step=%d thread-A-under-churn != CPU-oracle\n", i);
				all_match = false;
			}
		}
		CHECK(all_match, "Gate 2: thread A's long-lived sequence diverged from the CPU oracle "
		                  "while thread B churned handles concurrently -- residue crossed, the "
		                  "precise D-SLM3311 shape this cell exists to catch");
		std::fprintf(stderr,
		             "  Gate 2 (dim1xdim3 crossed cell -- long-lived decode under concurrent handle "
		             "churn, %d steps + %d churn iterations): %s\n",
		             kSteps, kChurnIterations, all_match ? "OK, bit-identical every step, zero residue" : "FAIL");
		sslm_gpu_seq_release(ctx, long_lived);
	}

	// ============================================================================
	// Gate 3 -- the single-writer residual (design Sec5.4/Sec11 dim3's own second bullet): NOT
	// re-run here. Already proven by real execution, this arc, against this same production
	// entry point (sslm_decode_step_gpu, no drain between two concurrent calls on the SAME
	// sequence handle) -- the T-2112 suite's own dim3 M2 cell reproduced a real D3D12 device
	// fault (HR FAIL 0x80004005) doing exactly this, Claude/Brunel/t2113-1p0-core-build-
	// 2026-08-15.md Sec11.4. Re-running it here would only re-trigger the same real device fault
	// for no new evidence -- cited, not duplicated.
	// ============================================================================

	sslm_gpu_model_unmap(ctx, model);
	sslm_gpu_context_destroy(ctx);

	std::fprintf(stderr, "T-2113 B8 thread smoke: checks=%d failures=%d\n", g_checks, g_failures);
	return g_failures ? 1 : 0;
}
