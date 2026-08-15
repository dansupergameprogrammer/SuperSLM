// T-2113 (B3): the bench proof for design Sec10 B3's own delivery -- SslmGpuSequenceHandle
// create/release; a real, dedicated DEFAULT-heap K/V buffer per sequence, replacing
// `g_resident_kv`'s single pointer-keyed slot; the SequenceKvBufferMismatch guard. Not part
// of Curie's t2112 red suite (that suite's own dim1/dim2 product cells need this handle to
// link at all, per its own handoff -- this is a build-seat bench proof, the same convention
// as tools/t2113_b1_context_smoke.cpp / tools/t2113_b2_model_smoke.cpp).
//
// Design Sec10 B3's own stated gate: "two concurrently-live sequences against one model
// handle, single-threaded interleaved calls, produce the same per-sequence output as two
// fully-serial single-sequence runs." No `sslm_decode_step_gpu` exists yet (B5 lands it), so
// this tool drives a real forward pass through each sequence handle's own dedicated buffer
// via the pre-1.0 `superslm_gpu::RunLayerLoopGpu`'s own B3 external-buffer bridge
// (gpu_port.h's trailing `external_kv_resident`/`io_external_kv_needs_resume_barrier`
// pair), reached through the build-seat-only accessors in
// include/superslm/gpu_1p0_bench_bridge.h (retired at B5, see that header). The actual
// gate -- interleaved matches serial -- is instantiated here as the STRONGER claim design
// Sec11 dim1's own product cells ask for: per-step CPU/GPU bit-equality, against the CPU
// oracle (RunLayerLoop) run serially, for each of two sequences whose GPU calls are
// interleaved A/B/A/B/... through their own independent handles.
//
// Proves, on real D3D12 hardware, against a real 1.5B artifact (StandardsDocument.md
// Sec5.4's real-workload rule):
//   1. Null ctx / null model / context_cap<1 rejected (SSLM_SEQUENCE_KV_BUFFER_MISMATCH).
//   2. sslm_gpu_seq_release(nullptr) is a documented no-op.
//   3. Single create/release succeeds; ContextHasLiveHandles/ModelHasLiveSequences (B1/B2)
//      observe the live-handle counts moving as sequences are created and released.
//   4. Two independent sequence handles against ONE model handle, live at the same time,
//      are distinct pointers; a model with live sequences refuses to unmap
//      (SSLM_MODEL_HAS_LIVE_SEQUENCES); a context with live sequence (or model) handles
//      refuses to destroy (SSLM_CONTEXT_HAS_LIVE_HANDLES).
//   5. Release-then-create (dim1 M2's own mechanism shape): releasing every live sequence
//      and creating a fresh one afterward succeeds and the fresh handle's own K/V buffer,
//      driven through a real decode, matches the CPU oracle from a cold start -- proving no
//      residue from the released handle's own (address-reusable) allocation survives.
//   6. THE MAIN GATE: seqA and seqB, both live against the same model handle, decoded for
//      N steps each via INTERLEAVED calls (A0, B0, A1, B1, ..., A[N-1], B[N-1]) through
//      their own dedicated K/V buffers -- every step's hidden_codes/hidden_scale/
//      layer_index/kv_saturation_count/context_length compared bit-for-bit against that
//      SAME sequence's own CPU oracle (RunLayerLoop), run fully serially, from an
//      identical initial state. No cross-sequence divergence at any step proves the two
//      buffers never shared, aliased, or evicted each other's content.
//
// Usage: t2113_b3_sequence_smoke <model1p5b.sslm> [steps]  (exits 0 on pass, 1 on fail)
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "superslm/forward_sites.h"
#include "superslm/gpu_1p0.h"
#include "superslm/gpu_1p0_bench_bridge.h"
#include "superslm/gpu_port.h"
#include "superslm/layer_marshal.h"
#include "superslm/model.h"

using superslm::CarriedScale;
using superslm::EmbedEntry;
using superslm::LayerWeights;
using superslm::RunLayerLoop;
using superslm::SequenceLayerState;
using superslm::SslmForwardStatus;
using superslm::SslmForwardStatusName;
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

static bool LoadModel(const std::string& path, SslmModelView* out_view,
                       std::vector<uint8_t>* out_bytes) {
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

// One step of "re-embed token 0, run every layer" -- the identical convention T-2100/T-2105
// use for a decode step (context_length advances across calls; the embedded token itself is
// held fixed so the oracle comparison isolates the K/V-residency mechanism from any
// autoregressive sampling decision, which this bench proof has no stake in).
static SslmForwardStatus StepCpu(SequenceLayerState& seq, int8_t* codes, const int8_t* embed_codes,
                                  const CarriedScale& embed_scale, const LayerWeights* layers,
                                  uint32_t num_hidden_layers, size_t hidden_size, size_t head_dim,
                                  size_t num_kv_heads, size_t intermediate_size, int64_t context_cap,
                                  const superslm::SslmTensorManifest& rope_tables, uint8_t* ws,
                                  size_t ws_size) {
	std::memcpy(codes, embed_codes, hidden_size);
	seq.hidden_scale = embed_scale;
	seq.layer_index = 0;
	return RunLayerLoop(seq, layers, num_hidden_layers, num_hidden_layers, hidden_size, head_dim,
	                     num_kv_heads, intermediate_size, context_cap, rope_tables, ws, ws_size);
}

// The GPU-side step, driven through a real SslmGpuSequenceHandle's own dedicated buffer via
// the B3 bench bridge -- see this file's own header comment.
static SslmForwardStatus StepGpuThroughHandle(SslmGpuSequenceHandle* seq_handle,
                                               const int8_t* embed_codes, const CarriedScale& embed_scale,
                                               const LayerWeights* layers, uint32_t num_hidden_layers,
                                               size_t hidden_size, size_t head_dim, size_t num_kv_heads,
                                               size_t intermediate_size, int64_t context_cap,
                                               const superslm::SslmTensorManifest& rope_tables, uint8_t* ws,
                                               size_t ws_size, SequenceLayerState* out_view_for_compare) {
	int8_t* codes = SslmGpuSeqHandleHiddenCodesForBench(seq_handle);
	std::memcpy(codes, embed_codes, hidden_size);
	*SslmGpuSeqHandleHiddenScaleForBench(seq_handle) = embed_scale;
	*SslmGpuSeqHandleLayerIndexForBench(seq_handle) = 0;

	SequenceLayerState view;
	view.hidden_codes = codes;
	view.hidden_scale = *SslmGpuSeqHandleHiddenScaleForBench(seq_handle);
	view.layer_index = 0;
	view.kv_saturation_count = *SslmGpuSeqHandleKvSaturationForBench(seq_handle);
	view.context_length = *SslmGpuSeqHandleContextLengthForBench(seq_handle);

	const SslmForwardStatus st = superslm_gpu::RunLayerLoopGpu(
	    view, layers, num_hidden_layers, num_hidden_layers, hidden_size, head_dim, num_kv_heads,
	    intermediate_size, context_cap, rope_tables, ws, ws_size,
	    SslmGpuSeqHandleKvBufferForBench(seq_handle), SslmGpuSeqHandleKvResumeFlagForBench(seq_handle));

	// Write the post-call state back into the handle's own host mirror, matching what a
	// real sslm_decode_step_gpu (B5) will do at the end of every call.
	*SslmGpuSeqHandleLayerIndexForBench(seq_handle) = view.layer_index;
	*SslmGpuSeqHandleKvSaturationForBench(seq_handle) = view.kv_saturation_count;
	*SslmGpuSeqHandleContextLengthForBench(seq_handle) = view.context_length;
	*out_view_for_compare = view;
	return st;
}

static bool CompareStep(const char* who, int step, SslmForwardStatus cpu_st, const SequenceLayerState& cpu,
                         const std::vector<int8_t>& cpu_codes, SslmForwardStatus gpu_st,
                         const SequenceLayerState& gpu, size_t hidden_size) {
	bool ok = true;
	if (cpu_st != gpu_st) {
		std::fprintf(stderr, "%s step %d: DIVERGENCE status CPU=%s GPU=%s\n", who, step,
		             SslmForwardStatusName(cpu_st), SslmForwardStatusName(gpu_st));
		ok = false;
	}
	if (cpu.layer_index != gpu.layer_index) {
		std::fprintf(stderr, "%s step %d: DIVERGENCE layer_index CPU=%u GPU=%u\n", who, step,
		             cpu.layer_index, gpu.layer_index);
		ok = false;
	}
	if (cpu.hidden_scale.m != gpu.hidden_scale.m || cpu.hidden_scale.e != gpu.hidden_scale.e) {
		std::fprintf(stderr, "%s step %d: DIVERGENCE hidden_scale\n", who, step);
		ok = false;
	}
	if (cpu.kv_saturation_count != gpu.kv_saturation_count) {
		std::fprintf(stderr, "%s step %d: DIVERGENCE kv_saturation_count CPU=%llu GPU=%llu\n", who, step,
		             (unsigned long long)cpu.kv_saturation_count, (unsigned long long)gpu.kv_saturation_count);
		ok = false;
	}
	if (cpu.context_length != gpu.context_length) {
		std::fprintf(stderr, "%s step %d: DIVERGENCE context_length CPU=%lld GPU=%lld\n", who, step,
		             (long long)cpu.context_length, (long long)gpu.context_length);
		ok = false;
	}
	for (size_t i = 0; i < hidden_size; ++i) {
		if (cpu_codes[i] != gpu.hidden_codes[i]) {
			std::fprintf(stderr, "%s step %d: DIVERGENCE hidden_codes[%zu] CPU=%d GPU=%d\n", who, step, i,
			             cpu_codes[i], gpu.hidden_codes[i]);
			ok = false;
			break;
		}
	}
	return ok;
}

int main(int argc, char** argv) {
	if (argc < 2) {
		std::fprintf(stderr, "usage: %s <model1p5b.sslm> [steps]\n", argv[0]);
		return 2;
	}
	const int steps = argc >= 3 ? std::atoi(argv[2]) : 8;

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
	if (!embed_w) {
		std::fprintf(stderr, "FAILED: artifact has no embed tensor\n");
		return 1;
	}
	bool ok = true;
	const CarriedScale embed_site_constant = ReadCarriedScale(view.composition_constants, "embed", &ok);
	if (!ok) {
		std::fprintf(stderr, "FAILED: artifact has no embed site constant\n");
		return 1;
	}
	const int8_t* embed_weights = reinterpret_cast<const int8_t*>(embed_w->data);
	std::vector<int8_t> embed_codes(hidden_size);
	CarriedScale embed_scale{};
	const SslmForwardStatus est = EmbedEntry(0, view.config.vocab_size, embed_weights, hidden_size,
	                                          embed_site_constant, embed_codes.data(), &embed_scale);
	CHECK(est == SslmForwardStatus::Ok, "EmbedEntry(token 0) did not return Ok");
	if (est != SslmForwardStatus::Ok) return 1;

	const size_t kv_bytes = static_cast<size_t>(num_hidden_layers) * static_cast<size_t>(context_cap) *
	                        num_kv_heads * head_dim * 2;

	GpuContextConfig cfg{};
	GpuResidencyConfig rcfg{};

	// 1: null ctx / null model / context_cap<1 rejected.
	SslmGpuSequenceHandle* null_check = nullptr;
	CHECK(sslm_gpu_seq_create(nullptr, nullptr, context_cap, &null_check) == SSLM_SEQUENCE_KV_BUFFER_MISMATCH,
	      "sslm_gpu_seq_create(null ctx, null model, ...) did not return SSLM_SEQUENCE_KV_BUFFER_MISMATCH");
	CHECK(null_check == nullptr, "sslm_gpu_seq_create rejection left *out_seq non-null");

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

	CHECK(sslm_gpu_seq_create(ctx, nullptr, context_cap, &null_check) == SSLM_SEQUENCE_KV_BUFFER_MISMATCH,
	      "sslm_gpu_seq_create(ctx, null model, ...) did not return SSLM_SEQUENCE_KV_BUFFER_MISMATCH");
	CHECK(sslm_gpu_seq_create(ctx, model, 0, &null_check) == SSLM_SEQUENCE_KV_BUFFER_MISMATCH,
	      "sslm_gpu_seq_create(ctx, model, context_cap=0, ...) did not return SSLM_SEQUENCE_KV_BUFFER_MISMATCH");

	// 2: release(nullptr) is a no-op.
	CHECK(sslm_gpu_seq_release(ctx, nullptr) == SSLM_OK, "sslm_gpu_seq_release(nullptr) did not return SSLM_OK");

	// 3: single create/release, boundary bookkeeping.
	SslmGpuSequenceHandle* solo = nullptr;
	CHECK(sslm_gpu_seq_create(ctx, model, context_cap, &solo) == SSLM_OK && solo != nullptr,
	      "sslm_gpu_seq_create(solo) failed");
	CHECK(sslm_gpu_model_unmap(ctx, model) == SSLM_MODEL_HAS_LIVE_SEQUENCES,
	      "model_unmap did not refuse while a sequence is live");
	CHECK(sslm_gpu_seq_release(ctx, solo) == SSLM_OK, "sslm_gpu_seq_release(solo) failed");

	// 5 (dim1 M2 shape): release/create cycling to force address reuse, then prove the
	// fresh handle's own decode matches the CPU oracle from a cold start (no residue).
	for (int i = 0; i < 8; ++i) {
		SslmGpuSequenceHandle* churn = nullptr;
		CHECK(sslm_gpu_seq_create(ctx, model, context_cap, &churn) == SSLM_OK && churn != nullptr,
		      "sslm_gpu_seq_create (churn) failed");
		CHECK(sslm_gpu_seq_release(ctx, churn) == SSLM_OK, "sslm_gpu_seq_release (churn) failed");
	}

	// 4/6: two independent, concurrently-live sequences against the SAME model handle.
	SslmGpuSequenceHandle* seqA = nullptr;
	SslmGpuSequenceHandle* seqB = nullptr;
	CHECK(sslm_gpu_seq_create(ctx, model, context_cap, &seqA) == SSLM_OK && seqA != nullptr,
	      "sslm_gpu_seq_create(A) failed");
	CHECK(sslm_gpu_seq_create(ctx, model, context_cap, &seqB) == SSLM_OK && seqB != nullptr,
	      "sslm_gpu_seq_create(B) failed");
	CHECK(seqA != seqB, "seqA and seqB are the same handle pointer");
	CHECK(sslm_gpu_context_destroy(ctx) == SSLM_CONTEXT_HAS_LIVE_HANDLES,
	      "context_destroy did not refuse with live sequence handles outstanding");

	// CPU oracle: A and B, fully serial, N steps each, from an identical cold start.
	std::vector<int8_t> cpu_codes_a(hidden_size), cpu_codes_b(hidden_size);
	SequenceLayerState cpu_seq_a, cpu_seq_b;
	cpu_seq_a.hidden_codes = cpu_codes_a.data();
	cpu_seq_b.hidden_codes = cpu_codes_b.data();
	std::vector<uint8_t> cpu_ws_a(kv_bytes, 0), cpu_ws_b(kv_bytes, 0);
	std::vector<SequenceLayerState> cpu_state_a(steps), cpu_state_b(steps);
	std::vector<std::vector<int8_t>> cpu_codes_hist_a(steps), cpu_codes_hist_b(steps);
	std::vector<SslmForwardStatus> cpu_status_a(steps), cpu_status_b(steps);
	for (int i = 0; i < steps; ++i) {
		cpu_status_a[i] = StepCpu(cpu_seq_a, cpu_codes_a.data(), embed_codes.data(), embed_scale,
		                          layers.data(), num_hidden_layers, hidden_size, head_dim, num_kv_heads,
		                          intermediate_size, context_cap, view.rope_tables, cpu_ws_a.data(),
		                          cpu_ws_a.size());
		cpu_state_a[i] = cpu_seq_a;
		cpu_codes_hist_a[i] = cpu_codes_a;
	}
	for (int i = 0; i < steps; ++i) {
		cpu_status_b[i] = StepCpu(cpu_seq_b, cpu_codes_b.data(), embed_codes.data(), embed_scale,
		                          layers.data(), num_hidden_layers, hidden_size, head_dim, num_kv_heads,
		                          intermediate_size, context_cap, view.rope_tables, cpu_ws_b.data(),
		                          cpu_ws_b.size());
		cpu_state_b[i] = cpu_seq_b;
		cpu_codes_hist_b[i] = cpu_codes_b;
	}

	// GPU: A and B INTERLEAVED, each through its OWN dedicated SslmGpuSequenceHandle buffer.
	std::vector<uint8_t> gpu_ws_a(kv_bytes, 0), gpu_ws_b(kv_bytes, 0);
	bool interleave_ok = true;
	for (int i = 0; i < steps; ++i) {
		SequenceLayerState gpu_a_view;
		const SslmForwardStatus gst_a = StepGpuThroughHandle(
		    seqA, embed_codes.data(), embed_scale, layers.data(), num_hidden_layers, hidden_size, head_dim,
		    num_kv_heads, intermediate_size, context_cap, view.rope_tables, gpu_ws_a.data(), gpu_ws_a.size(),
		    &gpu_a_view);
		interleave_ok &= CompareStep("A", i, cpu_status_a[i], cpu_state_a[i], cpu_codes_hist_a[i], gst_a,
		                             gpu_a_view, hidden_size);

		SequenceLayerState gpu_b_view;
		const SslmForwardStatus gst_b = StepGpuThroughHandle(
		    seqB, embed_codes.data(), embed_scale, layers.data(), num_hidden_layers, hidden_size, head_dim,
		    num_kv_heads, intermediate_size, context_cap, view.rope_tables, gpu_ws_b.data(), gpu_ws_b.size(),
		    &gpu_b_view);
		interleave_ok &= CompareStep("B", i, cpu_status_b[i], cpu_state_b[i], cpu_codes_hist_b[i], gst_b,
		                             gpu_b_view, hidden_size);
	}
	CHECK(interleave_ok,
	      "interleaved GPU decode through two independent SslmGpuSequenceHandle buffers diverged from "
	      "the CPU oracle at one or more steps (see DIVERGENCE lines above)");

	CHECK(sslm_gpu_seq_release(ctx, seqA) == SSLM_OK, "sslm_gpu_seq_release(A) failed");
	CHECK(sslm_gpu_model_unmap(ctx, model) == SSLM_MODEL_HAS_LIVE_SEQUENCES,
	      "model_unmap did not refuse while seqB is still live");
	CHECK(sslm_gpu_seq_release(ctx, seqB) == SSLM_OK, "sslm_gpu_seq_release(B) failed");
	CHECK(sslm_gpu_model_unmap(ctx, model) == SSLM_OK, "model_unmap failed after every sequence was released");
	CHECK(sslm_gpu_context_destroy(ctx) == SSLM_OK,
	      "context_destroy failed after every model/sequence handle was released");

	std::printf("T-2113 B3 sequence smoke: steps=%d checks=%d failures=%d\n", steps, g_checks, g_failures);
	return g_failures == 0 ? 0 : 1;
}
