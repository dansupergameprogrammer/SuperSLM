// T-2112 (Curie) -- shared CHECK harness and real-artifact loading helper for the 1.0 GPU core
// red suite. Reuses this repo's own established conventions rather than inventing new ones:
// the CHECK/CHECK_MSG counter pair is tests/t2018-slora-serial/t2018_offline_red.cpp's own
// convention (itself tests/test_main.cpp's own convention, cited there); the real-artifact
// load path (SslmModel::Load over a file read into memory) is tools/t2100_gpu_throughput.cpp's
// own convention (T-2100, the substrate's own throughput harness), reused verbatim because
// every "real 1.5B artifact" product cell in this suite needs the identical load path that
// harness already proved works.
//
// Model/adapter paths are NEVER hardcoded (StandardsDocument.md Sec5.2, cold-read self-contained
// / no project-specific-only assumption an artifact lives at one fixed path): every product cell
// that needs a real artifact takes it from g_model_path/g_adapter_path, populated from argv by
// each suite binary's own main(), matching tools/t2100_gpu_throughput.cpp's own
// `<model.sslm> [steps] [token_id]` argv convention. A cell whose product half needs an artifact
// argv did not supply reports SKIP (counted separately from PASS/FAIL/RED so a suite run without
// artifacts on the invoking machine is still legible), never a silent pass.
#ifndef SSLM_T2112_FIXTURE_COMMON_H
#define SSLM_T2112_FIXTURE_COMMON_H

#include <cstdio>
#include <cstring>
#include <mutex>
#include <string>
#include <vector>

#include "superslm/artifact.h"
#include "superslm/forward_sites.h"   // superslm::EmbedEntry/RunLayerLoop/SequenceLayerState -- the CPU oracle
#include "superslm/layer_marshal.h"   // superslm_marshal::LayerBacking/MarshalLayer/ReadCarriedScale
#include "superslm/model.h"

// The declared 1.0 API surface this whole suite is written against (Claude/Vitruvius/
// t2107-gpu-core-1p0-design-2026-08-14.md Sec4/Sec5, post T-2111-rung-5 fold). Promoted copy,
// this directory -- see sslm_gpu_1p0.h's own header comment.
#include "sslm_gpu_1p0.h"

static int GChecks = 0;
static int GFailures = 0;
static int GSkips = 0;

#define CHECK(cond) \
	do { \
		++GChecks; \
		if (!(cond)) { \
			++GFailures; \
			std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
		} \
	} while (0)

#define CHECK_MSG(cond, ...) \
	do { \
		++GChecks; \
		if (!(cond)) { \
			++GFailures; \
			std::printf("FAIL %s:%d: %s -- ", __FILE__, __LINE__, #cond); \
			std::printf(__VA_ARGS__); \
			std::printf("\n"); \
		} \
	} while (0)

#define SKIP_MSG(...) \
	do { \
		++GSkips; \
		std::printf("SKIP %s:%d -- ", __FILE__, __LINE__); \
		std::printf(__VA_ARGS__); \
		std::printf("\n"); \
	} while (0)

// argv-populated real-artifact paths (StandardsDocument.md Sec5.4 real-workload rule: every
// applicable dimension's product cell runs the real 1.5B artifact where the design's own Sec11
// says so, never a hidden_size=2 stand-in). Empty means "not supplied on this invocation" --
// the cell SKIPs its product half rather than silently passing or fabricating a result.
static std::string g_model_1p5b_path;
static std::string g_model_0p5b_path;
static std::string g_adapter_path;   // real adapter artifact (e.g. shopkeeper-v2), for dim8/dim10
// T-2113 (P2, design Sec4.2/Sec22, D-SLM3415): a SECOND real artifact sharing the 1.5B tier's own
// declared hidden_size/context_cap but a genuinely different content hash (e.g. the base,
// qwen2.5-1.5b-shopkeeper-lora-v2-merged-t2102.sslm, a same-tier adapter-merged derivative,
// alongside qwen2.5-1.5b-instruct.sslm) -- design Sec22's own dim-2 "same-shape, different-content" product cell
// needs a pair that forces the model_content_hash check to do work the size/hidden_size checks do
// not (a shape mismatch alone, the g_model_0p5b_path pairing already covers as the foreign-blob
// cell). Optional like every other real-artifact flag: absent means that one product cell SKIPs.
static std::string g_model_1p5b_variant_path;

// Reuses tools/t2100_gpu_throughput.cpp's own load path verbatim (T-2100, this repo).
inline bool LoadRealModel(const std::string& path, superslm::SslmModelView* out_view,
                           std::vector<uint8_t>* out_bytes, std::string* out_err) {
	if (path.empty()) return false;
	std::FILE* f = std::fopen(path.c_str(), "rb");
	if (!f) {
		if (out_err) *out_err = "could not open " + path;
		return false;
	}
	std::fseek(f, 0, SEEK_END);
	const long sz = std::ftell(f);
	std::fseek(f, 0, SEEK_SET);
	out_bytes->resize(sz > 0 ? (size_t)sz : 0);
	if (sz > 0) {
		const size_t n = std::fread(out_bytes->data(), 1, (size_t)sz, f);
		std::fclose(f);
		if (n != (size_t)sz) {
			if (out_err) *out_err = "short read on " + path;
			return false;
		}
	} else {
		std::fclose(f);
	}
	const superslm::SslmModelStatus st =
	    superslm::SslmModel::Load(out_bytes->data(), out_bytes->size(), *out_view, out_err);
	return st == superslm::SslmModelStatus::Ok;
}

// Parses this suite's own argv convention: <binary> [--model1p5b=PATH] [--model0p5b=PATH]
// [--adapter=PATH] [--model1p5bvariant=PATH]. Every flag optional; a cell whose fixture is
// missing SKIPs rather than asserting against nothing.
inline void ParseFixtureArgs(int argc, char** argv) {
	for (int i = 1; i < argc; ++i) {
		const std::string a = argv[i];
		auto take = [&](const char* flag) -> const char* {
			const size_t n = std::strlen(flag);
			return a.compare(0, n, flag) == 0 ? a.c_str() + n : nullptr;
		};
		if (const char* v = take("--model1p5b=")) g_model_1p5b_path = v;
		else if (const char* v = take("--model0p5b=")) g_model_0p5b_path = v;
		else if (const char* v = take("--adapter=")) g_adapter_path = v;
		else if (const char* v = take("--model1p5bvariant=")) g_model_1p5b_variant_path = v;
	}
}

// =============================================================================================
// D-SLM3380/D-SLM3412 REPAIR (Curie, 2026-08-15): shared drain, snapshot, and CPU-oracle
// machinery, reused across dim1/dim3/dim4/dim5/dim6/dim8/dim9/dim11 so the fix for "a decode call
// against a still-Submitted sequence returns SSLM_BUSY" (the async lifecycle, design Sec4.3/Sec9)
// lives in ONE place rather than being re-derived per file, and so the five files the T-2114 fix
// round named as carrying comment-only "FEATURE ORACLE" stubs (Sec22.10 of the build log) gain
// the SAME real, executed, per-step CPU-oracle comparison dim9's own C1 rewrite already proved
// out (Claude/Poirot/50f3d5d-t2113-1p0-gpu-core-build-review.md C1/O2) and tools/t2113_b7_batch_
// smoke.cpp / tools/t2113_b8_thread_smoke.cpp already established as this build's own precedent
// (StepCpu/Snapshot/Drain, reused here nearly verbatim, StandardsDocument.md Sec6.6 -- one real
// implementation, not a second drifting copy per file).
// =============================================================================================

// Bench-bridge read accessors (gpu_1p0.cpp, global scope -- see dim9_persistence_red.cpp's own
// prior header comment on why these are declared OUTSIDE any anonymous namespace: a textually
// nested `extern` inside an anonymous namespace binds to that TU-local namespace, not the real
// global symbol, and fails to link).
extern int8_t* SslmGpuSeqHandleHiddenCodesForBench(SslmGpuSequenceHandle*);
extern superslm::CarriedScale* SslmGpuSeqHandleHiddenScaleForBench(SslmGpuSequenceHandle*);
extern uint32_t* SslmGpuSeqHandleLayerIndexForBench(SslmGpuSequenceHandle*);
extern uint64_t* SslmGpuSeqHandleKvSaturationForBench(SslmGpuSequenceHandle*);
extern int64_t* SslmGpuSeqHandleContextLengthForBench(SslmGpuSequenceHandle*);
extern size_t SslmGpuSeqHandleHiddenSizeForBench(SslmGpuSequenceHandle*);

// The comparable surface a sequence handle exposes -- the SequenceLayerState-complete set C1's
// own defect lived in (a restored non-zero hidden_scale paired with an all-zero hidden_codes).
struct SeqSnapshot {
	std::vector<int8_t> hidden_codes;
	int64_t hidden_scale_m = 0, hidden_scale_e = 0;
	uint32_t layer_index = 0;
	uint64_t kv_saturation_count = 0;
	int64_t context_length = 0;
};

inline bool CaptureSnapshot(SslmGpuSequenceHandle* seq, SeqSnapshot* out) {
	const size_t hidden_size = SslmGpuSeqHandleHiddenSizeForBench(seq);
	const int8_t* codes = SslmGpuSeqHandleHiddenCodesForBench(seq);
	if (!codes) return false;
	out->hidden_codes.assign(codes, codes + hidden_size);
	const superslm::CarriedScale* scale = SslmGpuSeqHandleHiddenScaleForBench(seq);
	out->hidden_scale_m = scale->m;
	out->hidden_scale_e = scale->e;
	out->layer_index = *SslmGpuSeqHandleLayerIndexForBench(seq);
	out->kv_saturation_count = *SslmGpuSeqHandleKvSaturationForBench(seq);
	out->context_length = *SslmGpuSeqHandleContextLengthForBench(seq);
	return true;
}

inline bool SnapshotsBitEqual(const SeqSnapshot& a, const SeqSnapshot& b) {
	return a.hidden_codes == b.hidden_codes && a.hidden_scale_m == b.hidden_scale_m &&
	       a.hidden_scale_e == b.hidden_scale_e && a.layer_index == b.layer_index &&
	       a.kv_saturation_count == b.kv_saturation_count && a.context_length == b.context_length;
}

// D-SLM3387-SHAPE REPAIR (Curie, 2026-08-15): `dispatch_budget` is a DISPATCH count, and
// `kDispatchesPerLayer` (24, design Sec6.1) dispatches complete exactly ONE LAYER, never one
// whole token, on any real artifact this suite loads (24-28 hidden layers). A cell that submits
// budget=24 in a loop and compares the result to `StepCpu` (which decodes one COMPLETE token per
// call, resetting `layer_index` to 0 each time -- the same convention tools/t2113_b7_batch_
// smoke.cpp/tools/t2113_b8_thread_smoke.cpp's own StepCpu already establishes) is comparing two
// different granularities and will diverge at the first token boundary crossed, not because
// either side is wrong. `FullTokenBudget` is the batch-wide/per-call budget that lets ONE decode
// call complete exactly one whole token on a real artifact, matching StepCpu's own granularity --
// use it for every cell whose own claim is "per-STEP" (= per-generated-token) bit-equality; keep
// the bare `kDispatchesPerLayer` literal only for cells whose own claim is specifically about
// layer/dispatch-count granularity (a budget-floor rejection, a fine-grained slice sweep).
constexpr uint32_t kDispatchesPerLayer = 24;
inline uint32_t FullTokenBudget(uint32_t num_hidden_layers) {
	return kDispatchesPerLayer * num_hidden_layers;
}

// Drains one sequence's in-flight decode to completion (block=1) -- design Sec4.3's own async
// contract: sslm_decode_step_gpu submits without fencing and returns immediately; a second call
// (or a release/unmap/destroy) against a still-Submitted sequence returns SSLM_BUSY until this is
// called. Matches tools/t2113_b7_batch_smoke.cpp's/tools/t2113_b8_thread_smoke.cpp's own `Drain`
// helper exactly.
inline SslmGpuStatus Drain(SslmGpuContext* ctx, SslmGpuSequenceHandle* seq) {
	int32_t ready = 0;
	SslmGpuStatus out_status = SSLM_OK;
	SslmGpuStatus st = SSLM_OK;
	while (!ready) st = sslm_gpu_ready(ctx, seq, /*block=*/1, &ready, &out_status);
	return st != SSLM_OK ? st : out_status;
}

// Submits one decode step and drains it to completion before returning -- the idiom every cell
// that issues more than one decode call in a row against the SAME sequence handle now uses
// (D-SLM3380's own named fix shape: "insert sslm_gpu_ready(...) after each sslm_decode_step_gpu
// call whose own next line reuses the same sequence handle").
inline bool RunStepBlocking(SslmGpuContext* ctx, SslmGpuSequenceHandle* seq,
                             const SslmGpuAdapterHandle* adapter, uint32_t budget) {
	if (sslm_decode_step_gpu(ctx, seq, adapter, budget) != SSLM_OK) return false;
	return Drain(ctx, seq) == SSLM_OK;
}

// Embeds `token_id` fresh (resetting layer_index to 0, design Sec5.3a) then submits+drains one
// COMPLETE token's own decode (FullTokenBudget dispatches) -- the idiom every cell that runs a
// MULTI-STEP loop compared against the CPU oracle now uses. Re-embedding before every call is
// required, not optional: unlike the CPU oracle's own StepCpu (which resets layer_index to 0
// itself every call), a GPU sequence handle's layer_index does NOT auto-wrap after completing a
// token -- a second FullTokenBudget call with no intervening re-embed would resume an
// already-`layer_index == num_hidden_layers` sequence, not start a fresh token (design Sec5.3a's
// own "resets layer_index to 0" is the embed call's own side effect, never the decode call's).
inline bool RunFullTokenStep(SslmGpuContext* ctx, SslmGpuSequenceHandle* seq,
                              const SslmGpuAdapterHandle* adapter, uint32_t num_hidden_layers,
                              int32_t token_id) {
	if (sslm_gpu_seq_embed_token(ctx, seq, token_id) != SSLM_OK) return false;
	return RunStepBlocking(ctx, seq, adapter, FullTokenBudget(num_hidden_layers));
}

// The mutex EXTERNAL to the 1.0 API (design Sec5.4/Sec13: "a caller driving two threads' decode
// calls through one context's queue concurrently must serialize the submit call itself" -- the
// decode dispatch path still routes through the pre-1.0 substrate's process-wide device singleton,
// not any one context's own device, so two genuinely concurrent unsynchronized calls race the
// same ID3D12CommandAllocator::Reset()-while-executing hazard D-SLM3384 already proved by
// execution). Reused verbatim from tools/t2113_b8_thread_smoke.cpp's own g_submit_mutex/
// SubmitAndDrainSerialized, which is the established, commissioned pattern for driving concurrent
// decode calls against a shared context SAFELY -- every cell that needs concurrent decode (dim3,
// dim8 cell 5) now uses this instead of an unguarded std::thread + sslm_decode_step_gpu loop.
inline std::mutex& SubmitMutex() {
	static std::mutex m;
	return m;
}

inline SslmGpuStatus SubmitAndDrainSerialized(SslmGpuContext* ctx, SslmGpuSequenceHandle* seq,
                                               const SslmGpuAdapterHandle* adapter_or_null,
                                               uint32_t dispatch_budget) {
	std::lock_guard<std::mutex> lock(SubmitMutex());
	const SslmGpuStatus submit_status = sslm_decode_step_gpu(ctx, seq, adapter_or_null, dispatch_budget);
	if (submit_status != SSLM_OK) return submit_status;
	return Drain(ctx, seq);
}

// The CPU oracle a real artifact's own GPU decode is checked against -- the SAME construction
// tools/t2113_b7_batch_smoke.cpp's/tools/t2113_b8_thread_smoke.cpp's own StepCpu already
// established (EmbedEntry(token) then RunLayerLoop over every layer), bundled here as one loaded
// model so every dim file's own product cell can compare a real GPU run to it without
// re-marshaling the artifact's layer weights per file.
struct CpuOracleModel {
	std::vector<superslm_marshal::LayerBacking> backings;
	std::vector<superslm::LayerWeights> layers;
	const int8_t* embed_weights = nullptr;
	superslm::CarriedScale embed_site_constant{};
	uint32_t num_hidden_layers = 0;
	size_t hidden_size = 0, head_dim = 0, num_kv_heads = 0, intermediate_size = 0;
	int64_t context_cap = 0;
	int32_t vocab_size = 0;
	const superslm::SslmTensorManifest* rope_tables = nullptr;
};

inline bool LoadCpuOracleModel(const superslm::SslmModelView& view, CpuOracleModel* out,
                                std::string* err) {
	out->num_hidden_layers = view.config.num_hidden_layers;
	out->hidden_size = view.config.hidden_size;
	out->head_dim = view.config.head_dim;
	out->num_kv_heads = view.config.num_key_value_heads;
	out->intermediate_size = view.config.intermediate_size;
	out->context_cap = static_cast<int64_t>(view.config.context_cap);
	out->vocab_size = view.config.vocab_size;
	out->rope_tables = &view.rope_tables;
	out->backings.resize(out->num_hidden_layers);
	out->layers.resize(out->num_hidden_layers);
	for (uint32_t l = 0; l < out->num_hidden_layers; ++l) {
		if (!superslm_marshal::MarshalLayer(view, l, view.config.num_attention_heads,
		                                     view.config.num_key_value_heads, out->backings[l],
		                                     out->layers[l], err)) {
			return false;
		}
	}
	const superslm::SslmTensorView* embed_w = view.weights.Tensor("embed");
	if (!embed_w) {
		if (err) *err = "artifact has no embed tensor";
		return false;
	}
	out->embed_weights = reinterpret_cast<const int8_t*>(embed_w->data);
	bool ok = true;
	out->embed_site_constant =
	    superslm_marshal::ReadCarriedScale(view.composition_constants, "embed", &ok);
	if (!ok) {
		if (err) *err = "artifact has no embed site constant";
		return false;
	}
	return true;
}

// One CPU decode step: EmbedEntry(token) then RunLayerLoop over every layer -- byte-for-byte the
// same construction tools/t2113_b7_batch_smoke.cpp/tools/t2113_b8_thread_smoke.cpp already run as
// their own oracle.
inline superslm::SslmForwardStatus StepCpu(superslm::SequenceLayerState& seq, int32_t token,
                                            const CpuOracleModel& m, uint8_t* ws, size_t ws_size) {
	std::vector<int8_t> embed_codes(m.hidden_size);
	superslm::CarriedScale embed_scale{};
	const superslm::SslmForwardStatus est =
	    superslm::EmbedEntry(token, m.vocab_size, m.embed_weights, m.hidden_size,
	                          m.embed_site_constant, embed_codes.data(), &embed_scale);
	if (est != superslm::SslmForwardStatus::Ok) return est;
	std::memcpy(seq.hidden_codes, embed_codes.data(), m.hidden_size);
	seq.hidden_scale = embed_scale;
	seq.layer_index = 0;
	return superslm::RunLayerLoop(seq, m.layers.data(), m.num_hidden_layers, m.num_hidden_layers,
	                               m.hidden_size, m.head_dim, m.num_kv_heads, m.intermediate_size,
	                               m.context_cap, *m.rope_tables, ws, ws_size);
}

// A CPU-oracle-tracking scratch pad -- the CPU-side SequenceLayerState plus its owned buffers,
// bundled so a cell can run N steps of the CPU oracle and compare each against the GPU snapshot
// for the same step without re-deriving the buffer sizing per cell.
struct CpuOracleRunner {
	superslm::SequenceLayerState seq{};
	std::vector<int8_t> codes;
	std::vector<uint8_t> ws;

	void Init(const CpuOracleModel& m) {
		codes.assign(m.hidden_size, 0);
		seq = superslm::SequenceLayerState{};
		seq.hidden_codes = codes.data();
		const size_t kv_bytes = static_cast<size_t>(m.num_hidden_layers) *
		                         static_cast<size_t>(m.context_cap) * m.num_kv_heads * m.head_dim * 2;
		ws.assign(kv_bytes, 0);
	}

	// Runs one CPU step and reports whether it is bit-equal to `gpu` on the comparable surface
	// (hidden_codes, kv_saturation_count, context_length -- the same fields SnapshotsBitEqual
	// checks, hidden_scale is re-derived from hidden_codes by RunLayerLoop and not carried
	// separately here since CpuOracleModel's own seq owns hidden_scale directly).
	bool StepMatchesGpu(int32_t token, const CpuOracleModel& m, const SeqSnapshot& gpu) {
		const superslm::SslmForwardStatus st = StepCpu(seq, token, m, ws.data(), ws.size());
		if (st != superslm::SslmForwardStatus::Ok) return false;
		if (seq.kv_saturation_count != gpu.kv_saturation_count) return false;
		if (seq.context_length != gpu.context_length) return false;
		for (size_t j = 0; j < m.hidden_size; ++j) {
			if (codes[j] != gpu.hidden_codes[j]) return false;
		}
		return true;
	}
};

#endif  // SSLM_T2112_FIXTURE_COMMON_H
