// T-2178 (Curie) -- shared CHECK harness, real-artifact loading, and per-arm comparison
// machinery for the T-2169 GPU-side batched-prefill red suite. Realizes
// Claude/Vitruvius/t2169-gpu-batched-prefill-design-2026-08-18.md Sec9's Coverage Model.
//
// Reuses this project's own established conventions rather than inventing new ones: the
// CHECK/CHECK_MSG/SKIP_MSG trio and the argv-populated real-artifact path globals are
// tests/t2112-gpu-1p0-red-suite/fixture_common.h's own convention (T-2112); the two-arm
// reference-vs-candidate construction (Arm A = shipped path, driven at the granularity under
// test) is Claude/Loki/t2176-probe-cap-straddle-bridge-semantics.cpp's own convention -- adapted
// here into committed, gated cells per this suite's own casebook Sec2, not reinvented.
//
// UNLIKE T-2112's suite, T-2169's own oracle is NOT a CPU reference model -- the design's own
// central claim (Sec6) is that the batched GPU path reproduces the EXISTING unbatched per-token
// GPU path bit-for-bit; CPU/GPU parity (D-SLM3487) is a separate, already-proven property this
// design does not touch (Sec7) and cell 3 below re-confirms only as a byproduct of matching the
// GPU reference. No CpuOracleModel/CpuOracleRunner machinery is needed or included here.
#ifndef SSLM_T2178_FIXTURE_COMMON_H
#define SSLM_T2178_FIXTURE_COMMON_H

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "superslm/checked_chain_funnel.h"   // superslm::CarriedScale
#include "superslm/gpu_1p0.h"
#include "superslm/gpu_1p0_bench_bridge.h"
#include "superslm/model.h"
#include "support/gpu_chunk_dispatch_instrument.h"

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

// argv-populated real-artifact paths (StandardsDocument.md Sec5.4 real-workload rule). Empty
// means "not supplied on this invocation" -- a cell whose product half needs an artifact SKIPs
// rather than fabricating a result. `g_g5_fixture_path` is the schema-bound fixture
// (tools/t2132_g5_gpu_parity_gpu.cpp's own "shopkeeper_intent_extraction"-schema artifact) the
// schema-twin cells (S1-S5, DFA-admission cells) need; `g_model_1p5b_path` is a plain,
// unconstrained artifact for the prompt-twin cells (P1-P4) that need no schema.
static std::string g_model_1p5b_path;
static std::string g_g5_fixture_path;
static std::string g_schema_name = "shopkeeper_intent_extraction";

inline bool LoadRealModel(const std::string& path, superslm::SslmModelView* out_view,
                           std::vector<uint8_t>* out_bytes, std::string* out_err) {
	std::FILE* f = std::fopen(path.c_str(), "rb");
	if (!f) {
		if (out_err) *out_err = "cannot open " + path;
		return false;
	}
	std::fseek(f, 0, SEEK_END);
	const long sz = std::ftell(f);
	std::fseek(f, 0, SEEK_SET);
	out_bytes->resize(static_cast<size_t>(sz));
	if (sz > 0) {
		const size_t n = std::fread(out_bytes->data(), 1, static_cast<size_t>(sz), f);
		std::fclose(f);
		if (n != static_cast<size_t>(sz)) {
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

// Parses this suite's own argv convention: <binary> [--model1p5b=PATH] [--g5fixture=PATH]
// [--schema=NAME]. Every flag optional; a cell whose fixture is missing SKIPs.
inline void ParseFixtureArgs(int argc, char** argv) {
	for (int i = 1; i < argc; ++i) {
		const std::string a = argv[i];
		auto take = [&](const char* flag) -> const char* {
			const size_t n = std::strlen(flag);
			return a.compare(0, n, flag) == 0 ? a.c_str() + n : nullptr;
		};
		if (const char* v = take("--model1p5b=")) g_model_1p5b_path = v;
		else if (const char* v = take("--g5fixture=")) g_g5_fixture_path = v;
		else if (const char* v = take("--schema=")) g_schema_name = v;
	}
}

const char* GpuStatusName(SslmGpuStatus s) {
	switch (s) {
		case SSLM_OK: return "SSLM_OK";
		case SSLM_DISPATCH_BUDGET_TOO_SMALL: return "SSLM_DISPATCH_BUDGET_TOO_SMALL";
		case SSLM_BUSY: return "SSLM_BUSY";
		case SSLM_SEQUENCE_KV_BUFFER_MISMATCH: return "SSLM_SEQUENCE_KV_BUFFER_MISMATCH";
		case SSLM_DEVICE_LOST: return "SSLM_DEVICE_LOST";
		case SSLM_TOKEN_ID_OUT_OF_RANGE: return "SSLM_TOKEN_ID_OUT_OF_RANGE";
		case SSLM_SEQUENCE_REJECTED: return "SSLM_SEQUENCE_REJECTED";
		default: return "OTHER";
	}
}

// The comparable state surface a sequence handle exposes -- the design's own Sec6 bit-identity
// claim ("K/V, dfa_walk_state, forced_token_count/consumed, and post-prefill hidden_codes are
// bit-identical") realized against the bench-bridge accessors this project's own GPU suites
// already use (tests/t2112-gpu-1p0-red-suite/fixture_common.h SeqSnapshot, extended here with
// hidden_size so a differently-shaped handle cannot silently short-read).
struct SeqSnapshot {
	std::vector<int8_t> hidden_codes;
	int64_t hidden_scale_m = 0, hidden_scale_e = 0;
	uint32_t layer_index = 0;
	uint64_t kv_saturation_count = 0;
	int64_t context_length = 0;
	uint32_t dfa_walk_state = 0;
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
	out->dfa_walk_state = SslmGpuSeqWalkStateForG5Bridge(seq);
	return true;
}

inline bool SnapshotsBitEqual(const SeqSnapshot& a, const SeqSnapshot& b) {
	return a.hidden_codes == b.hidden_codes && a.hidden_scale_m == b.hidden_scale_m &&
	       a.hidden_scale_e == b.hidden_scale_e && a.layer_index == b.layer_index &&
	       a.kv_saturation_count == b.kv_saturation_count &&
	       a.context_length == b.context_length && a.dfa_walk_state == b.dfa_walk_state;
}

// Drains one sequence's in-flight decode to completion (block=1) -- design Sec4.3's own async
// contract for the LOW-LEVEL sslm_decode_step_gpu call (submits without fencing, returns
// immediately). Mirrors tests/t2112-gpu-1p0-red-suite/fixture_common.h's own established `Drain`
// helper. The SslmGpuSeqPrefillPromptForG5Bridge/SslmGpuSeqDecodeStepForG5Bridge entry points
// this suite otherwise uses are already blocking (they call this internally per token), so a
// genuinely in-flight state is observable only through the low-level call, on ONE thread --
// gpu_1p0.h's own Sec5.4 contract states "two threads driving the SAME sequence handle
// concurrently is an unguarded caller error, not a supported use," so this suite's own
// SSLM_BUSY-window cells drive the in-flight window through this single-threaded idiom, never a
// raw std::thread race against a shared handle.
inline SslmGpuStatus Drain(SslmGpuContext* ctx, SslmGpuSequenceHandle* seq) {
	int32_t ready = 0;
	SslmGpuStatus out_status = SSLM_OK;
	SslmGpuStatus st = SSLM_OK;
	while (!ready) st = sslm_gpu_ready(ctx, seq, /*block=*/1, &ready, &out_status);
	return st != SSLM_OK ? st : out_status;
}

constexpr uint32_t kDispatchBudget = 24;  // one whole layer's worth per call, this project's
                                           // own established kDispatchesPerLayer convention
                                           // (tools/t2132_g5_gpu_parity_gpu.cpp, gpu_port.h).

// The two-arm construction every bit-identity/trust-boundary/census cell below shares: a
// reference sequence driven ENTIRELY through single-token calls (count == 1, still the only
// granularity Sec6's own fold-round-2 re-disposition calls "mechanism-only" / trivially-current
// today AND after this design lands -- StandardsDocument.md Sec5.4 "a comparison is evidence
// only if both sides are the same quantity": both arms call the IDENTICAL public entry point,
// only the chunking of the call sequence differs), and a candidate sequence driven by ONE call
// carrying every token in `chunk`. Both start from the identical primed state (`prime`), on
// freshly created sequence handles sharing the same context_cap, so any divergence is
// attributable only to submission granularity.
struct TwoArmPromptResult {
	SslmGpuStatus ref_status = SSLM_OK, cand_status = SSLM_OK;
	SeqSnapshot ref_snap, cand_snap;
	int64_t ref_submits_delta = 0, cand_submits_delta = 0;
	int64_t ref_dispatch_delta = 0, cand_dispatch_delta = 0;
	bool ok = false;
};

inline bool PrimeSeq(SslmGpuContext* ctx, SslmGpuSequenceHandle* seq,
                      const std::vector<int32_t>& prime) {
	if (prime.empty()) return true;
	return SslmGpuSeqPrefillPromptForG5Bridge(ctx, seq, prime.data(),
	                                           static_cast<int32_t>(prime.size()),
	                                           kDispatchBudget) == SSLM_OK;
}

inline TwoArmPromptResult RunTwoArmPrompt(SslmGpuContext* ctx, SslmGpuModelHandle* model,
                                           int64_t context_cap, const std::vector<int32_t>& prime,
                                           const std::vector<int32_t>& chunk) {
	TwoArmPromptResult r;

	SslmGpuSequenceHandle* ref_seq = nullptr;
	if (sslm_gpu_seq_create(ctx, model, context_cap, &ref_seq) != SSLM_OK || !ref_seq) return r;
	if (!PrimeSeq(ctx, ref_seq, prime)) { sslm_gpu_seq_release(ctx, ref_seq); return r; }
	const int64_t ref_submits_before = superslm_test::g_gpu_chunk_submit_count_probe.load();
	const int64_t ref_dispatch_before = superslm_test::g_gpu_chunk_dispatch_count_probe.load();
	// Reference arm: chunk.size() SEPARATE calls, each carrying exactly one token -- the
	// granularity design Sec6 fold round 2 (D-SLM3610) names as trivially current at every
	// stage of this design, never itself a claim under test.
	for (size_t i = 0; i < chunk.size(); ++i) {
		r.ref_status = SslmGpuSeqPrefillPromptForG5Bridge(ctx, ref_seq, &chunk[i], 1, kDispatchBudget);
		if (r.ref_status != SSLM_OK) break;
	}
	r.ref_submits_delta = superslm_test::g_gpu_chunk_submit_count_probe.load() - ref_submits_before;
	r.ref_dispatch_delta = superslm_test::g_gpu_chunk_dispatch_count_probe.load() - ref_dispatch_before;
	if (!CaptureSnapshot(ref_seq, &r.ref_snap)) { sslm_gpu_seq_release(ctx, ref_seq); return r; }
	sslm_gpu_seq_release(ctx, ref_seq);

	SslmGpuSequenceHandle* cand_seq = nullptr;
	if (sslm_gpu_seq_create(ctx, model, context_cap, &cand_seq) != SSLM_OK || !cand_seq) return r;
	if (!PrimeSeq(ctx, cand_seq, prime)) { sslm_gpu_seq_release(ctx, cand_seq); return r; }
	const int64_t cand_submits_before = superslm_test::g_gpu_chunk_submit_count_probe.load();
	const int64_t cand_dispatch_before = superslm_test::g_gpu_chunk_dispatch_count_probe.load();
	// Candidate arm: ONE call carrying every token in `chunk` -- the batched submission this
	// design's whole Sec4/Sec5 architecture exists to make cheap.
	r.cand_status = SslmGpuSeqPrefillPromptForG5Bridge(ctx, cand_seq, chunk.data(),
	                                                    static_cast<int32_t>(chunk.size()),
	                                                    kDispatchBudget);
	r.cand_submits_delta = superslm_test::g_gpu_chunk_submit_count_probe.load() - cand_submits_before;
	r.cand_dispatch_delta = superslm_test::g_gpu_chunk_dispatch_count_probe.load() - cand_dispatch_before;
	if (!CaptureSnapshot(cand_seq, &r.cand_snap)) { sslm_gpu_seq_release(ctx, cand_seq); return r; }
	sslm_gpu_seq_release(ctx, cand_seq);

	r.ok = true;
	return r;
}

#endif  // SSLM_T2178_FIXTURE_COMMON_H
