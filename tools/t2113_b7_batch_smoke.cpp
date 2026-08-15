// T-2113 (B7): the bench proof for design Sec10 B7's own delivery -- `sslm_decode_step_batch_gpu`
// (design Sec4.3/Sec7), repaired to the batch-wide dispatch_budget + out_statuses[n] signature
// (the T-2111 fold's own specimen finding, D-SLM3344). Real D3D12 hardware, real 1.5B artifact,
// real shopkeeper adapter. Drives the PRODUCTION surface only (sslm_gpu_seq_embed_token +
// sslm_decode_step_batch_gpu/sslm_decode_step_gpu/sslm_gpu_ready) -- the read-only bench-bridge
// accessors are used only to snapshot handle state for comparison, never to write it.
//
// Design Sec10 B7's own gate: "a batch of N>=3 sequences with a mix of {no adapter, adapter A,
// adapter B} produces, per sequence, output bit-identical to that same sequence decoded alone via
// the single-sequence call -- the batch call changes submission grouping, never any sequence's
// own numerics." Plus: the dim 8 batch-async product cell's own repaired budget-cut semantics
// (BatchBudgetExhausted for a sequence reached after the batch-wide budget runs out), and the
// build seat's own additional asks (slice-invariance through the batch path, guard rejections not
// aborting the batch, throughput at N=1/2/4).
//
// NAMED LIMITATION (see gpu_1p0.cpp's own sslm_decode_step_batch_gpu header comment): this
// implementation submits every sequence in a batch through the SAME single-sequence
// RunLayerLoopGpuSubmit call the non-batch path uses -- each sequence still gets its own
// ExecuteCommandLists/Signal pair, not the ONE-command-list fusion design Sec7's own text
// describes. Bit-identity to the serial case is therefore a property of the code being literally
// the same code (proven below), not a separately-derived numerical claim; the submission-side
// throughput amortization is NOT realized, and Gate 4 below reports the real, measured
// consequence of that rather than assuming batching helps.
//
// Usage: t2113_b7_batch_smoke <model1p5b.sslm> <adapter.sslm>  (exits 0 on pass, 1 on fail)
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
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

// Snapshot of a sequence handle's own comparable state, read via the bench-bridge's READ-ONLY
// accessors -- never used to write.
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
static bool SnapshotsMatch(const SeqSnapshot& a, const SeqSnapshot& b) {
	if (a.hidden_scale.m != b.hidden_scale.m || a.hidden_scale.e != b.hidden_scale.e) return false;
	if (a.layer_index != b.layer_index) return false;
	if (a.kv_saturation_count != b.kv_saturation_count) return false;
	if (a.context_length != b.context_length) return false;
	if (a.hidden_codes.size() != b.hidden_codes.size()) return false;
	for (size_t i = 0; i < a.hidden_codes.size(); ++i) {
		if (a.hidden_codes[i] != b.hidden_codes[i]) return false;
	}
	return true;
}

// Drain a single sequence's own in-flight decode to completion (block=1), returning the decoded
// per-call status.
static SslmGpuStatus Drain(SslmGpuContext* ctx, SslmGpuSequenceHandle* seq) {
	int32_t ready = 0;
	SslmGpuStatus out_status = SSLM_OK;
	SslmGpuStatus st = SSLM_OK;
	while (!ready) st = sslm_gpu_ready(ctx, seq, /*block=*/1, &ready, &out_status);
	return st != SSLM_OK ? st : out_status;
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

int main(int argc, char** argv) {
	if (argc < 3) {
		std::fprintf(stderr, "usage: %s <model1p5b.sslm> <adapter.sslm>\n", argv[0]);
		return 2;
	}

	std::vector<uint8_t> model_bytes, adapter_bytes;
	SslmModelView view, adapter_view;
	if (!LoadModel(argv[1], &view, &model_bytes)) return 1;
	if (!LoadModel(argv[2], &adapter_view, &adapter_bytes)) return 1;

	const uint32_t num_heads = view.config.num_attention_heads;
	const uint32_t num_kv_heads = view.config.num_key_value_heads;
	const uint32_t num_hidden_layers = view.config.num_hidden_layers;
	const size_t hidden_size = view.config.hidden_size;
	const size_t head_dim = view.config.head_dim;
	const size_t intermediate_size = view.config.intermediate_size;
	const int64_t context_cap = static_cast<int64_t>(view.config.context_cap);
	const int32_t vocab_size = view.config.vocab_size;
	constexpr uint32_t kDispatchesPerLayer = 24;

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
	SslmGpuAdapterHandle* adapter = nullptr;
	CHECK(sslm_gpu_adapter_map(ctx, model, &adapter_view, &adapter) == SSLM_OK && adapter != nullptr,
	      "sslm_gpu_adapter_map(shopkeeper) failed");

	// ============================================================================
	// Gate 1a -- determinism: batch-of-3, mixed adapters {none, adapter, none},
	// bit-identical per sequence to the SAME sequence decoded alone (serial), per step,
	// over a real multi-token prefill.
	// ============================================================================
	{
		const std::vector<int32_t> tokens = {5, 128, 1000, 42, 777, 63};  // 6-token real prefill
		const int n = 3;
		const SslmGpuAdapterHandle* adapters_for_batch[3] = {nullptr, adapter, nullptr};

		// --- Serial reference: each of the 3 sequences run alone, via sslm_decode_step_gpu. ---
		std::vector<std::vector<SeqSnapshot>> serial_snaps(n);
		for (int s = 0; s < n; ++s) {
			SslmGpuSequenceHandle* seq = nullptr;
			CHECK(sslm_gpu_seq_create(ctx, model, context_cap, &seq) == SSLM_OK && seq != nullptr,
			      "serial seq_create failed");
			if (!seq) continue;
			for (int32_t tok : tokens) {
				CHECK(sslm_gpu_seq_embed_token(ctx, seq, tok) == SSLM_OK, "serial embed failed");
				const SslmGpuStatus dst = sslm_decode_step_gpu(ctx, seq, adapters_for_batch[s], 0xFFFFFFFFu);
				CHECK(dst == SSLM_OK, "serial decode did not return SSLM_OK");
				CHECK(Drain(ctx, seq) == SSLM_OK, "serial drain did not return SSLM_OK");
				serial_snaps[s].push_back(Snapshot(seq));
			}
			sslm_gpu_seq_release(ctx, seq);
		}

		// --- Batch: the SAME 3 sequences, SAME tokens, SAME adapter assignment, driven
		// through sslm_decode_step_batch_gpu, one batch call per token. ---
		SslmGpuSequenceHandle* batch_seqs[3] = {nullptr, nullptr, nullptr};
		for (int s = 0; s < n; ++s) {
			CHECK(sslm_gpu_seq_create(ctx, model, context_cap, &batch_seqs[s]) == SSLM_OK &&
			          batch_seqs[s] != nullptr,
			      "batch seq_create failed");
		}
		std::vector<std::vector<SeqSnapshot>> batch_snaps(n);
		bool batch_ok = true;
		for (int32_t tok : tokens) {
			for (int s = 0; s < n; ++s) {
				if (batch_seqs[s]) {
					CHECK(sslm_gpu_seq_embed_token(ctx, batch_seqs[s], tok) == SSLM_OK, "batch embed failed");
				}
			}
			SslmGpuStatus out_statuses[3] = {SSLM_OK, SSLM_OK, SSLM_OK};
			const SslmGpuStatus call_status = sslm_decode_step_batch_gpu(
			    ctx, batch_seqs, adapters_for_batch, 3u, 0xFFFFFFFFu, out_statuses);
			CHECK(call_status == SSLM_OK, "batch call-level status was not SSLM_OK");
			for (int s = 0; s < n; ++s) {
				CHECK(out_statuses[s] == SSLM_OK, "batch out_statuses[i] was not SSLM_OK");
				if (out_statuses[s] != SSLM_OK) batch_ok = false;
			}
			for (int s = 0; s < n; ++s) {
				CHECK(Drain(ctx, batch_seqs[s]) == SSLM_OK, "batch drain did not return SSLM_OK");
				batch_snaps[s].push_back(Snapshot(batch_seqs[s]));
			}
		}
		for (int s = 0; s < n; ++s) sslm_gpu_seq_release(ctx, batch_seqs[s]);

		bool all_match = batch_ok;
		for (int s = 0; s < n && all_match; ++s) {
			for (size_t t = 0; t < tokens.size() && all_match; ++t) {
				if (!SnapshotsMatch(serial_snaps[s][t], batch_snaps[s][t])) {
					std::fprintf(stderr, "  DIVERGENCE: seq=%d step=%zu batch != serial\n", s, t);
					all_match = false;
				}
			}
		}
		CHECK(all_match, "Gate 1a: batch-of-3 (mixed adapter) not bit-identical to serial, per step");
		std::fprintf(stderr, "  Gate 1a (batch vs serial, mixed adapter): %s\n",
		             all_match ? "OK, bit-identical every step" : "FAIL");

		// --- Gate 1b: the base (no-adapter) sequence's own batch-driven run, compared against
		// the CPU oracle directly (not merely against the GPU-serial reference) -- the "per-step
		// vs CPU oracle" half of the gate. Re-run sequence 0 (no adapter) through the batch path
		// once more, this time compared to a fresh CPU-oracle run over the identical prefill. ---
		{
			SslmGpuSequenceHandle* seq = nullptr;
			CHECK(sslm_gpu_seq_create(ctx, model, context_cap, &seq) == SSLM_OK && seq != nullptr,
			      "oracle-compare seq_create failed");
			SequenceLayerState cpu_seq{};
			std::vector<int8_t> cpu_codes(hidden_size, 0);
			cpu_seq.hidden_codes = cpu_codes.data();
			std::vector<uint8_t> cpu_ws(kv_bytes, 0);
			bool oracle_match = true;
			const SslmGpuAdapterHandle* solo_adapters[1] = {nullptr};
			SslmGpuSequenceHandle* solo_seqs[1] = {seq};
			for (int32_t tok : tokens) {
				const SslmForwardStatus cpu_st =
				    StepCpu(cpu_seq, tok, vocab_size, embed_weights, hidden_size, embed_site_constant,
				            layers.data(), num_hidden_layers, head_dim, num_kv_heads, intermediate_size,
				            context_cap, view.rope_tables, cpu_ws.data(), cpu_ws.size());
				CHECK(sslm_gpu_seq_embed_token(ctx, seq, tok) == SSLM_OK, "oracle-compare embed failed");
				SslmGpuStatus out_statuses[1] = {SSLM_OK};
				const SslmGpuStatus call_status =
				    sslm_decode_step_batch_gpu(ctx, solo_seqs, solo_adapters, 1u, 0xFFFFFFFFu, out_statuses);
				CHECK(call_status == SSLM_OK && out_statuses[0] == SSLM_OK, "oracle-compare batch call failed");
				CHECK(Drain(ctx, seq) == SSLM_OK, "oracle-compare drain failed");
				const SeqSnapshot gpu_snap = Snapshot(seq);
				bool step_ok = cpu_st == SslmForwardStatus::Ok &&
				               cpu_seq.kv_saturation_count == gpu_snap.kv_saturation_count &&
				               cpu_seq.context_length == gpu_snap.context_length;
				for (size_t i = 0; step_ok && i < hidden_size; ++i) {
					if (cpu_codes[i] != gpu_snap.hidden_codes[i]) step_ok = false;
				}
				oracle_match = oracle_match && step_ok;
			}
			CHECK(oracle_match, "Gate 1b: batch call (n=1, base) not bit-identical to the CPU oracle");
			std::fprintf(stderr, "  Gate 1b (batch call, n=1, vs CPU oracle): %s\n",
			             oracle_match ? "OK, bit-identical every step" : "FAIL");
			sslm_gpu_seq_release(ctx, seq);
		}
	}

	// ============================================================================
	// Gate 1c -- slice-invariance through the batch path: the SAME batch-of-2 session
	// (fixed prompt) driven at several batch-wide dispatch_budget granularities -- one
	// call per token (uninterrupted) down to one call per layer PER SEQUENCE's own turn
	// (the every-call extreme a batch-wide budget allows) -- compared per-step, bit-for-
	// bit, against the SAME session's own CPU oracle.
	// ============================================================================
	{
		const std::vector<int32_t> tokens = {5, 128, 1000};
		const uint32_t granularities[] = {0xFFFFFFFFu, kDispatchesPerLayer * num_hidden_layers,
		                                    kDispatchesPerLayer * 4u, kDispatchesPerLayer};
		for (uint32_t budget_layers_worth : granularities) {
			const int n = 2;
			SslmGpuSequenceHandle* seqs[2] = {nullptr, nullptr};
			for (int s = 0; s < n; ++s) {
				CHECK(sslm_gpu_seq_create(ctx, model, context_cap, &seqs[s]) == SSLM_OK && seqs[s] != nullptr,
				      "slice-invariance seq_create failed");
			}
			const SslmGpuAdapterHandle* adapters2[2] = {nullptr, nullptr};

			// CPU oracle, one instance per sequence (both sequences run the identical prompt).
			SequenceLayerState cpu_seq[2]{};
			std::vector<int8_t> cpu_codes[2] = {std::vector<int8_t>(hidden_size, 0),
			                                     std::vector<int8_t>(hidden_size, 0)};
			std::vector<uint8_t> cpu_ws[2] = {std::vector<uint8_t>(kv_bytes, 0),
			                                   std::vector<uint8_t>(kv_bytes, 0)};
			cpu_seq[0].hidden_codes = cpu_codes[0].data();
			cpu_seq[1].hidden_codes = cpu_codes[1].data();

			bool all_match = true;
			for (int32_t tok : tokens) {
				for (int s = 0; s < n; ++s) {
					const SslmForwardStatus cpu_st = StepCpu(
					    cpu_seq[s], tok, vocab_size, embed_weights, hidden_size, embed_site_constant,
					    layers.data(), num_hidden_layers, head_dim, num_kv_heads, intermediate_size,
					    context_cap, view.rope_tables, cpu_ws[s].data(), cpu_ws[s].size());
					CHECK(cpu_st == SslmForwardStatus::Ok, "slice-invariance CPU oracle step failed");
				}
				for (int s = 0; s < n; ++s) {
					CHECK(sslm_gpu_seq_embed_token(ctx, seqs[s], tok) == SSLM_OK,
					      "slice-invariance embed failed");
				}
				// Drive this ONE token to completion, possibly across MULTIPLE batch calls at
				// this granularity (the every-layer extreme needs num_hidden_layers calls/token).
				bool token_done[2] = {false, false};
				int guard_iterations = 0;
				while (!(token_done[0] && token_done[1]) && guard_iterations < 200) {
					++guard_iterations;
					SslmGpuStatus out_statuses[2] = {SSLM_OK, SSLM_OK};
					const SslmGpuStatus call_status = sslm_decode_step_batch_gpu(
					    ctx, seqs, adapters2, 2u, budget_layers_worth, out_statuses);
					CHECK(call_status == SSLM_OK, "slice-invariance batch call-level status not Ok");
					for (int s = 0; s < n; ++s) {
						if (token_done[s]) continue;
						// SSLM_BATCH_BUDGET_EXHAUSTED is a LEGITIMATE per-call outcome here, not a
						// failure: at a fine granularity (budget_layers_worth covering only 1-4
						// layers, batch-wide, for 2 sequences), design Sec7's own strict-array-order
						// consumption can let seqs[0] take the WHOLE per-call budget, leaving
						// nothing for seqs[1] on THIS call -- seqs[1] simply gets its own turn on a
						// LATER call (the while loop's own next iteration), never dropping any
						// content or diverging (proven below by the per-step CPU-oracle comparison,
						// which stays bit-identical regardless). Only a genuine rejection (anything
						// other than Ok or the expected exhaustion) is a real failure.
						CHECK(out_statuses[s] == SSLM_OK || out_statuses[s] == SSLM_BATCH_BUDGET_EXHAUSTED,
						      "slice-invariance out_statuses[i] was a genuine rejection, not Ok/Exhausted");
						if (out_statuses[s] != SSLM_OK) continue;  // this sequence's own turn is later
						CHECK(Drain(ctx, seqs[s]) == SSLM_OK, "slice-invariance drain not Ok");
						if (*SslmGpuSeqHandleLayerIndexForBench(seqs[s]) >= num_hidden_layers) {
							token_done[s] = true;
						}
					}
				}
				CHECK(guard_iterations < 200, "slice-invariance token never completed (runaway loop)");
				for (int s = 0; s < n; ++s) {
					const SeqSnapshot gpu_snap = Snapshot(seqs[s]);
					bool step_ok = cpu_seq[s].kv_saturation_count == gpu_snap.kv_saturation_count &&
					               cpu_seq[s].context_length == gpu_snap.context_length;
					for (size_t i = 0; step_ok && i < hidden_size; ++i) {
						if (cpu_codes[s][i] != gpu_snap.hidden_codes[i]) step_ok = false;
					}
					all_match = all_match && step_ok;
				}
			}
			CHECK(all_match, "Gate 1c: slice-invariance through the batch path diverged from the CPU oracle");
			std::fprintf(stderr, "  Gate 1c granularity budget_layers_worth=%u: %s\n", budget_layers_worth,
			             all_match ? "OK, bit-identical every step" : "FAIL");
			for (int s = 0; s < n; ++s) sslm_gpu_seq_release(ctx, seqs[s]);
		}
	}

	// ============================================================================
	// Gate 2 -- the dim 8 batch-async budget-cut cell (design Sec11 dim8, cell 2; the
	// probe's own cell_dim8_batch_async.c worked example, real model this time instead
	// of a tiny fixture): n=4 real sequences, batch-wide budget covering 2 FULL tokens
	// plus exactly ONE further layer -- seqs[0]/[1] complete a full token each,
	// seqs[2] gets exactly one layer (still SSLM_OK -- Ok means "some dispatches
	// recorded," not "token complete"), seqs[3] gets SSLM_BATCH_BUDGET_EXHAUSTED and
	// stays Idle (verified by a follow-up decode succeeding as a fresh sequence would).
	// ============================================================================
	{
		const uint32_t full_token = kDispatchesPerLayer * num_hidden_layers;
		const uint32_t batch_budget = full_token * 2 + kDispatchesPerLayer;  // 2 full + 1 layer
		SslmGpuSequenceHandle* seqs[4] = {nullptr, nullptr, nullptr, nullptr};
		for (int i = 0; i < 4; ++i) {
			CHECK(sslm_gpu_seq_create(ctx, model, context_cap, &seqs[i]) == SSLM_OK && seqs[i] != nullptr,
			      "dim8-cut seq_create failed");
			if (seqs[i]) CHECK(sslm_gpu_seq_embed_token(ctx, seqs[i], 5) == SSLM_OK, "dim8-cut embed failed");
		}
		const SslmGpuAdapterHandle* adapters4[4] = {nullptr, nullptr, nullptr, nullptr};
		SslmGpuStatus out_statuses[4] = {SSLM_OK, SSLM_OK, SSLM_OK, SSLM_OK};
		const SslmGpuStatus call_status =
		    sslm_decode_step_batch_gpu(ctx, seqs, adapters4, 4u, batch_budget, out_statuses);
		CHECK(call_status == SSLM_OK, "dim8-cut call-level status not Ok");
		CHECK(out_statuses[0] == SSLM_OK, "dim8-cut seqs[0] did not read Ok");
		CHECK(out_statuses[1] == SSLM_OK, "dim8-cut seqs[1] did not read Ok");
		CHECK(out_statuses[2] == SSLM_OK, "dim8-cut seqs[2] (partial, one layer) did not read Ok");
		CHECK(out_statuses[3] == SSLM_BATCH_BUDGET_EXHAUSTED,
		      "dim8-cut seqs[3] did not read SSLM_BATCH_BUDGET_EXHAUSTED");
		if (out_statuses[0] == SSLM_OK) CHECK(Drain(ctx, seqs[0]) == SSLM_OK, "dim8-cut drain seqs[0] failed");
		if (out_statuses[1] == SSLM_OK) CHECK(Drain(ctx, seqs[1]) == SSLM_OK, "dim8-cut drain seqs[1] failed");
		if (out_statuses[2] == SSLM_OK) CHECK(Drain(ctx, seqs[2]) == SSLM_OK, "dim8-cut drain seqs[2] failed");
		CHECK(*SslmGpuSeqHandleLayerIndexForBench(seqs[0]) == num_hidden_layers,
		      "dim8-cut seqs[0] did not complete its own full token");
		CHECK(*SslmGpuSeqHandleLayerIndexForBench(seqs[1]) == num_hidden_layers,
		      "dim8-cut seqs[1] did not complete its own full token");
		CHECK(*SslmGpuSeqHandleLayerIndexForBench(seqs[2]) == 1,
		      "dim8-cut seqs[2] did not record exactly its own one layer");
		// seqs[3] never submitted -- state stays Idle: a follow-up single-sequence decode call
		// must succeed exactly as it would against a never-touched fresh sequence.
		if (seqs[3]) {
			const SslmGpuStatus follow_up = sslm_decode_step_gpu(ctx, seqs[3], nullptr, kDispatchesPerLayer);
			CHECK(follow_up == SSLM_OK, "dim8-cut seqs[3] was not a genuinely fresh, untouched Idle sequence");
			if (follow_up == SSLM_OK) Drain(ctx, seqs[3]);
		}
		std::fprintf(stderr, "  Gate 2 (dim8 batch-budget-cut, real model): out_statuses=[%d,%d,%d,%d]\n",
		             (int)out_statuses[0], (int)out_statuses[1], (int)out_statuses[2], (int)out_statuses[3]);
		for (int i = 0; i < 4; ++i)
			if (seqs[i]) sslm_gpu_seq_release(ctx, seqs[i]);
	}

	// ============================================================================
	// Gate 3 -- a guard rejection for one sequence does not abort the batch: n=3, seqs[1]
	// is a null handle (SslmGpuSequenceHandle*)nullptr -- seqs[0]/[2] must still be
	// recorded and reach SSLM_OK.
	// ============================================================================
	{
		SslmGpuSequenceHandle* seq0 = nullptr;
		SslmGpuSequenceHandle* seq2 = nullptr;
		CHECK(sslm_gpu_seq_create(ctx, model, context_cap, &seq0) == SSLM_OK && seq0 != nullptr,
		      "guard-mid-batch seq0 create failed");
		CHECK(sslm_gpu_seq_create(ctx, model, context_cap, &seq2) == SSLM_OK && seq2 != nullptr,
		      "guard-mid-batch seq2 create failed");
		if (seq0) CHECK(sslm_gpu_seq_embed_token(ctx, seq0, 5) == SSLM_OK, "guard-mid-batch embed0 failed");
		if (seq2) CHECK(sslm_gpu_seq_embed_token(ctx, seq2, 5) == SSLM_OK, "guard-mid-batch embed2 failed");
		SslmGpuSequenceHandle* seqs3[3] = {seq0, nullptr, seq2};
		const SslmGpuAdapterHandle* adapters3[3] = {nullptr, nullptr, nullptr};
		SslmGpuStatus out_statuses[3] = {SSLM_OK, SSLM_OK, SSLM_OK};
		const SslmGpuStatus call_status =
		    sslm_decode_step_batch_gpu(ctx, seqs3, adapters3, 3u, 0xFFFFFFFFu, out_statuses);
		CHECK(call_status == SSLM_OK, "guard-mid-batch call-level status not Ok");
		CHECK(out_statuses[0] == SSLM_OK, "guard-mid-batch seqs[0] did not read Ok");
		CHECK(out_statuses[1] == SSLM_SEQUENCE_KV_BUFFER_MISMATCH,
		      "guard-mid-batch seqs[1] (null) did not read SSLM_SEQUENCE_KV_BUFFER_MISMATCH");
		CHECK(out_statuses[2] == SSLM_OK, "guard-mid-batch seqs[2] (AFTER the rejected slot) did not read Ok");
		std::fprintf(stderr, "  Gate 3 (guard rejection mid-batch does not abort): out_statuses=[%d,%d,%d]\n",
		             (int)out_statuses[0], (int)out_statuses[1], (int)out_statuses[2]);
		if (seq0) { Drain(ctx, seq0); sslm_gpu_seq_release(ctx, seq0); }
		if (seq2) { Drain(ctx, seq2); sslm_gpu_seq_release(ctx, seq2); }
	}

	// ============================================================================
	// Gate 4 -- batch throughput at N=1/2/4, 3-run each, spread reported, vs B5's own
	// serial figure (measured, not assumed favorable -- design Sec13's own "UNDERIVED"
	// framing, and this implementation's own named limitation above).
	// ============================================================================
	{
		const uint32_t full_token = kDispatchesPerLayer * num_hidden_layers;
		constexpr int kSteps = 32;  // per run; batching N sequences means N*kSteps decode-steps
		for (int n : {1, 2, 4}) {
			double run_tok_s[3] = {0, 0, 0};
			bool timing_ok = true;
			for (int run = 0; run < 3 && timing_ok; ++run) {
				std::vector<SslmGpuSequenceHandle*> seqs(n, nullptr);
				std::vector<const SslmGpuAdapterHandle*> adapters(n, nullptr);
				for (int i = 0; i < n; ++i) {
					if (sslm_gpu_seq_create(ctx, model, context_cap, &seqs[i]) != SSLM_OK || !seqs[i]) {
						timing_ok = false;
					}
				}
				// One warmup step, discarded (this codebase's own established convention).
				for (int w = 0; w < 1 && timing_ok; ++w) {
					for (int i = 0; i < n; ++i) sslm_gpu_seq_embed_token(ctx, seqs[i], 5);
					std::vector<SslmGpuStatus> st(n, SSLM_OK);
					sslm_decode_step_batch_gpu(ctx, seqs.data(), adapters.data(), static_cast<uint32_t>(n),
					                            full_token * static_cast<uint32_t>(n), st.data());
					for (int i = 0; i < n; ++i) Drain(ctx, seqs[i]);
				}
				const auto t0 = std::chrono::steady_clock::now();
				for (int s = 0; s < kSteps && timing_ok; ++s) {
					for (int i = 0; i < n; ++i) sslm_gpu_seq_embed_token(ctx, seqs[i], 5);
					std::vector<SslmGpuStatus> st(n, SSLM_OK);
					const SslmGpuStatus cst = sslm_decode_step_batch_gpu(
					    ctx, seqs.data(), adapters.data(), static_cast<uint32_t>(n),
					    full_token * static_cast<uint32_t>(n), st.data());
					if (cst != SSLM_OK) timing_ok = false;
					for (int i = 0; i < n; ++i) {
						if (Drain(ctx, seqs[i]) != SSLM_OK) timing_ok = false;
					}
				}
				const auto t1 = std::chrono::steady_clock::now();
				const double elapsed_s = std::chrono::duration<double>(t1 - t0).count();
				// Aggregate tokens/s: n sequences x kSteps tokens each, over the wall-clock time
				// of that many batch calls.
				run_tok_s[run] = (static_cast<double>(n) * kSteps) / elapsed_s;
				for (int i = 0; i < n; ++i)
					if (seqs[i]) sslm_gpu_seq_release(ctx, seqs[i]);
			}
			CHECK(timing_ok, "batch throughput protocol did not complete cleanly");
			std::fprintf(stderr,
			             "  THROUGHPUT batch N=%d (aggregate tok/s, %d steps/sequence, 3 runs): "
			             "run1=%.2f run2=%.2f run3=%.2f\n",
			             n, kSteps, run_tok_s[0], run_tok_s[1], run_tok_s[2]);
		}
	}

	sslm_gpu_adapter_unmap(ctx, adapter);
	sslm_gpu_model_unmap(ctx, model);
	sslm_gpu_context_destroy(ctx);

	std::fprintf(stderr, "T-2113 B7 batch smoke: checks=%d failures=%d\n", g_checks, g_failures);
	return g_failures ? 1 : 0;
}
