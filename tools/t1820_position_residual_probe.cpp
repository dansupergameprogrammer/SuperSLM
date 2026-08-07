// t1820_position_residual_probe.cpp -- T-1820: dump the compiled engine's REAL residual
// stream at EVERY prompt position (not only the last), as raw int8 codes plus each state's
// own CarriedScale, so the position axis of the residual-stream drift can be measured.
//
// This file is tools/t1795_residual_probe.cpp (branch claude/t1795-residual-drift@9eef187,
// read in full, not modified) with exactly two changes:
//   1. every prompt position is stepped one layer at a time and its 29 committed residual
//      states are captured, where T-1795 stepped only the last position and ran the earlier
//      ones through a single RunLayerLoop(layer_budget=28) call;
//   2. the dump carries the committed int8 CODES and the CarriedScale{m,e} rather than a
//      dequantized float text row -- exact rather than 6-significant-digit, and it makes the
//      state's own quantization step available to the analysis.
// T-1795's checkpoint-layer manual replay and its Phase-B stage capture are NOT carried
// over; this probe measures the committed residual stream only. No arithmetic is
// reimplemented here: every forward call is PRODUCTION's own RunLayerLoop.
//
// Self-checks, both executed every run:
//   - budget invariance PER POSITION: after each position's 28x RunLayerLoop(layer_budget=1)
//     stepping, an independent second engine instance replays the whole prefix through
//     position t with a single RunLayerLoop(layer_budget=28) call per position, and the two
//     committed states are compared bit-for-bit. T-1795 ran this check once, at the last
//     position; the per-position stepping this probe introduces is exactly what it gates,
//     so it is run at every position here.
//   - the dumped codes/scale are the committed sequence state itself, copied, not recomputed.
//
// Dump format (binary, little-endian, the platform's own layout):
//   int32 num_positions, int32 num_states (=29), int32 hidden_size
//   int32 token_id[num_positions]
//   then num_positions * num_states records of: int64 m, int64 e, int8 code[hidden_size]

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <span>
#include <string>
#include <vector>

#include "superslm/artifact.h"
#include "superslm/checked_chain_funnel.h"
#include "superslm/forward_sites.h"
#include "superslm/intmath.h"
#include "superslm/matmul.h"
#include "superslm/model.h"
#include "superslm/tokenizer.h"
#include "sslm_marshal.h"

using namespace superslm;
using superslm_marshal::LayerBacking;
using superslm_marshal::MarshalLayer;
using superslm_marshal::PreflightScanWscFolds;
using superslm_marshal::ReadCarriedScale;
using superslm_marshal::ReadFile;

namespace {

struct StateRecord {
	int64_t m = 0;
	int64_t e = 0;
	std::vector<int8_t> codes;
};

}  // namespace

int main(int argc, char** argv) {
	if (argc < 5) {
		std::fprintf(stderr,
		             "usage: %s <model.sslm> <tokenizer.sslm> \"<prompt>\" <prompt_id> --dump-dir <dir>\n",
		             argv[0]);
		return 2;
	}
	const std::string model_path = argv[1];
	const std::string tokenizer_path = argv[2];
	const std::string prompt = argv[3];
	const std::string prompt_id = argv[4];
	std::string dump_dir = "out/t1820";
	for (int i = 5; i < argc; ++i) {
		if (std::strcmp(argv[i], "--dump-dir") == 0 && i + 1 < argc) dump_dir = argv[++i];
	}

	std::vector<uint8_t> tok_bytes;
	if (!ReadFile(tokenizer_path.c_str(), tok_bytes)) {
		std::fprintf(stderr, "FAILED at stage=tokenizer_file_read\n");
		return 1;
	}
	SslmArtifact tok_artifact;
	SslmError tok_open_err;
	if (SslmArtifact::OpenFromMemory(tok_bytes.data(), tok_bytes.size(), tok_artifact, &tok_open_err) !=
	    SslmStatus::Ok) {
		std::fprintf(stderr, "FAILED at stage=tokenizer_artifact_open\n");
		return 1;
	}
	TokenizerView tokenizer;
	std::string tok_err;
	if (!TokenizerView::Open(tok_artifact, tokenizer, &tok_err)) {
		std::fprintf(stderr, "FAILED at stage=tokenizer_view_open: %s\n", tok_err.c_str());
		return 1;
	}
	const std::vector<int32_t> prompt_tokens = tokenizer.Encode(prompt);
	if (prompt_tokens.empty()) {
		std::fprintf(stderr, "FAILED at stage=tokenizer_encode\n");
		return 1;
	}

	std::vector<uint8_t> model_bytes;
	if (!ReadFile(model_path.c_str(), model_bytes)) {
		std::fprintf(stderr, "FAILED at stage=model_file_read\n");
		return 1;
	}
	SslmModelView model_view;
	std::string model_err;
	if (SslmModel::Load(model_bytes.data(), model_bytes.size(), model_view, &model_err) !=
	    SslmModelStatus::Ok) {
		std::fprintf(stderr, "FAILED at stage=model_load: %s\n", model_err.c_str());
		return 1;
	}

	const uint32_t num_heads = model_view.config.num_attention_heads;
	const uint32_t num_kv_heads = model_view.config.num_key_value_heads;
	const uint32_t num_hidden_layers = model_view.config.num_hidden_layers;
	const size_t hidden_size = model_view.config.hidden_size;
	const size_t head_dim = model_view.config.head_dim;
	const size_t intermediate_size = model_view.config.intermediate_size;

	std::printf("model loaded: hidden_size=%zu layers=%u heads=%u/%u head_dim=%zu prompt_tokens=%zu (%s)\n",
	            hidden_size, num_hidden_layers, num_heads, num_kv_heads, head_dim, prompt_tokens.size(),
	            prompt_id.c_str());

	PreflightScanWscFolds(model_view);

	std::vector<LayerBacking> backings(num_hidden_layers);
	std::vector<LayerWeights> layers(num_hidden_layers);
	for (uint32_t l = 0; l < num_hidden_layers; ++l) {
		std::string marshal_err;
		if (!MarshalLayer(model_view, l, num_heads, num_kv_heads, backings[l], layers[l], &marshal_err)) {
			std::fprintf(stderr, "FAILED at stage=layer_weights_marshal: layer=%u: %s\n", l,
			             marshal_err.c_str());
			return 1;
		}
	}

	const SslmTensorView* embed_w = model_view.weights.Tensor("embed");
	if (!embed_w) {
		std::fprintf(stderr, "FAILED at stage=head_marshal: missing embed tensor\n");
		return 1;
	}
	bool ok = true;
	const CarriedScale embed_site_constant =
	    ReadCarriedScale(model_view.composition_constants, "embed", &ok);
	if (!ok) {
		std::fprintf(stderr, "FAILED at stage=head_marshal: missing embed site constant\n");
		return 1;
	}
	const int8_t* const embed_weights = reinterpret_cast<const int8_t*>(embed_w->data);

	const int64_t context_cap = static_cast<int64_t>(model_view.config.context_cap);
	const size_t kv_bytes = static_cast<size_t>(num_hidden_layers) * static_cast<size_t>(context_cap) *
	                        num_kv_heads * head_dim * 2;

	// --- capture instance: every position stepped one layer at a time ---
	std::vector<uint8_t> workspace(kv_bytes);
	std::vector<int8_t> hidden_codes(hidden_size);
	SequenceLayerState seq;
	seq.hidden_codes = hidden_codes.data();

	// --- check instance: every position run with a single full-budget call ---
	std::vector<uint8_t> chk_workspace(kv_bytes);
	std::vector<int8_t> chk_hidden_codes(hidden_size);
	SequenceLayerState chk_seq;
	chk_seq.hidden_codes = chk_hidden_codes.data();

	auto Embed = [&](SequenceLayerState& s, int32_t token) -> SslmForwardStatus {
		std::vector<int8_t> embed_codes(hidden_size);
		CarriedScale embed_scale{};
		const SslmForwardStatus est =
		    EmbedEntry(token, static_cast<int32_t>(model_view.config.vocab_size), embed_weights,
		               hidden_size, embed_site_constant, embed_codes.data(), &embed_scale);
		if (est != SslmForwardStatus::Ok) return est;
		for (size_t i = 0; i < hidden_size; ++i) s.hidden_codes[i] = embed_codes[i];
		s.hidden_scale = embed_scale;
		s.layer_index = 0;
		return SslmForwardStatus::Ok;
	};

	const size_t num_positions = prompt_tokens.size();
	const size_t num_states = static_cast<size_t>(num_hidden_layers) + 1;
	std::vector<StateRecord> records;
	records.reserve(num_positions * num_states);
	size_t invariance_checks = 0;

	for (size_t t = 0; t < num_positions; ++t) {
		SslmForwardStatus st = Embed(seq, prompt_tokens[t]);
		if (st != SslmForwardStatus::Ok) {
			std::fprintf(stderr, "FAILED at stage=embed: position=%zu status=%s\n", t,
			             SslmForwardStatusName(st));
			return 1;
		}
		auto push = [&]() {
			StateRecord r;
			r.m = seq.hidden_scale.m;
			r.e = seq.hidden_scale.e;
			r.codes.assign(seq.hidden_codes, seq.hidden_codes + hidden_size);
			records.push_back(std::move(r));
		};
		push();
		for (uint32_t step = 0; step < num_hidden_layers; ++step) {
			st = RunLayerLoop(seq, layers.data(), num_hidden_layers, /*layer_budget=*/1, hidden_size,
			                  head_dim, num_kv_heads, intermediate_size, context_cap,
			                  model_view.rope_tables, workspace.data(), workspace.size());
			if (st != SslmForwardStatus::Ok) {
				std::fprintf(stderr, "FAILED at stage=layer_step: position=%zu layer=%u status=%s\n", t,
				             step, SslmForwardStatusName(st));
				return 1;
			}
			push();
		}

		// Per-position budget-invariance self-check on an independent engine instance.
		st = Embed(chk_seq, prompt_tokens[t]);
		if (st != SslmForwardStatus::Ok) {
			std::fprintf(stderr, "FAILED at stage=chk_embed: position=%zu\n", t);
			return 1;
		}
		st = RunLayerLoop(chk_seq, layers.data(), num_hidden_layers,
		                  /*layer_budget=*/num_hidden_layers, hidden_size, head_dim, num_kv_heads,
		                  intermediate_size, context_cap, model_view.rope_tables, chk_workspace.data(),
		                  chk_workspace.size());
		if (st != SslmForwardStatus::Ok) {
			std::fprintf(stderr, "FAILED at stage=chk_layers: position=%zu status=%s\n", t,
			             SslmForwardStatusName(st));
			return 1;
		}
		const bool match = std::memcmp(chk_seq.hidden_codes, seq.hidden_codes, hidden_size) == 0 &&
		                   chk_seq.hidden_scale.m == seq.hidden_scale.m &&
		                   chk_seq.hidden_scale.e == seq.hidden_scale.e;
		if (!match) {
			std::fprintf(stderr, "FAILED at stage=budget_invariance: position=%zu\n", t);
			return 1;
		}
		++invariance_checks;
	}

	std::printf("budget_invariance_check: %zu/%zu positions MATCH (28x layer_budget=1 vs one "
	            "layer_budget=28, independent engine instance)\n",
	            invariance_checks, num_positions);

	const std::string path = dump_dir + "/" + prompt_id + "_engine.bin";
	std::ofstream f(path, std::ios::binary);
	if (!f) {
		std::fprintf(stderr, "FAILED at stage=dump_open: %s\n", path.c_str());
		return 1;
	}
	const int32_t np = static_cast<int32_t>(num_positions);
	const int32_t ns = static_cast<int32_t>(num_states);
	const int32_t nh = static_cast<int32_t>(hidden_size);
	f.write(reinterpret_cast<const char*>(&np), sizeof(np));
	f.write(reinterpret_cast<const char*>(&ns), sizeof(ns));
	f.write(reinterpret_cast<const char*>(&nh), sizeof(nh));
	for (size_t t = 0; t < num_positions; ++t) {
		const int32_t tok = prompt_tokens[t];
		f.write(reinterpret_cast<const char*>(&tok), sizeof(tok));
	}
	for (const StateRecord& r : records) {
		f.write(reinterpret_cast<const char*>(&r.m), sizeof(r.m));
		f.write(reinterpret_cast<const char*>(&r.e), sizeof(r.e));
		f.write(reinterpret_cast<const char*>(r.codes.data()),
		        static_cast<std::streamsize>(r.codes.size()));
	}
	f.close();
	std::printf("engine dump written: %s (%zu positions x %zu states x %zu channels)\n", path.c_str(),
	            num_positions, num_states, hidden_size);
	return 0;
}
