// t1797_multipos_probe.cpp -- T-1797 E0: dump the compiled engine's REAL residual stream
// (embedding output + all 28 decoder-layer committed outputs = 29 states) at EVERY prompt
// position, so the drift against the independent float32 reference can be characterized as
// a population over (prompt, position) samples rather than a single last-token anecdote.
//
// This file is a stripped copy of tools/t1795_residual_probe.cpp (branch
// claude/t1795-residual-drift@9eef187, read in full): the manual replay and checkpoint
// machinery are removed -- EVERY state dumped here is production's own committed
// hidden_codes/hidden_scale from RunLayerLoop(layer_budget=1) stepping, dequantized by
// CarriedScale's own documented meaning (value = code * m * 2^e, checked_chain_funnel.h:66).
// No arithmetic is reimplemented. The one self-check retained is T-1795's own redundant
// budget-invariance cross-check: a fresh full-budget (layer_budget=28 per position) replay
// of the whole prompt must land bit-for-bit on the stepped run's final state.
//
// Dump: raw little-endian float32, layout [position][state 0..28][hidden_size], plus a
// sidecar .meta text file carrying the counts.

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include "superslm/artifact.h"
#include "superslm/checked_chain_funnel.h"
#include "superslm/forward_sites.h"
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

double Dequant(int8_t code, CarriedScale scale) {
	return static_cast<double>(code) * static_cast<double>(scale.m) *
	       std::pow(2.0, static_cast<double>(scale.e));
}

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
	std::string dump_dir = "out/t1797";
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
	if (SslmArtifact::OpenFromMemory(tok_bytes.data(), tok_bytes.size(), tok_artifact,
	                                 &tok_open_err) != SslmStatus::Ok) {
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
	std::printf("model loaded: hidden_size=%zu layers=%u prompt_tokens=%zu (%s)\n", hidden_size,
	            num_hidden_layers, prompt_tokens.size(), prompt_id.c_str());

	PreflightScanWscFolds(model_view);

	std::vector<LayerBacking> backings(num_hidden_layers);
	std::vector<LayerWeights> layers(num_hidden_layers);
	for (uint32_t l = 0; l < num_hidden_layers; ++l) {
		std::string marshal_err;
		if (!MarshalLayer(model_view, l, num_heads, num_kv_heads, backings[l], layers[l],
		                  &marshal_err)) {
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
	CarriedScale embed_site_constant = ReadCarriedScale(model_view.composition_constants, "embed", &ok);
	if (!ok) {
		std::fprintf(stderr, "FAILED at stage=head_marshal: missing embed site constant\n");
		return 1;
	}
	const int8_t* embed_weights = reinterpret_cast<const int8_t*>(embed_w->data);

	const int64_t context_cap = static_cast<int64_t>(model_view.config.context_cap);
	const size_t kv_bytes = static_cast<size_t>(num_hidden_layers) *
	                        static_cast<size_t>(context_cap) * num_kv_heads * head_dim * 2;

	std::vector<uint8_t> workspace(kv_bytes);
	std::vector<int8_t> hidden_codes(hidden_size);
	SequenceLayerState seq;
	seq.hidden_codes = hidden_codes.data();

	auto EmbedToken = [&](int32_t token, SequenceLayerState& s, int8_t* codes) -> SslmForwardStatus {
		std::vector<int8_t> embed_codes(hidden_size);
		CarriedScale embed_scale{};
		const SslmForwardStatus est =
		    EmbedEntry(token, static_cast<int32_t>(model_view.config.vocab_size), embed_weights,
		               hidden_size, embed_site_constant, embed_codes.data(), &embed_scale);
		if (est != SslmForwardStatus::Ok) return est;
		for (size_t i = 0; i < hidden_size; ++i) codes[i] = embed_codes[i];
		s.hidden_scale = embed_scale;
		s.layer_index = 0;
		return SslmForwardStatus::Ok;
	};

	const size_t n_pos = prompt_tokens.size();
	const size_t n_states = static_cast<size_t>(num_hidden_layers) + 1;
	std::vector<float> dump;  // [pos][state][hidden]
	dump.reserve(n_pos * n_states * hidden_size);

	for (size_t t = 0; t < n_pos; ++t) {
		SslmForwardStatus st = EmbedToken(prompt_tokens[t], seq, seq.hidden_codes);
		if (st != SslmForwardStatus::Ok) {
			std::fprintf(stderr, "FAILED at stage=embed: position=%zu status=%s\n", t,
			             SslmForwardStatusName(st));
			return 1;
		}
		for (size_t i = 0; i < hidden_size; ++i)
			dump.push_back(static_cast<float>(Dequant(seq.hidden_codes[i], seq.hidden_scale)));
		for (uint32_t step = 0; step < num_hidden_layers; ++step) {
			st = RunLayerLoop(seq, layers.data(), num_hidden_layers, /*layer_budget=*/1, hidden_size,
			                  head_dim, num_kv_heads, intermediate_size, context_cap,
			                  model_view.rope_tables, workspace.data(), workspace.size());
			if (st != SslmForwardStatus::Ok) {
				std::fprintf(stderr, "FAILED at stage=layers: position=%zu layer=%u status=%s\n", t,
				             step, SslmForwardStatusName(st));
				return 1;
			}
			for (size_t i = 0; i < hidden_size; ++i)
				dump.push_back(static_cast<float>(Dequant(seq.hidden_codes[i], seq.hidden_scale)));
		}
	}

	// Redundant budget-invariance cross-check (T-1795's own): fresh replay, one
	// layer_budget=28 call per position, final committed state must match bit-for-bit.
	{
		std::vector<uint8_t> check_workspace(kv_bytes);
		std::vector<int8_t> check_codes(hidden_size);
		SequenceLayerState check_seq;
		check_seq.hidden_codes = check_codes.data();
		for (size_t t = 0; t < n_pos; ++t) {
			SslmForwardStatus st = EmbedToken(prompt_tokens[t], check_seq, check_seq.hidden_codes);
			if (st != SslmForwardStatus::Ok) {
				std::fprintf(stderr, "FAILED at stage=check_embed\n");
				return 1;
			}
			st = RunLayerLoop(check_seq, layers.data(), num_hidden_layers, num_hidden_layers,
			                  hidden_size, head_dim, num_kv_heads, intermediate_size, context_cap,
			                  model_view.rope_tables, check_workspace.data(), check_workspace.size());
			if (st != SslmForwardStatus::Ok) {
				std::fprintf(stderr, "FAILED at stage=check_layers\n");
				return 1;
			}
		}
		const bool match =
		    std::memcmp(check_seq.hidden_codes, seq.hidden_codes, hidden_size) == 0 &&
		    check_seq.hidden_scale.m == seq.hidden_scale.m &&
		    check_seq.hidden_scale.e == seq.hidden_scale.e;
		std::printf("budget_invariance_check: %s\n", match ? "MATCH" : "MISMATCH");
		if (!match) return 1;
	}

	{
		const std::string bin_path = dump_dir + "/" + prompt_id + "_eng.bin";
		std::ofstream f(bin_path, std::ios::binary);
		if (!f) {
			std::fprintf(stderr, "FAILED at stage=dump_open: %s\n", bin_path.c_str());
			return 1;
		}
		f.write(reinterpret_cast<const char*>(dump.data()),
		        static_cast<std::streamsize>(dump.size() * sizeof(float)));
		const std::string meta_path = dump_dir + "/" + prompt_id + "_eng.meta";
		std::ofstream m(meta_path);
		m << "positions " << n_pos << "\nstates " << n_states << "\nhidden " << hidden_size
		  << "\ndtype float32-le\nlayout pos,state,hidden\n";
		std::printf("dump written: %s (%zu positions x %zu states x %zu)\n", bin_path.c_str(), n_pos,
		            n_states, hidden_size);
	}
	return 0;
}
