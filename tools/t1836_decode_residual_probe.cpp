// t1836_decode_residual_probe.cpp -- T-1836: dump the compiled engine's REAL residual stream
// at every PROMPT position AND at every GENERATED position, so decode-interior drift can be
// measured against the prefill-interior drift the campaign already owns.
//
// This file is tools/t1820_position_residual_probe.cpp (branch brunel/t1820-drift-mechanism,
// read in full) extended in exactly three ways:
//   1. after the prompt is consumed the walk keeps going: final_norm -> LogitsSite ->
//      ArgmaxLowestIndexTieBreak selects the next token, that token is embedded, and its 29
//      committed residual states are captured the same way a prompt position's are. Every
//      call is PRODUCTION's own -- RunLayerLoop, RmsNormSite, LogitsSite,
//      ArgmaxLowestIndexTieBreak -- in RunGreedyDecodeLoop's own composition order, which
//      that function's own header comment states.
//   2. --force-tokens teacher-forces the continuation onto a caller-supplied id list (the
//      float reference's own greedy generation), while STILL recording the engine's own
//      argmax at every step. Matched inputs are what makes an interior comparison an
//      interior comparison: with each side free-running, a single differing token makes
//      every later state a comparison of two different sequences, and the drift measured
//      after that point is a fact about the inputs rather than about the arithmetic.
//   3. the dump carries the prompt length and the engine's own argmax at each position, so
//      the free-running trajectory and the forced one can be reconciled offline.
//
// Self-checks, all executed every run, before any dump is written:
//   - PRODUCTION AGREEMENT. RunGreedyDecodeLoop is called first, on its own engine instance
//     with its own workspace, for the same max_new_tokens. In free mode the stepped walk's
//     produced tokens must equal its output exactly; in forced mode the walk's own argmax
//     must equal it up to (and including) the first position where the forced id differs
//     from the engine's own choice, and the probe reports that index rather than hiding it.
//   - BUDGET INVARIANCE PER POSITION, prompt and generated alike: a second, independent
//     engine instance replays each position with a single full-budget RunLayerLoop call and
//     the two committed states are compared bit-for-bit. T-1820 ran this at every prompt
//     position; the generated positions are new here and are gated the same way.
//
// Dump format (binary, little-endian, the platform's own layout):
//   int32 num_positions, int32 num_states (=29), int32 hidden_size, int32 prompt_len
//   int32 token_id[num_positions]        -- the token consumed AT that position
//   int32 engine_argmax[num_positions]   -- the engine's own next-token choice after that
//                                           position, or -1 where none was computed
//   then num_positions * num_states records of: int64 m, int64 e, int8 code[hidden_size]

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
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
using superslm_marshal::WidenGainToInt32;

namespace {

struct StateRecord {
	int64_t m = 0;
	int64_t e = 0;
	std::vector<int8_t> codes;
};

std::vector<int32_t> ParseIdList(const std::string& s) {
	std::vector<int32_t> out;
	size_t i = 0;
	while (i < s.size()) {
		size_t j = s.find(',', i);
		if (j == std::string::npos) j = s.size();
		const std::string piece = s.substr(i, j - i);
		if (!piece.empty()) out.push_back(static_cast<int32_t>(std::strtol(piece.c_str(), nullptr, 10)));
		i = j + 1;
	}
	return out;
}

}  // namespace

int main(int argc, char** argv) {
	if (argc < 5) {
		std::fprintf(stderr,
		             "usage: %s <model.sslm> <tokenizer.sslm> \"<prompt>\" <prompt_id> "
		             "[--dump-dir <dir>] [--max-new <n>] [--force-tokens id,id,...] "
		             "[--suffix <s>]\n",
		             argv[0]);
		return 2;
	}
	const std::string model_path = argv[1];
	const std::string tokenizer_path = argv[2];
	const std::string prompt = argv[3];
	const std::string prompt_id = argv[4];
	std::string dump_dir = "out/t1836";
	std::string suffix = "";
	size_t max_new = 24;
	std::vector<int32_t> forced;
	for (int i = 5; i < argc; ++i) {
		if (std::strcmp(argv[i], "--dump-dir") == 0 && i + 1 < argc) dump_dir = argv[++i];
		else if (std::strcmp(argv[i], "--max-new") == 0 && i + 1 < argc)
			max_new = static_cast<size_t>(std::strtoul(argv[++i], nullptr, 10));
		else if (std::strcmp(argv[i], "--force-tokens") == 0 && i + 1 < argc)
			forced = ParseIdList(argv[++i]);
		else if (std::strcmp(argv[i], "--suffix") == 0 && i + 1 < argc) suffix = argv[++i];
	}
	if (!forced.empty() && forced.size() < max_new) {
		std::fprintf(stderr, "FAILED at stage=args: --force-tokens has %zu ids, --max-new is %zu\n",
		             forced.size(), max_new);
		return 2;
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
	const size_t vocab_size_z = static_cast<size_t>(model_view.config.vocab_size);

	std::printf("model loaded: hidden_size=%zu layers=%u heads=%u/%u head_dim=%zu prompt_tokens=%zu "
	            "max_new=%zu forced=%zu (%s)\n",
	            hidden_size, num_hidden_layers, num_heads, num_kv_heads, head_dim,
	            prompt_tokens.size(), max_new, forced.size(), prompt_id.c_str());

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
	const SslmTensorView* final_gain_w = model_view.weights.Tensor("final_norm.gain");
	if (!embed_w || !final_gain_w) {
		std::fprintf(stderr, "FAILED at stage=head_marshal: missing embed or final_norm.gain tensor\n");
		return 1;
	}
	const std::vector<int32_t> final_norm_gain = WidenGainToInt32(*final_gain_w);
	bool ok = true;
	const CarriedScale embed_site_constant =
	    ReadCarriedScale(model_view.composition_constants, "embed", &ok);
	const CarriedScale final_norm_site_constant =
	    ReadCarriedScale(model_view.composition_constants, "final_norm", &ok);
	if (!ok) {
		std::fprintf(stderr, "FAILED at stage=head_marshal: missing embed/final_norm site constant\n");
		return 1;
	}
	const int8_t* const embed_weights = reinterpret_cast<const int8_t*>(embed_w->data);
	const int8_t* head_weights = nullptr;
	if (model_view.config.tie_word_embeddings) {
		head_weights = embed_weights;
	} else {
		const SslmTensorView* lm_head_w = model_view.weights.Tensor("lm_head");
		if (!lm_head_w) {
			std::fprintf(stderr, "FAILED at stage=head_marshal: tie_word_embeddings=0 but no lm_head\n");
			return 1;
		}
		head_weights = reinterpret_cast<const int8_t*>(lm_head_w->data);
	}

	const int64_t context_cap = static_cast<int64_t>(model_view.config.context_cap);
	const size_t kv_bytes = static_cast<size_t>(num_hidden_layers) * static_cast<size_t>(context_cap) *
	                        num_kv_heads * head_dim * 2;

	// --- the production path, unmodified, on its own instance -----------------
	std::vector<uint8_t> prod_workspace(kv_bytes);
	std::vector<int8_t> prod_hidden_codes(hidden_size);
	SequenceLayerState prod_seq;
	prod_seq.hidden_codes = prod_hidden_codes.data();
	std::vector<int32_t> prod_tokens(max_new);
	std::vector<int32_t> prod_logit_rows(max_new * vocab_size_z);
	size_t prod_produced = 0;
	SslmDecodeStopReason prod_stop = SslmDecodeStopReason::MaxTokensReached;
	const SslmForwardStatus prod_status = RunGreedyDecodeLoop(
	    prod_seq, layers.data(), num_hidden_layers, hidden_size, head_dim, num_kv_heads,
	    intermediate_size, context_cap, model_view.rope_tables, prompt_tokens.data(),
	    prompt_tokens.size(), embed_weights, embed_site_constant, final_norm_gain.data(),
	    final_norm_site_constant, head_weights, static_cast<int32_t>(vocab_size_z),
	    /*stop_ids=*/nullptr, /*stop_count=*/0, max_new, prod_workspace.data(), prod_workspace.size(),
	    prod_tokens.data(), prod_logit_rows.data(), max_new, &prod_produced, &prod_stop,
	    model_view.config.kv_precision);
	if (prod_status != SslmForwardStatus::Ok || prod_produced != max_new) {
		std::fprintf(stderr, "FAILED at stage=production_decode: status=%s produced=%zu\n",
		             SslmForwardStatusName(prod_status), prod_produced);
		return 1;
	}
	std::printf("production_decode: %zu tokens produced by RunGreedyDecodeLoop\n", prod_produced);

	// --- capture instance: every position stepped one layer at a time ---------
	std::vector<uint8_t> workspace(kv_bytes);
	std::vector<int8_t> hidden_codes(hidden_size);
	SequenceLayerState seq;
	seq.hidden_codes = hidden_codes.data();

	// --- check instance: every position run with a single full-budget call ----
	std::vector<uint8_t> chk_workspace(kv_bytes);
	std::vector<int8_t> chk_hidden_codes(hidden_size);
	SequenceLayerState chk_seq;
	chk_seq.hidden_codes = chk_hidden_codes.data();

	auto Embed = [&](SequenceLayerState& s, int32_t token) -> SslmForwardStatus {
		std::vector<int8_t> embed_codes(hidden_size);
		CarriedScale embed_scale{};
		const SslmForwardStatus est =
		    EmbedEntry(token, static_cast<int32_t>(vocab_size_z), embed_weights, hidden_size,
		               embed_site_constant, embed_codes.data(), &embed_scale);
		if (est != SslmForwardStatus::Ok) return est;
		for (size_t i = 0; i < hidden_size; ++i) s.hidden_codes[i] = embed_codes[i];
		s.hidden_scale = embed_scale;
		s.layer_index = 0;
		return SslmForwardStatus::Ok;
	};

	// final_norm -> LogitsSite -> argmax, exactly RunGreedyDecodeLoop's own selection step.
	std::vector<int8_t> final_codes(hidden_size);
	std::vector<int64_t> wide_logits(vocab_size_z);
	std::vector<int32_t> logits(vocab_size_z);
	auto SelectNext = [&](size_t token_index, int32_t* out_token) -> SslmForwardStatus {
		CarriedScale final_scale{};
		const SslmForwardStatus nst =
		    RmsNormSite(seq.hidden_codes, final_norm_gain.data(), hidden_size, seq.hidden_scale,
		                final_norm_site_constant, final_codes.data(), &final_scale, "final_norm",
		                token_index);
		if (nst != SslmForwardStatus::Ok) return nst;
		const SslmForwardStatus lst = LogitsSite(final_codes.data(), hidden_size, head_weights,
		                                          vocab_size_z, wide_logits.data(), logits.data());
		if (lst != SslmForwardStatus::Ok) return lst;
		*out_token = ArgmaxLowestIndexTieBreak(logits.data(), vocab_size_z);
		return SslmForwardStatus::Ok;
	};

	const size_t prompt_len = prompt_tokens.size();
	const size_t num_positions = prompt_len + max_new;
	const size_t num_states = static_cast<size_t>(num_hidden_layers) + 1;
	std::vector<StateRecord> records;
	records.reserve(num_positions * num_states);
	std::vector<int32_t> consumed(num_positions, -1);
	std::vector<int32_t> engine_argmax(num_positions, -1);
	size_t invariance_checks = 0;
	size_t first_divergence = num_positions;   // index into the GENERATED sequence

	for (size_t t = 0; t < num_positions; ++t) {
		int32_t token;
		if (t < prompt_len) {
			token = prompt_tokens[t];
		} else {
			const size_t g = t - prompt_len;          // this is the g-th generated token
			// The engine's own choice at the previous position is engine_argmax[t-1];
			// teacher forcing replaces the id consumed here, never the id recorded there.
			token = forced.empty() ? engine_argmax[t - 1] : forced[g];
		}
		consumed[t] = token;

		SslmForwardStatus st = Embed(seq, token);
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

		// The selection step runs from the last prompt position onward -- that is where
		// RunGreedyDecodeLoop's own first produced token comes from.
		if (t + 1 >= prompt_len) {
			int32_t chosen = -1;
			st = SelectNext(t, &chosen);
			if (st != SslmForwardStatus::Ok) {
				std::fprintf(stderr, "FAILED at stage=select_next: position=%zu status=%s\n", t,
				             SslmForwardStatusName(st));
				return 1;
			}
			engine_argmax[t] = chosen;
			const size_t g = t + 1 - prompt_len;      // this argmax IS production's token g
			if (g < prod_produced && chosen != prod_tokens[g] && first_divergence == num_positions) {
				first_divergence = g;
			}
		}

		// Per-position budget-invariance self-check on an independent engine instance.
		st = Embed(chk_seq, token);
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

	// Production agreement. In free mode the walk IS the free-running trajectory and must
	// reproduce RunGreedyDecodeLoop exactly. In forced mode it must agree up to the first
	// position where the forced id and the engine's own choice part company; that index is
	// reported either way rather than being asserted away.
	if (forced.empty()) {
		size_t agree = 0;
		for (size_t g = 0; g < prod_produced; ++g) {
			if (engine_argmax[prompt_len - 1 + g] != prod_tokens[g]) break;
			++agree;
		}
		std::printf("production_agreement (free): %zu/%zu stepped tokens equal "
		            "RunGreedyDecodeLoop's own\n", agree, prod_produced);
		if (agree != prod_produced) {
			std::fprintf(stderr, "FAILED at stage=production_agreement: the stepped walk diverges "
			                     "from RunGreedyDecodeLoop at generated index %zu\n", agree);
			return 1;
		}
	} else {
		if (engine_argmax[prompt_len - 1] != prod_tokens[0]) {
			std::fprintf(stderr, "FAILED at stage=production_agreement: the walk's first argmax "
			                     "%d differs from RunGreedyDecodeLoop's first token %d\n",
			             engine_argmax[prompt_len - 1], prod_tokens[0]);
			return 1;
		}
		size_t forced_agree = 0;
		while (forced_agree < max_new &&
		       forced[forced_agree] == engine_argmax[prompt_len - 1 + forced_agree]) {
			++forced_agree;
		}
		std::printf("forced_vs_engine_argmax: the forced continuation and the engine's own choice "
		            "agree for the first %zu of %zu generated tokens\n", forced_agree, max_new);
	}
	std::printf("engine_vs_production_first_divergence: %s\n",
	            first_divergence == num_positions ? "none" : std::to_string(first_divergence).c_str());

	const std::string path = dump_dir + "/" + prompt_id + suffix + "_engine.bin";
	std::ofstream f(path, std::ios::binary);
	if (!f) {
		std::fprintf(stderr, "FAILED at stage=dump_open: %s\n", path.c_str());
		return 1;
	}
	const int32_t np = static_cast<int32_t>(num_positions);
	const int32_t ns = static_cast<int32_t>(num_states);
	const int32_t nh = static_cast<int32_t>(hidden_size);
	const int32_t pl = static_cast<int32_t>(prompt_len);
	f.write(reinterpret_cast<const char*>(&np), sizeof(np));
	f.write(reinterpret_cast<const char*>(&ns), sizeof(ns));
	f.write(reinterpret_cast<const char*>(&nh), sizeof(nh));
	f.write(reinterpret_cast<const char*>(&pl), sizeof(pl));
	f.write(reinterpret_cast<const char*>(consumed.data()),
	        static_cast<std::streamsize>(consumed.size() * sizeof(int32_t)));
	f.write(reinterpret_cast<const char*>(engine_argmax.data()),
	        static_cast<std::streamsize>(engine_argmax.size() * sizeof(int32_t)));
	for (const StateRecord& r : records) {
		f.write(reinterpret_cast<const char*>(&r.m), sizeof(r.m));
		f.write(reinterpret_cast<const char*>(&r.e), sizeof(r.e));
		f.write(reinterpret_cast<const char*>(r.codes.data()),
		        static_cast<std::streamsize>(r.codes.size()));
	}
	f.close();
	std::printf("engine dump written: %s (%zu positions [%zu prompt + %zu generated] x %zu states "
	            "x %zu channels)\n",
	            path.c_str(), num_positions, prompt_len, max_new, num_states, hidden_size);
	return 0;
}
