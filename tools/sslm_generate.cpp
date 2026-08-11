// sslm_generate.cpp -- the real-model decode driver (T-1664, WSC1 marshaling
// completed T-1666-driver). Loads a `.sslm` model artifact and a `.sslm`
// tokenizer artifact, encodes a prompt, marshals the artifact's raw
// manifests into a `LayerWeights[]` for all 28 layers, and runs
// `RunGreedyDecodeLoop` to produce output token ids.
//
// THIS BUILD'S STATUS, STATED PLAINLY. T-1657/T-1654/T-1655/T-1656 closed the
// four blockers a prior pass (T-1652/T-1653, branch claude/smoke-driver)
// found: the BIA1 load-time gate is gone (SslmModel::Load accepts the real
// artifact with NO relaxation, on `main`'s own gate), RunLayerLoop is
// GQA-capable, LayerWeights carries a bias field with a live caller, and the
// per-query i-exp constants are derived from artifact composition constants
// rather than fixed per-layer.
//
// THE FIELD T-1664 FOUND NOT REPRESENTABLE IS NOW REPRESENTABLE. T-1664
// discovered and verified against the real artifact's own bytes that the
// real Qwen2.5-1.5B-Instruct artifact's WSC1 weight-scale-fold data is
// genuinely PER-OUTPUT-CHANNEL for all seven projections -- q_proj/o_proj/
// down_proj carry 1536 distinct (identity,mult,shift) triples, k_proj/v_proj
// carry 256, gate_proj/up_proj carry 8960 -- while `LayerWeights` (at that
// time) carried only a single scalar triple shared across a whole
// projection. T-1666 (design doc
// Claude/Vitruvius/superslm-t1666-wsc1-per-channel-fold-design-2026-08-02.md)
// closed that gap in production: `LayerWeights` now carries seven per-
// projection `{proj}_fold_identity`/`{proj}_fold_mult`/`{proj}_fold_shift`
// arrays (forward_sites.h), one triple per output channel, and
// `ProjectAndFunnel`/the k/v-landing fold loop index them by output channel
// (forward_sites.cpp). This driver's job -- marshaling the artifact's real
// WSC1 rows into those arrays, for every layer and all seven projections --
// is what this file now does. Every array is sized to the exact output-
// channel count `ProjectAndFunnel`/the k/v-landing loop read it at: hidden_size
// for q/o/down, `num_key_value_heads * head_dim` for k/v, intermediate_size
// for gate/up (§8.2 "Weight-scale fold blob", docs/sslm_format.md: each WSC1
// tensor is a row-major `[num_channels, 3]` int32 array, row i's triple at
// elements `[3i, 3i+3)` -- identity, mult, shift in that column order).
//
// Usage: sslm_generate <model.sslm> <tokenizer.sslm> "<prompt>" [--max-new N]

#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "superslm/artifact.h"
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
using superslm_marshal::WidenGainToInt32;

namespace {

void PrintUsage(const char* argv0) {
	std::fprintf(stderr,
	             "usage: %s <model.sslm> <tokenizer.sslm> \"<prompt>\" [--max-new N] [--stop "
	             "a,b,c] [--dump-logits <path>]\n",
	             argv0);
}

// T-1681: writes `out_logit_rows` (RunGreedyDecodeLoop's own raw int32 logit
// output, one row of `vocab_size` elements per produced token, populated
// immediately before that token's argmax -- see forward_sites.h's
// RunGreedyDecodeLoop comment) to a flat binary file for offline analysis.
// This is a read-only dump of an array the driver already computes and
// already owns (`out_logit_rows` was already sized and passed into
// RunGreedyDecodeLoop before this flag existed) -- no forward-path
// computation changes. Format: `rows_produced` (uint64 LE) then
// `vocab_size` (uint64 LE) then `rows_produced * vocab_size` little-endian
// int32 values, row-major (row i = the logit row that produced out_tokens[i]).
bool DumpLogitRows(const char* path, const int32_t* rows, size_t rows_produced,
                    size_t vocab_size) {
	std::ofstream f(path, std::ios::binary | std::ios::trunc);
	if (!f) return false;
	const uint64_t rp = static_cast<uint64_t>(rows_produced);
	const uint64_t vs = static_cast<uint64_t>(vocab_size);
	f.write(reinterpret_cast<const char*>(&rp), sizeof(rp));
	f.write(reinterpret_cast<const char*>(&vs), sizeof(vs));
	f.write(reinterpret_cast<const char*>(rows), sizeof(int32_t) * rows_produced * vocab_size);
	return static_cast<bool>(f);
}

}  // namespace

int main(int argc, char** argv) {
	if (argc < 4) {
		PrintUsage(argv[0]);
		return 2;
	}
	const std::string model_path = argv[1];
	const std::string tokenizer_path = argv[2];
	const std::string prompt = argv[3];
	size_t max_new_tokens = 32;
	// The .sslm format carries no end-of-sequence id and the tokenizer view exposes none
	// (tokenizer.h: "No BOS/EOS/chat markers are added -- that is the caller's job"), so the
	// stop set is supplied here rather than defaulted. Qwen2.5-instruct uses 151645
	// (<|im_end|>) and 151643 (<|endoftext|>); a different model family uses different ids.
	std::vector<int32_t> stop_ids;
	std::string dump_logits_path;
	for (int i = 4; i < argc; ++i) {
		if (std::strcmp(argv[i], "--dump-logits") == 0 && i + 1 < argc) {
			dump_logits_path = argv[++i];
		} else if (std::strcmp(argv[i], "--max-new") == 0 && i + 1 < argc) {
			const std::string val = argv[++i];
			try {
				max_new_tokens = static_cast<size_t>(std::stoul(val));
			} catch (const std::exception&) {
				std::fprintf(stderr, "invalid --max-new value: \"%s\" (expected an unsigned integer)\n",
				             val.c_str());
				PrintUsage(argv[0]);
				return 2;
			}
		} else if (std::strcmp(argv[i], "--stop") == 0 && i + 1 < argc) {
			const std::string spec = argv[++i];
			size_t pos = 0;
			try {
				while (pos < spec.size()) {
					const size_t comma = spec.find(',', pos);
					const std::string one = spec.substr(pos, comma == std::string::npos ? std::string::npos : comma - pos);
					if (!one.empty()) stop_ids.push_back(static_cast<int32_t>(std::stol(one)));
					if (comma == std::string::npos) break;
					pos = comma + 1;
				}
			} catch (const std::exception&) {
				std::fprintf(stderr, "invalid --stop value: \"%s\" (expected comma-separated integers)\n",
				             spec.c_str());
				PrintUsage(argv[0]);
				return 2;
			}
		} else {
			std::fprintf(stderr, "unrecognized argument: %s\n", argv[i]);
			PrintUsage(argv[0]);
			return 2;
		}
	}

	const auto t_start = std::chrono::steady_clock::now();

	// --- Stage 1: load the tokenizer artifact and encode the prompt. -------
	std::vector<uint8_t> tok_bytes;
	if (!ReadFile(tokenizer_path.c_str(), tok_bytes)) {
		std::fprintf(stderr, "FAILED at stage=tokenizer_file_read: could not read \"%s\"\n",
		             tokenizer_path.c_str());
		return 1;
	}
	SslmArtifact tok_artifact;
	SslmError tok_open_err;
	if (SslmArtifact::OpenFromMemory(tok_bytes.data(), tok_bytes.size(), tok_artifact,
	                                  &tok_open_err) != SslmStatus::Ok) {
		std::fprintf(stderr, "FAILED at stage=tokenizer_artifact_open: status=%s diagnostic=\"%s\"\n",
		             SslmStatusName(tok_open_err.code), tok_open_err.message.c_str());
		return 1;
	}
	TokenizerView tokenizer;
	std::string tok_err;
	if (!TokenizerView::Open(tok_artifact, tokenizer, &tok_err)) {
		std::fprintf(stderr, "FAILED at stage=tokenizer_view_open: diagnostic=\"%s\"\n", tok_err.c_str());
		return 1;
	}
	const std::vector<int32_t> prompt_tokens = tokenizer.Encode(prompt);

	std::printf("prompt: %s\n", prompt.c_str());
	std::printf("prompt_tokens (%zu):", prompt_tokens.size());
	for (int32_t t : prompt_tokens) std::printf(" %d", t);
	std::printf("\n");
	if (prompt_tokens.empty()) {
		std::fprintf(stderr, "FAILED at stage=tokenizer_encode: prompt encoded to zero tokens\n");
		return 1;
	}

	// --- Stage 2: load the model artifact through the real production ------
	// entry point (SslmModel::Load). T-1657 removed the BIA1 load-time gate
	// this driver's predecessor needed a branch-only relaxation for; this
	// call needs no relaxation of any kind.
	std::vector<uint8_t> model_bytes;
	if (!ReadFile(model_path.c_str(), model_bytes)) {
		std::fprintf(stderr, "FAILED at stage=model_file_read: could not read \"%s\"\n",
		             model_path.c_str());
		return 1;
	}
	SslmModelView model_view;
	std::string model_err;
	const SslmModelStatus load_status =
	    SslmModel::Load(model_bytes.data(), model_bytes.size(), model_view, &model_err);
	if (load_status != SslmModelStatus::Ok) {
		std::fprintf(stderr, "FAILED at stage=model_load: status=%s diagnostic=\"%s\"\n",
		             SslmModelStatusName(load_status), model_err.c_str());
		return 1;
	}
	std::printf("model loaded: hidden_size=%u layers=%u heads=%u/%u head_dim=%u intermediate=%u "
	            "vocab=%u context_cap=%u tie=%d\n",
	            model_view.config.hidden_size, model_view.config.num_hidden_layers,
	            model_view.config.num_attention_heads, model_view.config.num_key_value_heads,
	            model_view.config.head_dim, model_view.config.intermediate_size,
	            model_view.config.vocab_size, model_view.config.context_cap,
	            model_view.config.tie_word_embeddings ? 1 : 0);

	const uint32_t num_heads = model_view.config.num_attention_heads;
	const uint32_t num_kv_heads = model_view.config.num_key_value_heads;
	const uint32_t num_hidden_layers = model_view.config.num_hidden_layers;
	const size_t hidden_size = model_view.config.hidden_size;

	// --- Stage 3: the marshaling adapter. Full artifact-wide scan first ----
	// (this file's header comment; §3.1-style extent confirmation), then a
	// real attempt at every layer, in order -- stopping at the first field
	// this tree's LayerWeights cannot represent (missing/wrong-shaped
	// tensors remain possible; the per-channel WSC1 fold gap T-1664 found is
	// closed as of T-1666).
	PreflightScanWscFolds(model_view);

	std::vector<LayerBacking> backings(num_hidden_layers);
	std::vector<LayerWeights> layers(num_hidden_layers);
	for (uint32_t l = 0; l < num_hidden_layers; ++l) {
		std::string marshal_err;
		if (!MarshalLayer(model_view, l, num_heads, num_kv_heads, backings[l], layers[l],
		                   &marshal_err)) {
			std::fprintf(stderr, "FAILED at stage=layer_weights_marshal: layer=%u diagnostic=\"%s\"\n",
			             l, marshal_err.c_str());
			std::fprintf(stderr,
			             "The tokenizer and model both load and are real. The LayerWeights[] "
			             "marshaling adapter cannot represent this layer's data in the current struct "
			             "shape -- see the diagnostic above for the exact field. No forward pass was "
			             "attempted; RunGreedyDecodeLoop was never called.\n");
			return 1;
		}
	}

	// --- embed/final_norm/head marshaling, workspace sizing, and the -------
	// RunGreedyDecodeLoop call.
	const SslmTensorView* embed_w = model_view.weights.Tensor("embed");
	const SslmTensorView* final_gain_w = model_view.weights.Tensor("final_norm.gain");
	if (!embed_w || !final_gain_w) {
		std::fprintf(stderr, "FAILED at stage=head_marshal: missing embed or final_norm.gain tensor\n");
		return 1;
	}
	std::vector<int32_t> final_norm_gain = WidenGainToInt32(*final_gain_w);
	bool ok = true;
	CarriedScale embed_site_constant =
	    ReadCarriedScale(model_view.composition_constants, "embed", &ok);
	CarriedScale final_norm_site_constant =
	    ReadCarriedScale(model_view.composition_constants, "final_norm", &ok);
	if (!ok) {
		std::fprintf(stderr, "FAILED at stage=head_marshal: missing embed/final_norm site constant\n");
		return 1;
	}
	const int8_t* embed_weights = reinterpret_cast<const int8_t*>(embed_w->data);
	// tie_word_embeddings is the CALLER's resolution (forward_sites.h:828,
	// "weights[tie ? \"embed\" : \"lm_head\"]"): when the artifact ties the
	// head to the embedding matrix, the head reuses embed_weights; when it
	// does not, a dedicated "lm_head" WGT1 tensor must be present. This
	// artifact happens to tie (tie=1, printed above), but that is a fact
	// about THIS artifact, not a fact this driver may assume -- CFG1's
	// tie_word_embeddings is a legal 0 (accepted by SslmModel::Load,
	// src/model.cpp:483; already produced by
	// tools/sslm_pinned_calibration_fixture.py and
	// tools/test_sslm_convert_loader_join.py). An unconditional embed
	// fallback would compute logits against the wrong matrix on a tie=0
	// artifact and exit 0 -- the only silent-wrong-answer path in a file
	// whose every other stage fails loudly.
	const int8_t* head_weights = nullptr;
	if (model_view.config.tie_word_embeddings) {
		head_weights = embed_weights;
	} else {
		const SslmTensorView* lm_head_w = model_view.weights.Tensor("lm_head");
		if (!lm_head_w) {
			std::fprintf(stderr,
			             "FAILED at stage=head_marshal: tie_word_embeddings=0 but no \"lm_head\" WGT1 "
			             "tensor is present -- cannot resolve the head weight matrix\n");
			return 1;
		}
		head_weights = reinterpret_cast<const int8_t*>(lm_head_w->data);
	}

	const int64_t context_cap = static_cast<int64_t>(model_view.config.context_cap);
	const size_t kv_bytes = static_cast<size_t>(num_hidden_layers) * static_cast<size_t>(context_cap) *
	                        num_kv_heads * model_view.config.head_dim * 2;
	std::vector<uint8_t> workspace(kv_bytes);
	std::vector<int8_t> hidden_codes(hidden_size);
	SequenceLayerState seq;
	seq.hidden_codes = hidden_codes.data();

	std::vector<int32_t> out_tokens(max_new_tokens);
	std::vector<int32_t> out_logit_rows(max_new_tokens * static_cast<size_t>(model_view.config.vocab_size));
	size_t out_tokens_produced = 0;
	SslmDecodeStopReason stop_reason = SslmDecodeStopReason::MaxTokensReached;

	// T-1894 (design Sec31.2.1, round 4/D-SLM2423, link 4 of 5): passes
	// `model_view.option_g_fused_k_landing` -- the header `flags` bit
	// SslmModel::Load already read into this view (link 1/2) -- as
	// RunGreedyDecodeLoop's own new trailing parameter, mirroring exactly how
	// this call already passes `model_view.config.kv_precision` as its
	// previous last argument.
	const SslmForwardStatus decode_status = RunGreedyDecodeLoop(
	    seq, layers.data(), num_hidden_layers, hidden_size, model_view.config.head_dim, num_kv_heads,
	    model_view.config.intermediate_size, context_cap, model_view.rope_tables, prompt_tokens.data(),
	    prompt_tokens.size(), embed_weights, embed_site_constant, final_norm_gain.data(),
	    final_norm_site_constant, head_weights, static_cast<int32_t>(model_view.config.vocab_size),
	    stop_ids.empty() ? nullptr : stop_ids.data(), stop_ids.size(), max_new_tokens, workspace.data(), workspace.size(),
	    out_tokens.data(), out_logit_rows.data(), out_tokens.size(), &out_tokens_produced, &stop_reason,
	    model_view.config.kv_precision, model_view.option_g_fused_k_landing);

	if (decode_status != SslmForwardStatus::Ok) {
		std::fprintf(stderr, "FAILED at stage=decode: status=%s\n",
		             SslmForwardStatusName(decode_status));
		return 1;
	}

	std::printf("output_tokens (%zu):", out_tokens_produced);
	for (size_t i = 0; i < out_tokens_produced; ++i) std::printf(" %d", out_tokens[i]);
	std::printf("\n");
	std::printf("stop_reason: %d\n", static_cast<int>(stop_reason));

	if (!dump_logits_path.empty()) {
		if (!DumpLogitRows(dump_logits_path.c_str(), out_logit_rows.data(), out_tokens_produced,
		                    static_cast<size_t>(model_view.config.vocab_size))) {
			std::fprintf(stderr, "FAILED at stage=dump_logits: could not write \"%s\"\n",
			             dump_logits_path.c_str());
			return 1;
		}
		std::printf("logit_rows_dumped: %zu rows x %u vocab -> %s\n", out_tokens_produced,
		            model_view.config.vocab_size, dump_logits_path.c_str());
	}

	const auto t_end = std::chrono::steady_clock::now();
	std::printf("wall_time_seconds: %.3f\n", std::chrono::duration<double>(t_end - t_start).count());
	return 0;
}
