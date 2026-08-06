// t1740_pooled_trace.cpp -- T-1740 scratch instrument: captures the int8
// engine's per-layer residual stream at EVERY prompt position (not only the
// last), so a mean-pooled embedding (positions 1..n, excluding position 0)
// can be compared against the float reference at the pooling scheme D-SLM447
// ratified.
//
// This is a NEW file, not a modification of the reviewed/shipped
// tools/sslm_layer_trace.cpp (T-1685) -- that tool's own "position 0" naming
// convention (design S3: "the prompt's own LAST token's forward pass") means
// something different from this ticket's "position 0" (the sequence's FIRST
// token, D-SLM447's convention); reusing that tool's own code unmodified was
// not possible because it captures exactly one sequence position (the last)
// per invocation, and this ticket needs every position's per-layer residual
// from a single prompt in one run. The marshaling and per-layer forward
// composition (sslm_marshal.h, EmbedEntry, RunLayerLoop) are reused verbatim
// -- only the capture loop is new.
//
// Usage: t1740_pooled_trace <model.sslm> <tokenizer.sslm> "<prompt>" --dump <path>
//
// "<prompt>" is the full chat-templated prompt text, the same convention
// tools/sslm_generate.cpp and tools/sslm_layer_trace.cpp already use.
//
// Dump format (custom to this scratch tool -- read by
// tools/t1740_pooled_fidelity_report.py):
//   uint64 num_positions (== the prompt's own token count, n)
//   uint64 num_rows      (== num_hidden_layers + 1, embedding + each layer)
//   uint64 hidden_size
//   uint64 prompt_fingerprint (FNV-1a 64-bit hash of the raw prompt argument,
//       UTF-8, before tokenization -- identical formula to
//       tools/sslm_layer_trace.cpp's own Fnv1a64)
//   then, for position 0..num_positions-1, for row 0..num_rows-1:
//       int64 m, int64 e, then hidden_size int8 codes
//   (position-major, row-minor -- i.e. all 29 rows for position 0, then all
//   29 rows for position 1, ...)
//
// Two self-checks, run every invocation, before any dump is written:
//
//   1. Whole-sequence self-check (same shape as sslm_layer_trace.cpp's own):
//      the production RunGreedyDecodeLoop(max_new_tokens=1) call and this
//      tool's own manual replay of the LAST position (built by resuming
//      RunLayerLoop(layer_budget=1) 28 times from a fresh embed of the last
//      prompt token, after prefilling every earlier position the same way)
//      must agree bit-for-bit on the produced token id and the full logit
//      row. This is the same guarantee tools/sslm_layer_trace.cpp's own
//      self-check gives, and it depends on every earlier position's K/V
//      cache entries being correct -- so a wrong non-last position would
//      generally corrupt this check too.
//
//   2. Resumed-vs-one-shot equivalence at an INTERIOR position (not just the
//      last): this tool's own capture loop derives every position's every
//      layer by RESUMING RunLayerLoop(layer_budget=1) one layer at a time
//      (so intermediate layers can be snapshotted) -- a different call
//      pattern from a single RunLayerLoop(layer_budget=num_hidden_layers)
//      shot. The self-check above only ever certifies the LAST position
//      (where sslm_layer_trace.cpp's own design already established this
//      equivalence). This tool additionally runs ONE mid-sequence position
//      (position n/2, or 0 if n==1) through BOTH compositions -- a one-shot
//      full-budget call on an independent SequenceLayerState, and the
//      resumed 28-step walk this tool's capture loop already performs -- and
//      asserts the final-layer codes and carried scale are bit-identical
//      between the two. This is a new, targeted proof (not present in any
//      existing tool) that resuming one layer at a time produces the same
//      result as a single full-budget call at a position other than the
//      last, closing the one correctness gap this tool's capture shape opens
//      that sslm_layer_trace.cpp's own self-check does not cover.
//
// On either self-check's failure this tool exits non-zero with a loud
// diagnostic and writes no dump -- matching this codebase's existing
// fail-loud convention (sslm_generate.cpp's own "FAILED at stage=..." lines).

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
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
	             "usage: %s <model.sslm> <tokenizer.sslm> \"<prompt>\" --dump <path>\n", argv0);
}

// Verbatim of tools/sslm_layer_trace.cpp's own Fnv1a64.
uint64_t Fnv1a64(const std::string& s) {
	uint64_t h = UINT64_C(14695981039346656037);
	for (unsigned char c : s) {
		h ^= static_cast<uint64_t>(c);
		h *= UINT64_C(1099511628211);
	}
	return h;
}

struct RowSnapshot {
	std::vector<int8_t> codes;
	int64_t m;
	int64_t e;
};

}  // namespace

int main(int argc, char** argv) {
	if (argc < 4) {
		PrintUsage(argv[0]);
		return 2;
	}
	const std::string model_path = argv[1];
	const std::string tokenizer_path = argv[2];
	const std::string prompt = argv[3];
	std::string dump_path;
	for (int i = 4; i < argc; ++i) {
		if (std::strcmp(argv[i], "--dump") == 0 && i + 1 < argc) {
			dump_path = argv[++i];
		} else {
			std::fprintf(stderr, "unrecognized argument: %s\n", argv[i]);
			PrintUsage(argv[0]);
			return 2;
		}
	}
	if (dump_path.empty()) {
		std::fprintf(stderr, "FAILED at stage=args: --dump <path> is required\n");
		PrintUsage(argv[0]);
		return 2;
	}

	// --- tokenizer + prompt encode (identical to sslm_layer_trace.cpp). ------
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
	if (prompt_tokens.empty()) {
		std::fprintf(stderr, "FAILED at stage=tokenizer_encode: prompt encoded to zero tokens\n");
		return 1;
	}
	const size_t n_positions = prompt_tokens.size();

	// --- model load + per-layer marshal (identical to sslm_layer_trace.cpp). -
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
	            "vocab=%u context_cap=%u tie=%d prompt_tokens=%zu\n",
	            model_view.config.hidden_size, model_view.config.num_hidden_layers,
	            model_view.config.num_attention_heads, model_view.config.num_key_value_heads,
	            model_view.config.head_dim, model_view.config.intermediate_size,
	            model_view.config.vocab_size, model_view.config.context_cap,
	            model_view.config.tie_word_embeddings ? 1 : 0, n_positions);

	const uint32_t num_heads = model_view.config.num_attention_heads;
	const uint32_t num_kv_heads = model_view.config.num_key_value_heads;
	const uint32_t num_hidden_layers = model_view.config.num_hidden_layers;
	const size_t hidden_size = model_view.config.hidden_size;
	const size_t num_rows = static_cast<size_t>(num_hidden_layers) + 1;

	if (static_cast<int64_t>(n_positions) > static_cast<int64_t>(model_view.config.context_cap)) {
		std::fprintf(stderr,
		             "FAILED at stage=context_check: prompt has %zu tokens, context_cap is %u\n",
		             n_positions, model_view.config.context_cap);
		return 1;
	}

	PreflightScanWscFolds(model_view);

	std::vector<LayerBacking> backings(num_hidden_layers);
	std::vector<LayerWeights> layers(num_hidden_layers);
	for (uint32_t l = 0; l < num_hidden_layers; ++l) {
		std::string marshal_err;
		if (!MarshalLayer(model_view, l, num_heads, num_kv_heads, backings[l], layers[l],
		                   &marshal_err)) {
			std::fprintf(stderr, "FAILED at stage=layer_weights_marshal: layer=%u diagnostic=\"%s\"\n",
			             l, marshal_err.c_str());
			return 1;
		}
	}

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
	const size_t vocab_size_z = static_cast<size_t>(model_view.config.vocab_size);

	// --- Step A: the PRODUCTION path, unmodified (same as sslm_layer_trace.cpp). --
	std::vector<uint8_t> prod_workspace(kv_bytes);
	std::vector<int8_t> prod_hidden_codes(hidden_size);
	SequenceLayerState prod_seq;
	prod_seq.hidden_codes = prod_hidden_codes.data();

	std::vector<int32_t> prod_out_token(1);
	std::vector<int32_t> prod_out_logit_row(vocab_size_z);
	size_t prod_tokens_produced = 0;
	SslmDecodeStopReason prod_stop_reason = SslmDecodeStopReason::MaxTokensReached;

	const SslmForwardStatus prod_status = RunGreedyDecodeLoop(
	    prod_seq, layers.data(), num_hidden_layers, hidden_size, model_view.config.head_dim,
	    num_kv_heads, model_view.config.intermediate_size, context_cap, model_view.rope_tables,
	    prompt_tokens.data(), prompt_tokens.size(), embed_weights, embed_site_constant,
	    final_norm_gain.data(), final_norm_site_constant, head_weights,
	    static_cast<int32_t>(model_view.config.vocab_size), /*stop_ids=*/nullptr, /*stop_count=*/0,
	    /*max_new_tokens=*/1, prod_workspace.data(), prod_workspace.size(), prod_out_token.data(),
	    prod_out_logit_row.data(), prod_out_token.size(), &prod_tokens_produced, &prod_stop_reason,
	    model_view.config.kv_precision);
	if (prod_status != SslmForwardStatus::Ok) {
		std::fprintf(stderr, "FAILED at stage=production_decode: status=%s\n",
		             SslmForwardStatusName(prod_status));
		return 1;
	}
	if (prod_tokens_produced != 1) {
		std::fprintf(stderr,
		             "FAILED at stage=production_decode: expected exactly 1 produced token, got %zu\n",
		             prod_tokens_produced);
		return 1;
	}

	auto EmbedWholeTokenInto = [&](SequenceLayerState& seq, int32_t token) -> SslmForwardStatus {
		std::vector<int8_t> embed_codes(hidden_size);
		CarriedScale embed_scale{};
		const SslmForwardStatus est =
		    EmbedEntry(token, static_cast<int32_t>(model_view.config.vocab_size), embed_weights,
		               hidden_size, embed_site_constant, embed_codes.data(), &embed_scale);
		if (est != SslmForwardStatus::Ok) return est;
		for (size_t i = 0; i < hidden_size; ++i) seq.hidden_codes[i] = embed_codes[i];
		seq.hidden_scale = embed_scale;
		seq.layer_index = 0;
		return SslmForwardStatus::Ok;
	};

	// --- Step B: this tool's own capture -- EVERY position, ALL 29 rows. -----
	// A single continuous SequenceLayerState, resumed one layer at a time for
	// every position (not just the last), snapshotting after each layer.
	std::vector<uint8_t> all_workspace(kv_bytes);
	std::vector<int8_t> all_hidden_codes(hidden_size);
	SequenceLayerState all_seq;
	all_seq.hidden_codes = all_hidden_codes.data();

	// position-major, row-minor storage.
	std::vector<std::vector<RowSnapshot>> all_rows(n_positions);

	// The mid-sequence equivalence check (self-check 2, see file header):
	// captured here so it can be asserted after the capture loop below without
	// re-running the forward.
	const size_t equiv_position = (n_positions >= 2) ? (n_positions / 2) : 0;
	RowSnapshot equiv_final_from_resumed{};

	for (size_t pos = 0; pos < n_positions; ++pos) {
		SslmForwardStatus est = EmbedWholeTokenInto(all_seq, prompt_tokens[pos]);
		if (est != SslmForwardStatus::Ok) {
			std::fprintf(stderr, "FAILED at stage=capture_embed: position=%zu status=%s\n", pos,
			             SslmForwardStatusName(est));
			return 1;
		}
		all_rows[pos].reserve(num_rows);
		all_rows[pos].push_back(RowSnapshot{
		    std::vector<int8_t>(all_seq.hidden_codes, all_seq.hidden_codes + hidden_size),
		    all_seq.hidden_scale.m, all_seq.hidden_scale.e});

		for (uint32_t step = 0; step < num_hidden_layers; ++step) {
			const SslmForwardStatus st =
			    RunLayerLoop(all_seq, layers.data(), num_hidden_layers, /*layer_budget=*/1, hidden_size,
			                 model_view.config.head_dim, num_kv_heads,
			                 model_view.config.intermediate_size, context_cap, model_view.rope_tables,
			                 all_workspace.data(), all_workspace.size());
			if (st != SslmForwardStatus::Ok) {
				std::fprintf(stderr, "FAILED at stage=capture_layer_step: position=%zu layer=%u status=%s\n",
				             pos, step, SslmForwardStatusName(st));
				return 1;
			}
			all_rows[pos].push_back(RowSnapshot{
			    std::vector<int8_t>(all_seq.hidden_codes, all_seq.hidden_codes + hidden_size),
			    all_seq.hidden_scale.m, all_seq.hidden_scale.e});
		}

		if (pos == equiv_position) {
			equiv_final_from_resumed = all_rows[pos].back();
		}
	}

	// --- Self-check 1: the LAST position's resumed-28-step trace must match --
	// the production one-shot decode call bit-for-bit -- same shape as
	// tools/sslm_layer_trace.cpp's own self-check.
	{
		const RowSnapshot& last_row = all_rows[n_positions - 1].back();
		std::vector<int8_t> final_codes(hidden_size);
		CarriedScale final_scale{};
		SslmForwardStatus st =
		    RmsNormSite(last_row.codes.data(), final_norm_gain.data(), hidden_size,
		                CarriedScale{last_row.m, last_row.e}, final_norm_site_constant,
		                final_codes.data(), &final_scale, "final_norm");
		if (st != SslmForwardStatus::Ok) {
			std::fprintf(stderr, "FAILED at stage=self_check_final_norm: status=%s\n",
			             SslmForwardStatusName(st));
			return 1;
		}
		std::vector<int64_t> wide_logits(vocab_size_z);
		std::vector<int32_t> logit_row(vocab_size_z);
		st = LogitsSite(final_codes.data(), hidden_size, head_weights,
		                 static_cast<int32_t>(model_view.config.vocab_size), wide_logits.data(),
		                 logit_row.data());
		if (st != SslmForwardStatus::Ok) {
			std::fprintf(stderr, "FAILED at stage=self_check_logits: status=%s\n",
			             SslmForwardStatusName(st));
			return 1;
		}
		const int32_t trace_token = ArgmaxLowestIndexTieBreak(logit_row.data(), vocab_size_z);

		const bool token_match = (trace_token == prod_out_token[0]);
		const bool logits_match =
		    (std::memcmp(logit_row.data(), prod_out_logit_row.data(), vocab_size_z * sizeof(int32_t)) ==
		     0);
		if (!token_match || !logits_match) {
			std::fprintf(stderr,
			             "FAILED at stage=self_check: production and this tool's all-position trace "
			             "disagree at the last position -- production_token=%d trace_token=%d "
			             "token_match=%d logit_row_bit_identical=%d (no dump written)\n",
			             prod_out_token[0], trace_token, token_match ? 1 : 0, logits_match ? 1 : 0);
			return 1;
		}
		std::printf("self_check_1: production and this tool's all-position trace agree bit-for-bit "
		            "at the last position (token=%d, %zu logits)\n",
		            trace_token, vocab_size_z);
	}

	// --- Self-check 2: resumed-vs-one-shot equivalence at an INTERIOR --------
	// position (see file header). One-shot full-budget replay of the prefix
	// ending at `equiv_position`, on an independent SequenceLayerState.
	{
		std::vector<uint8_t> oneshot_workspace(kv_bytes);
		std::vector<int8_t> oneshot_hidden_codes(hidden_size);
		SequenceLayerState oneshot_seq;
		oneshot_seq.hidden_codes = oneshot_hidden_codes.data();

		for (size_t pos = 0; pos <= equiv_position; ++pos) {
			SslmForwardStatus est = EmbedWholeTokenInto(oneshot_seq, prompt_tokens[pos]);
			if (est != SslmForwardStatus::Ok) {
				std::fprintf(stderr, "FAILED at stage=equiv_embed: position=%zu status=%s\n", pos,
				             SslmForwardStatusName(est));
				return 1;
			}
			const SslmForwardStatus st = RunLayerLoop(
			    oneshot_seq, layers.data(), num_hidden_layers, /*layer_budget=*/num_hidden_layers,
			    hidden_size, model_view.config.head_dim, num_kv_heads,
			    model_view.config.intermediate_size, context_cap, model_view.rope_tables,
			    oneshot_workspace.data(), oneshot_workspace.size());
			if (st != SslmForwardStatus::Ok) {
				std::fprintf(stderr, "FAILED at stage=equiv_layers: position=%zu status=%s\n", pos,
				             SslmForwardStatusName(st));
				return 1;
			}
		}
		const bool codes_match = std::memcmp(oneshot_seq.hidden_codes, equiv_final_from_resumed.codes.data(),
		                                      hidden_size) == 0;
		const bool scale_match = oneshot_seq.hidden_scale.m == equiv_final_from_resumed.m &&
		                          oneshot_seq.hidden_scale.e == equiv_final_from_resumed.e;
		if (!codes_match || !scale_match) {
			std::fprintf(stderr,
			             "FAILED at stage=self_check_2: resumed-28-step and one-shot full-budget "
			             "composition disagree at position=%zu (interior position) -- "
			             "codes_match=%d scale_match=%d (no dump written)\n",
			             equiv_position, codes_match ? 1 : 0, scale_match ? 1 : 0);
			return 1;
		}
		std::printf("self_check_2: resumed-28-step and one-shot full-budget composition agree "
		            "bit-for-bit at interior position=%zu (of %zu)\n",
		            equiv_position, n_positions);
	}

	// --- Dump. -----------------------------------------------------------------
	const uint64_t fingerprint = Fnv1a64(prompt);
	{
		std::ofstream f(dump_path, std::ios::binary | std::ios::trunc);
		if (!f) {
			std::fprintf(stderr, "FAILED at stage=dump_write: could not open \"%s\"\n", dump_path.c_str());
			return 1;
		}
		const uint64_t np = static_cast<uint64_t>(n_positions);
		const uint64_t nr = static_cast<uint64_t>(num_rows);
		const uint64_t hs = static_cast<uint64_t>(hidden_size);
		f.write(reinterpret_cast<const char*>(&np), sizeof(np));
		f.write(reinterpret_cast<const char*>(&nr), sizeof(nr));
		f.write(reinterpret_cast<const char*>(&hs), sizeof(hs));
		f.write(reinterpret_cast<const char*>(&fingerprint), sizeof(fingerprint));
		for (size_t pos = 0; pos < n_positions; ++pos) {
			for (size_t row = 0; row < num_rows; ++row) {
				const RowSnapshot& snap = all_rows[pos][row];
				f.write(reinterpret_cast<const char*>(&snap.m), sizeof(snap.m));
				f.write(reinterpret_cast<const char*>(&snap.e), sizeof(snap.e));
				f.write(reinterpret_cast<const char*>(snap.codes.data()), snap.codes.size());
			}
		}
		if (!f) {
			std::fprintf(stderr, "FAILED at stage=dump_write: write error on \"%s\"\n", dump_path.c_str());
			return 1;
		}
	}
	std::printf("pooled_trace_dumped: %zu positions x %zu rows x %zu hidden_size, "
	            "prompt_fingerprint=0x%016llX -> %s\n",
	            n_positions, num_rows, hidden_size, static_cast<unsigned long long>(fingerprint),
	            dump_path.c_str());
	return 0;
}
