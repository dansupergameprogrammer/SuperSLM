// sslm_layer_trace.cpp -- T-1685, the int8 side of the T-1683 layer-bisection
// instrument (design Claude/Vitruvius/superslm-t1683-layer-bisection-design-
// 2026-08-02.md S4.1). A self-checking tool, not a second, unverified code
// path: it runs the PRODUCTION decode call (RunGreedyDecodeLoop) and, in the
// SAME process, a manual per-layer replay built only from public API calls
// already proven to compose this way in RunGreedyDecodeLoop's own body
// (forward_sites.cpp) -- then asserts the two agree, bit-for-bit, before
// writing anything.
//
// Usage: sslm_layer_trace <model.sslm> <tokenizer.sslm> "<prompt>" --dump <path>
//
// "<prompt>" is the FULL chat-templated prompt text (system + user +
// assistant-open), the same shape tools/sslm_generate.cpp's argv[3] already
// takes -- this tool computes no chat template of its own. Position 0 only
// (design S3): the prompt's own last token's forward pass, no force-feeding.
//
// Dump format (design S4.1 step 5): uint64 rows (29), uint64 hidden_size,
// uint64 prompt_fingerprint (FNV-1a 64-bit hash of the raw prompt argument,
// UTF-8, before tokenization), uint64 capture_mode (always 1 on this side --
// the int8 engine's own composition is inherently one-token-at-a-time), then
// per row: int64 m, int64 e, then hidden_size int8 codes.
//
// On a self-check mismatch (production vs. manual-replay token id or logit
// row), this tool exits non-zero with a loud diagnostic and writes no dump --
// matching this codebase's existing fail-loud convention
// (sslm_generate.cpp's own "FAILED at stage=..." lines).

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

// FNV-1a, 64-bit (design S4.1 step 5): the offset basis and prime are the
// standard FNV-1a constants. Computed over the prompt argument's raw UTF-8
// bytes, before tokenization -- independent of either side's tokenizer, so a
// tokenizer disagreement cannot itself produce a false-positive pairing
// match at the comparison tool's provenance check (design S5).
uint64_t Fnv1a64(const std::string& s) {
	uint64_t h = UINT64_C(14695981039346656037);
	for (unsigned char c : s) {
		h ^= static_cast<uint64_t>(c);
		h *= UINT64_C(1099511628211);
	}
	return h;
}

struct LayerSnapshot {
	std::vector<int8_t> codes;
	int64_t m;
	int64_t e;
};

bool WriteDump(const char* path, const std::vector<LayerSnapshot>& rows, uint64_t hidden_size,
               uint64_t prompt_fingerprint) {
	std::ofstream f(path, std::ios::binary | std::ios::trunc);
	if (!f) return false;
	const uint64_t nrows = static_cast<uint64_t>(rows.size());
	const uint64_t capture_mode = 1;  // design S4.1 step 5: always 1 on the int8 side.
	f.write(reinterpret_cast<const char*>(&nrows), sizeof(nrows));
	f.write(reinterpret_cast<const char*>(&hidden_size), sizeof(hidden_size));
	f.write(reinterpret_cast<const char*>(&prompt_fingerprint), sizeof(prompt_fingerprint));
	f.write(reinterpret_cast<const char*>(&capture_mode), sizeof(capture_mode));
	for (const auto& row : rows) {
		f.write(reinterpret_cast<const char*>(&row.m), sizeof(row.m));
		f.write(reinterpret_cast<const char*>(&row.e), sizeof(row.e));
		f.write(reinterpret_cast<const char*>(row.codes.data()), row.codes.size());
	}
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

	// --- Stage 1: tokenizer + prompt encode (identical to sslm_generate.cpp). --
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

	// --- Stage 2: model load + per-layer marshal (identical to sslm_generate.cpp). --
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

	// T-1900 fix round 3 (T-1901 New 1): both legs below must move TOGETHER
	// with the loaded artifact's own Option-G selection, or this tool's own
	// bit-for-bit self-check (Step 4, below) proves nothing about which
	// K-landing mode either leg actually ran in -- exactly the defect the
	// confirmation pass found (this file's production leg and its four
	// manual-replay RunLayerLoop calls all silently ran legacy on a flags=1
	// artifact while sslm_generate ran fused, because both legs moved
	// together on the SAME stale default). `option_g_mode` is derived ONCE,
	// from the same `SslmModelView::option_g_fused_k_landing` link 1/2
	// already reads off the artifact header, and threaded explicitly into
	// the production RunGreedyDecodeLoop call and every RunLayerLoop replay
	// call below -- never left to a default (forward_sites.h no longer
	// offers one for RunGreedyDecodeLoop's own selector, precisely so this
	// omission is a compile error rather than a second silent-agreement
	// instrument).
	const bool option_g_fused_k_landing = model_view.option_g_fused_k_landing;
	const OptionGKLandingMode option_g_mode =
	    option_g_fused_k_landing ? OptionGKLandingMode::kFused : OptionGKLandingMode::kLegacy;

	// --- Step 2 (design S4.1 step 2): the PRODUCTION path, unmodified. ------
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
	    model_view.config.kv_precision, option_g_fused_k_landing);
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

	// --- Step 3 (design S4.1 step 3): the manual per-layer replay, on a ------
	// SECOND SequenceLayerState, built only from public EmbedEntry/RunLayerLoop
	// calls -- the same composition RunGreedyDecodeLoop's own RunWholeToken
	// step already uses (forward_sites.cpp).
	std::vector<uint8_t> trace_workspace(kv_bytes);
	std::vector<int8_t> trace_hidden_codes(hidden_size);
	SequenceLayerState trace_seq;
	trace_seq.hidden_codes = trace_hidden_codes.data();

	std::vector<LayerSnapshot> rows;
	rows.reserve(num_hidden_layers + 1);

	auto EmbedWholeToken = [&](int32_t token) -> SslmForwardStatus {
		std::vector<int8_t> embed_codes(hidden_size);
		CarriedScale embed_scale{};
		const SslmForwardStatus est =
		    EmbedEntry(token, static_cast<int32_t>(model_view.config.vocab_size), embed_weights,
		               hidden_size, embed_site_constant, embed_codes.data(), &embed_scale);
		if (est != SslmForwardStatus::Ok) return est;
		for (size_t i = 0; i < hidden_size; ++i) trace_seq.hidden_codes[i] = embed_codes[i];
		trace_seq.hidden_scale = embed_scale;
		trace_seq.layer_index = 0;
		return SslmForwardStatus::Ok;
	};

	// Prefill: every prompt token except the last -- embed + full-budget
	// RunLayerLoop, exactly RunWholeToken's own composition.
	for (size_t i = 0; i + 1 < prompt_tokens.size(); ++i) {
		SslmForwardStatus st = EmbedWholeToken(prompt_tokens[i]);
		if (st != SslmForwardStatus::Ok) {
			std::fprintf(stderr, "FAILED at stage=trace_prefill_embed: position=%zu status=%s\n", i,
			             SslmForwardStatusName(st));
			return 1;
		}
		st = RunLayerLoop(trace_seq, layers.data(), num_hidden_layers,
		                   /*layer_budget=*/num_hidden_layers, hidden_size, model_view.config.head_dim,
		                   num_kv_heads, model_view.config.intermediate_size, context_cap,
		                   model_view.rope_tables, trace_workspace.data(), trace_workspace.size(),
		                   option_g_mode);
		if (st != SslmForwardStatus::Ok) {
			std::fprintf(stderr, "FAILED at stage=trace_prefill_layers: position=%zu status=%s\n", i,
			             SslmForwardStatusName(st));
			return 1;
		}
	}

	// The last prompt token: embed, snapshot "layer 0" (the embedding row),
	// then RunLayerLoop(layer_budget=1) 28 times, snapshotting after each.
	{
		const SslmForwardStatus st = EmbedWholeToken(prompt_tokens.back());
		if (st != SslmForwardStatus::Ok) {
			std::fprintf(stderr, "FAILED at stage=trace_last_token_embed: status=%s\n",
			             SslmForwardStatusName(st));
			return 1;
		}
	}
	rows.push_back(LayerSnapshot{
	    std::vector<int8_t>(trace_seq.hidden_codes, trace_seq.hidden_codes + hidden_size),
	    trace_seq.hidden_scale.m, trace_seq.hidden_scale.e});

	for (uint32_t step = 0; step < num_hidden_layers; ++step) {
		const SslmForwardStatus st =
		    RunLayerLoop(trace_seq, layers.data(), num_hidden_layers, /*layer_budget=*/1, hidden_size,
		                 model_view.config.head_dim, num_kv_heads, model_view.config.intermediate_size,
		                 context_cap, model_view.rope_tables, trace_workspace.data(),
		                 trace_workspace.size(), option_g_mode);
		if (st != SslmForwardStatus::Ok) {
			std::fprintf(stderr, "FAILED at stage=trace_layer_step: layer=%u status=%s\n", step,
			             SslmForwardStatusName(st));
			return 1;
		}
		rows.push_back(LayerSnapshot{
		    std::vector<int8_t>(trace_seq.hidden_codes, trace_seq.hidden_codes + hidden_size),
		    trace_seq.hidden_scale.m, trace_seq.hidden_scale.e});
	}

	// --- Step 3.5 (N1 remedy, D-SLM705; design S4.2 step 6's own shape --------
	// moved to where the int8 rows are actually produced): an independent
	// interior-row oracle, run every invocation, before any dump is written.
	// The trace walk above snapshots `rows` by pushing into a vector as it
	// goes -- exactly the shape a mislabeling defect (a duplicate or dropped
	// push) corrupts silently, because every check downstream of it only
	// ever reads the same vector, indexed the same wrong way, the mislabeling
	// produced. This oracle never reads `rows`: for each row i it recomputes
	// the value from scratch, on a genuinely separate SequenceLayerState and
	// a genuinely separate K/V workspace, and compares by POSITION i. A
	// mislabeled or miscounted entry in `rows` is caught here even though it
	// was invisible to the self-check above (which only ever compares the
	// LAST row against production) and to the resumed-walk fixture in
	// tests/test_main.cpp (which never runs this translation unit at all).
	//
	// The oracle's own prefill is redone ONCE, into its own workspace,
	// completely independent of `trace_workspace` above -- same composition
	// (EmbedEntry + full-budget RunLayerLoop per prefix token), separate
	// buffers, so a wiring bug in the trace walk's own state cannot also be
	// present here. Every row 1..num_hidden_layers then costs exactly one
	// straight-through RunLayerLoop(layer_budget=i) call from a freshly
	// embedded copy of the last prompt token -- never the resumed budget=1
	// walk the trace above uses -- summing to num_hidden_layers *
	// (num_hidden_layers + 1) / 2 single-layer-equivalent steps on the last
	// token, on top of the one-time prefill redo.
	std::vector<uint8_t> oracle_workspace(kv_bytes);
	std::vector<int8_t> oracle_prefill_codes(hidden_size);
	SequenceLayerState oracle_prefill_seq;
	oracle_prefill_seq.hidden_codes = oracle_prefill_codes.data();

	for (size_t i = 0; i + 1 < prompt_tokens.size(); ++i) {
		std::vector<int8_t> embed_codes(hidden_size);
		CarriedScale embed_scale{};
		const SslmForwardStatus est =
		    EmbedEntry(prompt_tokens[i], static_cast<int32_t>(model_view.config.vocab_size),
		               embed_weights, hidden_size, embed_site_constant, embed_codes.data(),
		               &embed_scale);
		if (est != SslmForwardStatus::Ok) {
			std::fprintf(stderr, "FAILED at stage=oracle_prefill_embed: position=%zu status=%s\n", i,
			             SslmForwardStatusName(est));
			return 1;
		}
		for (size_t k = 0; k < hidden_size; ++k) oracle_prefill_seq.hidden_codes[k] = embed_codes[k];
		oracle_prefill_seq.hidden_scale = embed_scale;
		oracle_prefill_seq.layer_index = 0;
		const SslmForwardStatus lst = RunLayerLoop(
		    oracle_prefill_seq, layers.data(), num_hidden_layers,
		    /*layer_budget=*/num_hidden_layers, hidden_size, model_view.config.head_dim, num_kv_heads,
		    model_view.config.intermediate_size, context_cap, model_view.rope_tables,
		    oracle_workspace.data(), oracle_workspace.size(), option_g_mode);
		if (lst != SslmForwardStatus::Ok) {
			std::fprintf(stderr, "FAILED at stage=oracle_prefill_layers: position=%zu status=%s\n", i,
			             SslmForwardStatusName(lst));
			return 1;
		}
	}
	const int64_t oracle_context_length = oracle_prefill_seq.context_length;

	// Row 0 (the embedding row): re-embed the last prompt token independently
	// -- a separate EmbedEntry call, same inputs, no shared buffer with the
	// trace walk's own embed -- and compare before any layer is touched.
	{
		std::vector<int8_t> oracle_embed_codes(hidden_size);
		CarriedScale oracle_embed_scale{};
		const SslmForwardStatus est =
		    EmbedEntry(prompt_tokens.back(), static_cast<int32_t>(model_view.config.vocab_size),
		               embed_weights, hidden_size, embed_site_constant, oracle_embed_codes.data(),
		               &oracle_embed_scale);
		if (est != SslmForwardStatus::Ok) {
			std::fprintf(stderr, "FAILED at stage=oracle_row_embed: row=0 status=%s\n",
			             SslmForwardStatusName(est));
			return 1;
		}
		const bool codes_match =
		    std::memcmp(oracle_embed_codes.data(), rows[0].codes.data(), hidden_size) == 0;
		const bool scale_match = oracle_embed_scale.m == rows[0].m && oracle_embed_scale.e == rows[0].e;
		if (!codes_match || !scale_match) {
			std::fprintf(stderr,
			             "FAILED at stage=interior_row_oracle: row=0 (embedding) captured value does "
			             "not match the independent oracle -- codes_match=%d scale_match=%d (no dump "
			             "written)\n",
			             codes_match ? 1 : 0, scale_match ? 1 : 0);
			return 1;
		}
	}

	for (uint32_t i = 1; i <= num_hidden_layers; ++i) {
		std::vector<int8_t> oracle_row_codes(hidden_size);
		CarriedScale oracle_row_scale{};
		const SslmForwardStatus est =
		    EmbedEntry(prompt_tokens.back(), static_cast<int32_t>(model_view.config.vocab_size),
		               embed_weights, hidden_size, embed_site_constant, oracle_row_codes.data(),
		               &oracle_row_scale);
		if (est != SslmForwardStatus::Ok) {
			std::fprintf(stderr, "FAILED at stage=oracle_row_embed: row=%u status=%s\n", i,
			             SslmForwardStatusName(est));
			return 1;
		}
		SequenceLayerState oracle_row_seq;
		oracle_row_seq.hidden_codes = oracle_row_codes.data();
		oracle_row_seq.hidden_scale = oracle_row_scale;
		oracle_row_seq.layer_index = 0;
		oracle_row_seq.context_length = oracle_context_length;

		const SslmForwardStatus st = RunLayerLoop(
		    oracle_row_seq, layers.data(), num_hidden_layers, /*layer_budget=*/i, hidden_size,
		    model_view.config.head_dim, num_kv_heads, model_view.config.intermediate_size, context_cap,
		    model_view.rope_tables, oracle_workspace.data(), oracle_workspace.size(), option_g_mode);
		if (st != SslmForwardStatus::Ok) {
			std::fprintf(stderr, "FAILED at stage=oracle_row_layers: row=%u status=%s\n", i,
			             SslmForwardStatusName(st));
			return 1;
		}

		const bool codes_match =
		    std::memcmp(oracle_row_codes.data(), rows[i].codes.data(), hidden_size) == 0;
		const bool scale_match =
		    oracle_row_seq.hidden_scale.m == rows[i].m && oracle_row_seq.hidden_scale.e == rows[i].e;
		if (!codes_match || !scale_match) {
			std::fprintf(stderr,
			             "FAILED at stage=interior_row_oracle: row=%u captured value does not match "
			             "the independent straight-through budget=%u oracle -- codes_match=%d "
			             "scale_match=%d (no dump written)\n",
			             i, i, codes_match ? 1 : 0, scale_match ? 1 : 0);
			return 1;
		}
	}
	std::printf(
	    "interior_row_oracle: all %u rows (embedding + %u layers) verified against an independent "
	    "straight-through oracle (design S4.2 step 6 shape)\n",
	    num_hidden_layers + 1, num_hidden_layers);

	// final_norm -> logits -> argmax, from the trace's own captured state.
	std::vector<int8_t> trace_final_codes(hidden_size);
	CarriedScale trace_final_scale{};
	SslmForwardStatus st =
	    RmsNormSite(trace_seq.hidden_codes, final_norm_gain.data(), hidden_size, trace_seq.hidden_scale,
	                final_norm_site_constant, trace_final_codes.data(), &trace_final_scale, "final_norm");
	if (st != SslmForwardStatus::Ok) {
		std::fprintf(stderr, "FAILED at stage=trace_final_norm: status=%s\n", SslmForwardStatusName(st));
		return 1;
	}
	std::vector<int64_t> trace_wide_logits(vocab_size_z);
	std::vector<int32_t> trace_logit_row(vocab_size_z);
	st = LogitsSite(trace_final_codes.data(), hidden_size, head_weights,
	                static_cast<int32_t>(model_view.config.vocab_size), trace_wide_logits.data(),
	                trace_logit_row.data());
	if (st != SslmForwardStatus::Ok) {
		std::fprintf(stderr, "FAILED at stage=trace_logits: status=%s\n", SslmForwardStatusName(st));
		return 1;
	}
	const int32_t trace_token = ArgmaxLowestIndexTieBreak(trace_logit_row.data(), vocab_size_z);

	// --- Step 4 (design S4.1 step 4): the self-check, before anything is -----
	// written. memcmp, exactly this codebase's own bit-for-bit convention.
	const bool token_match = (trace_token == prod_out_token[0]);
	const bool logits_match =
	    (std::memcmp(trace_logit_row.data(), prod_out_logit_row.data(), vocab_size_z * sizeof(int32_t)) ==
	     0);
	if (!token_match || !logits_match) {
		std::fprintf(stderr,
		             "FAILED at stage=self_check: production and manual-replay paths disagree -- "
		             "production_token=%d trace_token=%d token_match=%d logit_row_bit_identical=%d "
		             "(no dump written)\n",
		             prod_out_token[0], trace_token, token_match ? 1 : 0, logits_match ? 1 : 0);
		return 1;
	}
	std::printf("self_check: production and manual-replay paths agree bit-for-bit (token=%d, %zu "
	            "logits)\n",
	            trace_token, vocab_size_z);

	// --- Step 5 (design S4.1 step 5): dump. -----------------------------------
	const uint64_t fingerprint = Fnv1a64(prompt);
	if (!WriteDump(dump_path.c_str(), rows, static_cast<uint64_t>(hidden_size), fingerprint)) {
		std::fprintf(stderr, "FAILED at stage=dump_write: could not write \"%s\"\n", dump_path.c_str());
		return 1;
	}
	std::printf("layer_trace_dumped: %zu rows x %zu hidden_size, prompt_fingerprint=0x%016llX -> %s\n",
	            rows.size(), hidden_size, static_cast<unsigned long long>(fingerprint), dump_path.c_str());
	return 0;
}
