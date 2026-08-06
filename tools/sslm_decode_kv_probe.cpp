// sslm_decode_kv_probe.cpp -- T-1781 cell C10: does the K/V cache the engine
// writes and reads DURING GENERATION (positions the model's own greedy
// argmax produced, never a prompt token) compute the same forward that an
// independent, fresh recompute of the identical token sequence produces.
//
// WHY THIS TOOL EXISTS. Every trace/parity instrument in this campaign
// (sslm_layer_trace.cpp, the T-1691-T-1710 campaign, T-1762/1763/1768/1769/
// 1772's probes) captures a teacher-forced PROMPT's own last-token forward --
// verified at source, T-1780 sec 2/C10. None of them ever runs
// RunGreedyDecodeLoop's generation loop past its first produced token, so
// none of them ever exercises a K/V-cache read at a position the loop itself
// wrote moments earlier. This is also the one regime where this campaign's
// own reproducible output failure lives (Claude/Brunel/t1679-float-compare-
// build-2026-08-02.md: an arithmetic prompt answered 27 by float and 17 by
// the engine, reproduced) -- a wrong DIGIT chosen mid-generation is exactly
// what a defect in this regime would produce.
//
// TWO ARMS, ONE PRODUCTION CALL, ONE INDEPENDENT RECOMPUTE.
//
//   Arm A -- production, unmodified: ONE call to RunGreedyDecodeLoop with
//   max_new_tokens = K. Internally this prefills the prompt, then loops:
//   embed the last produced (or last prompt) token, run every layer, argmax,
//   append -- so produced_logit_rows[i] (i >= 1) is the logit row computed by
//   EMBEDDING A TOKEN THE MODEL ITSELF JUST CHOSE (produced_tokens[i-1]) and
//   reading back a K/V cache the SAME loop wrote a step earlier. i == 0 is
//   excluded from the self-generated population: it embeds the prompt's own
//   last token, not a self-generated one (forward_sites.cpp:1693).
//
//   Arm B -- independent recompute, per self-generated position: a BRAND
//   NEW SequenceLayerState and workspace, with NO relationship to Arm A's
//   cache. This arm re-embeds every token 0..p-1 (prompt tokens plus
//   whichever of the model's own earlier choices precede position p) ONE AT
//   A TIME via EmbedEntry + full-budget RunLayerLoop -- the same prefill
//   composition sslm_layer_trace.cpp's own prefill loop and interior-row
//   oracle already use -- then performs ONE MORE whole-token step at
//   position p itself, snapshotting all 29 rows (embedding + 28 layers) the
//   same way sslm_layer_trace.cpp's own per-layer walk does, so the result
//   is directly comparable to a float_reference_layer_dump.py-shaped dump.
//   Arm B never calls RunGreedyDecodeLoop and never touches Arm A's
//   workspace or SequenceLayerState -- a defect specific to how the
//   generation loop advances `seq` across the prefill/generation boundary
//   cannot also be present in Arm B, because Arm B is built via one uniform
//   prefill that treats every token (including the model's own prior
//   choices) as if it had been supplied as a prompt token from position 0.
//
// SELF-CHECK (before anything is written): Arm B's final logit row and
// argmax token, at each self-generated position, are compared bit-for-bit
// (memcmp) against Arm A's produced_logit_rows[i]/produced_tokens[i] at the
// same position. A mismatch is this tool's own finding about K/V-cache
// correctness at that position -- reported loud, no dump written for that
// position, exactly this codebase's established fail-loud convention
// (sslm_layer_trace.cpp's own "FAILED at stage=..." lines).
//
// This self-check is SELF-REFERENTIAL (T-1780 C10's own ruling): it tests
// whether the decode loop's cache bookkeeping agrees with a fresh recompute
// of the identical arithmetic, not whether either arm agrees with ground
// truth. Grading against an INDEPENDENT reference (float_reference_
// layer_dump.py, run on the same literal token sequence via a raw-text
// sibling script) is a SEPARATE step this tool enables but does not itself
// perform -- Arm B's dump is written in exactly sslm_layer_trace.cpp's own
// format so tools/layer_bisection_report.py's existing comparator
// (compare_layer_row, check_provenance) can be reused unmodified.
//
// Usage:
//   sslm_decode_kv_probe <model.sslm> <tokenizer.sslm> "<prompt>" \
//       --max-new K --dump-dir <path> --label <name>
//
// "<prompt>" is the full chat-templated prompt text (system+user+assistant-
// open), the same convention every other tool in this tree uses.
//
// For each self-generated position i = 1..K-1 (position p = prompt_len-1+i),
// on success this tool writes:
//   <dump-dir>/<label>.pos<i>.int8.bin   -- Arm B's 29-row snapshot, exactly
//                                            sslm_layer_trace.cpp's dump format
//   <dump-dir>/<label>.pos<i>.text.txt   -- the literal text
//                                            (prefix + decoded continuation
//                                            through position p) for the
//                                            float-side raw-text driver, plus
//                                            the expected total token count
//                                            and the Fnv1a64 fingerprint used
//                                            in the dump, one per line
// and prints, per position: self-check result, the round-trip verification
// (re-encoding the constructed text reproduces the exact expected token
// sequence -- checked, not assumed) result.

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
	             "usage: %s <model.sslm> <tokenizer.sslm> \"<prompt>\" --max-new K --dump-dir "
	             "<path> --label <name>\n",
	             argv0);
}

// Verbatim of sslm_layer_trace.cpp's own Fnv1a64 -- same constants, same
// composition, so a fingerprint computed here and one computed by the
// float-side raw-text driver (which imports the Python twin of this same
// function) agree over identical text.
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

bool WriteDump(const std::string& path, const std::vector<LayerSnapshot>& rows,
               uint64_t hidden_size, uint64_t prompt_fingerprint) {
	std::ofstream f(path, std::ios::binary | std::ios::trunc);
	if (!f) return false;
	const uint64_t nrows = static_cast<uint64_t>(rows.size());
	const uint64_t capture_mode = 1;
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
	size_t max_new = 4;
	std::string dump_dir;
	std::string label = "probe";
	for (int i = 4; i < argc; ++i) {
		if (std::strcmp(argv[i], "--max-new") == 0 && i + 1 < argc) {
			max_new = static_cast<size_t>(std::stoul(argv[++i]));
		} else if (std::strcmp(argv[i], "--dump-dir") == 0 && i + 1 < argc) {
			dump_dir = argv[++i];
		} else if (std::strcmp(argv[i], "--label") == 0 && i + 1 < argc) {
			label = argv[++i];
		} else {
			std::fprintf(stderr, "unrecognized argument: %s\n", argv[i]);
			PrintUsage(argv[0]);
			return 2;
		}
	}
	if (dump_dir.empty()) {
		std::fprintf(stderr, "FAILED at stage=args: --dump-dir <path> is required\n");
		return 2;
	}
	if (max_new < 2) {
		std::fprintf(stderr,
		             "FAILED at stage=args: --max-new must be >= 2 (need at least one "
		             "self-generated position; position i==0 embeds the prompt's own last "
		             "token, never a self-generated one)\n");
		return 2;
	}

	// --- Stage 1: tokenizer + prompt encode. -----------------------------
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

	// --- Stage 2: model load + per-layer marshal. -------------------------
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

	// --- Arm A: production, unmodified. One call, max_new_tokens = K. -----
	std::vector<uint8_t> prod_workspace(kv_bytes);
	std::vector<int8_t> prod_hidden_codes(hidden_size);
	SequenceLayerState prod_seq;
	prod_seq.hidden_codes = prod_hidden_codes.data();

	std::vector<int32_t> prod_out_tokens(max_new);
	std::vector<int32_t> prod_out_logit_rows(max_new * vocab_size_z);
	size_t prod_tokens_produced = 0;
	SslmDecodeStopReason prod_stop_reason = SslmDecodeStopReason::MaxTokensReached;

	const SslmForwardStatus prod_status = RunGreedyDecodeLoop(
	    prod_seq, layers.data(), num_hidden_layers, hidden_size, model_view.config.head_dim,
	    num_kv_heads, model_view.config.intermediate_size, context_cap, model_view.rope_tables,
	    prompt_tokens.data(), prompt_tokens.size(), embed_weights, embed_site_constant,
	    final_norm_gain.data(), final_norm_site_constant, head_weights,
	    static_cast<int32_t>(model_view.config.vocab_size), /*stop_ids=*/nullptr, /*stop_count=*/0,
	    max_new, prod_workspace.data(), prod_workspace.size(), prod_out_tokens.data(),
	    prod_out_logit_rows.data(), prod_out_tokens.size(), &prod_tokens_produced, &prod_stop_reason,
	    model_view.config.kv_precision);
	if (prod_status != SslmForwardStatus::Ok) {
		std::fprintf(stderr, "FAILED at stage=production_decode: status=%s\n",
		             SslmForwardStatusName(prod_status));
		return 1;
	}
	if (prod_tokens_produced < max_new) {
		std::fprintf(stderr,
		             "FAILED at stage=production_decode: stopped early at %zu/%zu tokens "
		             "(stop_reason=%d) -- self-generated population below what --max-new asked "
		             "for; re-run with a prompt/max-new combination that does not hit a stop id "
		             "this early\n",
		             prod_tokens_produced, max_new, static_cast<int>(prod_stop_reason));
		return 1;
	}
	std::printf("arm_a_production: %zu tokens produced, stop_reason=%d\n", prod_tokens_produced,
	            static_cast<int>(prod_stop_reason));
	{
		std::string decoded = tokenizer.Decode(
		    std::vector<int32_t>(prod_out_tokens.begin(), prod_out_tokens.begin() + prod_tokens_produced));
		std::printf("arm_a_produced_ids:");
		for (size_t i = 0; i < prod_tokens_produced; ++i) std::printf(" %d", prod_out_tokens[i]);
		std::printf("\narm_a_decoded: \"%s\"\n", decoded.c_str());
	}

	// --- Arm B: one independent recompute per self-generated position. ----
	// Position i (1 <= i < max_new): the forward call that PRODUCED
	// prod_out_tokens[i] embedded prod_out_tokens[i-1] (a self-generated
	// token) at absolute position (prompt_len - 1 + i). Arm B recomputes
	// that exact forward from a brand-new SequenceLayerState via one
	// uniform prefill of every token 0..(prompt_len-1+i)-1 followed by one
	// whole-token step at that position -- capturing all 29 rows
	// (embedding + 28 layers) the same way sslm_layer_trace.cpp's own
	// per-layer walk does.
	int mismatches = 0;
	int positions_checked = 0;
	int text_roundtrip_failures = 0;

	for (size_t i = 1; i < max_new; ++i) {
		// Full token sequence up to and including the position being probed.
		std::vector<int32_t> full_seq(prompt_tokens);
		for (size_t j = 0; j + 1 <= i; ++j) full_seq.push_back(prod_out_tokens[j]);
		// full_seq now has prompt_tokens.size() + i elements; the LAST
		// element (index full_seq.size()-1) is the token embedded at
		// position (prompt_len - 1 + i) -- prod_out_tokens[i-1] for i>=1.

		std::vector<uint8_t> b_workspace(kv_bytes);
		std::vector<int8_t> b_hidden_codes(hidden_size);
		SequenceLayerState b_seq;
		b_seq.hidden_codes = b_hidden_codes.data();

		std::vector<LayerSnapshot> rows;
		rows.reserve(num_hidden_layers + 1);

		auto EmbedWholeToken = [&](int32_t token) -> SslmForwardStatus {
			std::vector<int8_t> embed_codes(hidden_size);
			CarriedScale embed_scale{};
			const SslmForwardStatus est =
			    EmbedEntry(token, static_cast<int32_t>(model_view.config.vocab_size), embed_weights,
			               hidden_size, embed_site_constant, embed_codes.data(), &embed_scale);
			if (est != SslmForwardStatus::Ok) return est;
			for (size_t k = 0; k < hidden_size; ++k) b_seq.hidden_codes[k] = embed_codes[k];
			b_seq.hidden_scale = embed_scale;
			b_seq.layer_index = 0;
			return SslmForwardStatus::Ok;
		};

		// Prefill: every token before the target position.
		for (size_t j = 0; j + 1 < full_seq.size(); ++j) {
			SslmForwardStatus st = EmbedWholeToken(full_seq[j]);
			if (st != SslmForwardStatus::Ok) {
				std::fprintf(stderr, "FAILED at stage=armB_prefill_embed: pos=%zu index=%zu status=%s\n",
				             i, j, SslmForwardStatusName(st));
				return 1;
			}
			st = RunLayerLoop(b_seq, layers.data(), num_hidden_layers,
			                   /*layer_budget=*/num_hidden_layers, hidden_size,
			                   model_view.config.head_dim, num_kv_heads,
			                   model_view.config.intermediate_size, context_cap,
			                   model_view.rope_tables, b_workspace.data(), b_workspace.size());
			if (st != SslmForwardStatus::Ok) {
				std::fprintf(stderr, "FAILED at stage=armB_prefill_layers: pos=%zu index=%zu status=%s\n",
				             i, j, SslmForwardStatusName(st));
				return 1;
			}
		}

		// The target position: embed, snapshot row 0 (embedding), then
		// RunLayerLoop(layer_budget=1) 28 times, snapshotting after each --
		// exactly sslm_layer_trace.cpp's own per-layer walk for its last
		// token.
		{
			const SslmForwardStatus st = EmbedWholeToken(full_seq.back());
			if (st != SslmForwardStatus::Ok) {
				std::fprintf(stderr, "FAILED at stage=armB_target_embed: pos=%zu status=%s\n", i,
				             SslmForwardStatusName(st));
				return 1;
			}
		}
		rows.push_back(LayerSnapshot{
		    std::vector<int8_t>(b_seq.hidden_codes, b_seq.hidden_codes + hidden_size), b_seq.hidden_scale.m,
		    b_seq.hidden_scale.e});
		for (uint32_t step = 0; step < num_hidden_layers; ++step) {
			const SslmForwardStatus st =
			    RunLayerLoop(b_seq, layers.data(), num_hidden_layers, /*layer_budget=*/1, hidden_size,
			                 model_view.config.head_dim, num_kv_heads, model_view.config.intermediate_size,
			                 context_cap, model_view.rope_tables, b_workspace.data(), b_workspace.size());
			if (st != SslmForwardStatus::Ok) {
				std::fprintf(stderr, "FAILED at stage=armB_target_layer: pos=%zu layer=%u status=%s\n", i,
				             step, SslmForwardStatusName(st));
				return 1;
			}
			rows.push_back(LayerSnapshot{
			    std::vector<int8_t>(b_seq.hidden_codes, b_seq.hidden_codes + hidden_size),
			    b_seq.hidden_scale.m, b_seq.hidden_scale.e});
		}

		// final_norm -> logits -> argmax from Arm B's own captured state.
		std::vector<int8_t> b_final_codes(hidden_size);
		CarriedScale b_final_scale{};
		SslmForwardStatus st =
		    RmsNormSite(b_seq.hidden_codes, final_norm_gain.data(), hidden_size, b_seq.hidden_scale,
		                final_norm_site_constant, b_final_codes.data(), &b_final_scale, "final_norm");
		if (st != SslmForwardStatus::Ok) {
			std::fprintf(stderr, "FAILED at stage=armB_final_norm: pos=%zu status=%s\n", i,
			             SslmForwardStatusName(st));
			return 1;
		}
		std::vector<int64_t> b_wide_logits(vocab_size_z);
		std::vector<int32_t> b_logit_row(vocab_size_z);
		st = LogitsSite(b_final_codes.data(), hidden_size, head_weights,
		                static_cast<int32_t>(model_view.config.vocab_size), b_wide_logits.data(),
		                b_logit_row.data());
		if (st != SslmForwardStatus::Ok) {
			std::fprintf(stderr, "FAILED at stage=armB_logits: pos=%zu status=%s\n", i,
			             SslmForwardStatusName(st));
			return 1;
		}
		const int32_t b_token = ArgmaxLowestIndexTieBreak(b_logit_row.data(), vocab_size_z);

		// --- Self-check: Arm B vs Arm A, bit-for-bit, at this position. ---
		positions_checked++;
		const int32_t* a_row = prod_out_logit_rows.data() + i * vocab_size_z;
		const bool token_match = (b_token == prod_out_tokens[i]);
		const bool logits_match =
		    (std::memcmp(b_logit_row.data(), a_row, vocab_size_z * sizeof(int32_t)) == 0);
		if (!token_match || !logits_match) {
			mismatches++;
			std::fprintf(stderr,
			             "KV_CACHE_MISMATCH: pos=%zu (absolute token position %zu, embedding "
			             "self-generated token %d) armA_token=%d armB_token=%d token_match=%d "
			             "logit_row_bit_identical=%d\n",
			             i, prompt_tokens.size() - 1 + i, full_seq.back(), prod_out_tokens[i], b_token,
			             token_match ? 1 : 0, logits_match ? 1 : 0);
		} else {
			std::printf(
			    "self_check_ok: pos=%zu (absolute token position %zu) armA and armB agree "
			    "bit-for-bit (token=%d, %zu logits)\n",
			    i, prompt_tokens.size() - 1 + i, b_token, vocab_size_z);
		}

		// --- Text reconstruction + round-trip verification, for the float- --
		// side grading step. Decode ONLY the self-generated continuation
		// (never re-templated special tokens) and append to the literal
		// prompt text this tool was invoked with.
		const std::vector<int32_t> continuation(prod_out_tokens.begin(), prod_out_tokens.begin() + i);
		const std::string continuation_text = tokenizer.Decode(continuation);
		const std::string full_text = prompt + continuation_text;
		const std::vector<int32_t> reencoded = tokenizer.Encode(full_text);
		const bool roundtrip_ok = (reencoded == full_seq);
		if (!roundtrip_ok) {
			text_roundtrip_failures++;
			std::fprintf(stderr,
			             "TEXT_ROUNDTRIP_FAILED: pos=%zu re-encoding the decoded continuation text "
			             "does not reproduce the expected token sequence (expected %zu tokens, got "
			             "%zu) -- this position is EXCLUDED from float-reference grading, the "
			             "engine-internal self-check above is unaffected\n",
			             i, full_seq.size(), reencoded.size());
		} else {
			std::printf("text_roundtrip_ok: pos=%zu (%zu tokens)\n", i, full_seq.size());
		}

		// --- Dump (always written, even on a self-check mismatch, so the ---
		// disagreement itself can be graded against the float reference --
		// this tool's own fail-loud convention differs from
		// sslm_layer_trace.cpp's here deliberately: a K/V-cache mismatch IS
		// this tool's reportable finding, not a build defect to hide the
		// dump behind).
		if (!dump_dir.empty()) {
			const std::string int8_path = dump_dir + "/" + label + ".pos" + std::to_string(i) + ".int8.bin";
			const uint64_t fingerprint = Fnv1a64(full_text);
			if (!WriteDump(int8_path, rows, static_cast<uint64_t>(hidden_size), fingerprint)) {
				std::fprintf(stderr, "FAILED at stage=dump_write: could not write \"%s\"\n",
				             int8_path.c_str());
				return 1;
			}
			if (roundtrip_ok) {
				const std::string text_path =
				    dump_dir + "/" + label + ".pos" + std::to_string(i) + ".text.txt";
				std::ofstream tf(text_path, std::ios::binary | std::ios::trunc);
				tf << full_text << "\n";
				tf << "expect_tokens=" << full_seq.size() << "\n";
				tf << "fingerprint=" << fingerprint << "\n";
			}
			std::printf("dumped: pos=%zu -> %s (fingerprint=0x%016llX)\n", i, int8_path.c_str(),
			            static_cast<unsigned long long>(fingerprint));
		}
	}

	std::printf(
	    "SUMMARY: label=%s positions_checked=%d self_check_mismatches=%d "
	    "text_roundtrip_failures=%d\n",
	    label.c_str(), positions_checked, mismatches, text_roundtrip_failures);
	return mismatches > 0 ? 3 : 0;
}
