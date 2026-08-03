// sslm_hook_invariance_check.cpp -- T-1300, executing the instrumentation-
// invariance measurement §10.3 states by construction ("a forward with the
// hook installed produces digests identical to one without",
// include/superslm/trace_hook.h) but that no prior run had checked by
// execution (StandardsDocument.md §5.4: exactness is verified at source or
// by execution, never by construction).
//
// WHAT THIS TOOL DOES. Loads the real .sslm model artifact through the real
// production loader (SslmModel::Load -- never a fixture, D-SLM721), then for
// each of tools/logit_margin_report.py's own nine PROMPT_SET prompts runs
// three arms over that same loaded artifact and compares their digests
// (superslm::ComputeTokenDigest / ComputeFinalLogitDigest, the parity gate's
// own oracle, §10.1):
//
//   arm A -- PRODUCTION: RunGreedyDecodeLoop exactly as tools/sslm_generate.cpp
//            calls it. This is the code path that ships; it has no trace-hook
//            parameter at all (forward_sites.h's own declaration), so it is
//            incapable of installing a hook.
//   arm B -- MANUAL, NO HOOK: the same prefill-then-generate composition
//            RunGreedyDecodeLoop's own RunWholeToken step uses internally
//            (EmbedEntry + RunLayerLoop, then RmsNormSite("final_norm") +
//            LogitsSite + ArgmaxLowestIndexTieBreak per step) -- the same
//            composition tools/sslm_layer_trace.cpp's own manual replay
//            already runs and self-checks bit-for-bit against arm A's single-
//            token forward before it will write a dump. trace_hook_state is
//            passed as nullptr throughout, matching every default in
//            forward_sites.h.
//   arm C -- MANUAL, HOOK INSTALLED: byte-for-byte the same composition as
//            arm B, over a FRESH SequenceLayerState/workspace so no state
//            leaks between arms, with a real SslmTraceHookState installed
//            (SslmSetTraceHook) at every RunLayerLoop/RmsNormSite call. The
//            installed hook is not a no-op: it folds every record's fields
//            into a running FNV-1a accumulator (--hook-digest below), so the
//            optimizer cannot prove the hook call is dead and elide it.
//
// B vs A grounds that the manual composition this tool must use to reach the
// hook API at all (RunGreedyDecodeLoop itself cannot receive one) is not a
// second, different engine -- it is required to already match production
// bit-for-bit before B vs C is trusted. B vs C is §10.3's own claim, executed:
// hook-installed and hook-free digests over the SAME composition, the SAME
// loaded artifact, the SAME prompt.
//
// --corrupt-hook: proves the comparison CAN fail (StandardsDocument.md §4's
// independent-population validation habit, applied at n=1 by deliberate
// injection here since there is no independently-known population of hook-
// perturbation defects to sweep). With this flag, the installed hook in arm C
// additionally overwrites every byte of the chain record's own `x_int`/`codes`
// spans -- non-owning views into the SAME buffer RunLayerLoop's arithmetic
// reads next (trace_hook.h's own documented lifetime: "valid only for the
// duration of the hook call that carries them"), which is a real, physically
// reachable way a non-conformant hook breaks the read-only-by-construction
// contract that header claims. This is expected to make arm B and arm C's
// digests diverge; run once, confirm the tool reports MISMATCH, then re-run
// without the flag for the real measurement. Never used for the banked cell.
//
// Usage: sslm_hook_invariance_check <model.sslm> <tokenizer.sslm>
//            [--max-new N] [--corrupt-hook]
//
// Exit 0 iff every prompt's arm A/B and arm B/C digest pairs matched (or, in
// --corrupt-hook mode, iff every arm B/C pair MISMATCHED, proving the check
// has power). Exit 1 on any unexpected mismatch or engine-level failure.

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "superslm/artifact.h"
#include "superslm/decode_digest.h"
#include "superslm/forward_sites.h"
#include "superslm/model.h"
#include "superslm/tokenizer.h"
#include "superslm/trace_hook.h"
#include "sslm_marshal.h"

using namespace superslm;
using superslm_marshal::LayerBacking;
using superslm_marshal::MarshalLayer;
using superslm_marshal::PreflightScanWscFolds;
using superslm_marshal::ReadCarriedScale;
using superslm_marshal::ReadFile;
using superslm_marshal::WidenGainToInt32;

namespace {

// tools/logit_margin_report.py's own SYSTEM_PROMPT and chat-template shape
// (build_prompt), and its PROMPT_SET verbatim (nine prompts) -- the same
// population the T-1683 campaign's banked per-site results (D-SLM747) were
// captured over, so this measurement speaks to those results (T-1300 brief).
const char* kSystemPrompt = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.";

struct PromptCase {
	const char* name;
	const char* question;
};

const PromptCase kPromptSet[] = {
    {"digit_symbol_spaced", "What is 12 + 15? Give just the number."},
    {"digit_symbol_nospace", "What is 12+15? Give just the number."},
    {"plain_language_plus", "What is 12 plus 15? Give just the number."},
    {"digit_symbol_mult", "What is 17 x 23? Give just the number."},
    {"capital_of_germany", "What is the capital of Germany?"},
    {"capital_of_france", "What is the capital of France?"},
    {"capital_of_japan", "What is the capital of Japan?"},
    {"largest_planet", "Name the largest planet in the solar system."},
    {"days_in_week", "How many days are in a week? Give just the number."},
};

std::string BuildPrompt(const std::string& question) {
	return std::string("<|im_start|>system\n") + kSystemPrompt + "<|im_end|>\n<|im_start|>user\n" +
	       question + "<|im_end|>\n<|im_start|>assistant\n";
}

void PrintUsage(const char* argv0) {
	std::fprintf(stderr, "usage: %s <model.sslm> <tokenizer.sslm> [--max-new N] [--corrupt-hook]\n",
	             argv0);
}

void PrintDigest(const uint8_t d[32], char out_hex[65]) {
	static const char* kHex = "0123456789abcdef";
	for (int i = 0; i < 32; ++i) {
		out_hex[2 * i] = kHex[(d[i] >> 4) & 0xF];
		out_hex[2 * i + 1] = kHex[d[i] & 0xF];
	}
	out_hex[64] = '\0';
}

bool DigestsEqual(const uint8_t a[32], const uint8_t b[32]) { return std::memcmp(a, b, 32) == 0; }

// --- The hook under test. -------------------------------------------------
//
// Real work (never a no-op): folds every field of every chain/K-V-landing
// record it is called with into a running FNV-1a-64 accumulator, so the
// compiler cannot prove the call has no observable effect and delete it.
// Never writes through `record->x_int`/`record->codes` in the normal
// (conformant) mode -- those spans are read via their `const` accessor only.
struct HookSink {
	uint64_t fnv = UINT64_C(14695981039346656037);
	uint64_t call_count = 0;
	bool corrupt = false;  // --corrupt-hook: deliberately break the read-only contract.

	void Fold(uint64_t v) {
		fnv ^= v;
		fnv *= UINT64_C(1099511628211);
	}
};

void FoldBytes(HookSink& sink, const void* p, size_t n) {
	const auto* b = static_cast<const uint8_t*>(p);
	for (size_t i = 0; i < n; ++i) sink.Fold(b[i]);
}

void ChainHookFn(const SslmChainTraceRecord* chain, const SslmKvLandingTraceRecord* kv,
                  void* user) {
	auto* sink = static_cast<HookSink*>(user);
	sink->call_count++;
	if (chain != nullptr) {
		FoldBytes(*sink, chain->site.data(), chain->site.size());
		sink->Fold(static_cast<uint64_t>(chain->token_index));
		FoldBytes(*sink, chain->x_int.data(), chain->x_int.size() * sizeof(int64_t));
		sink->Fold(static_cast<uint64_t>(chain->d_prime));
		sink->Fold(static_cast<uint64_t>(chain->dn));
		sink->Fold(static_cast<uint64_t>(chain->s));
		sink->Fold(static_cast<uint64_t>(chain->r));
		FoldBytes(*sink, chain->codes.data(), chain->codes.size());
		sink->Fold(static_cast<uint64_t>(chain->m_out));
		sink->Fold(static_cast<uint64_t>(chain->e_out));
		if (sink->corrupt) {
			// --corrupt-hook only: the record's spans are non-owning views into
			// the SAME buffers RunLayerLoop's own arithmetic reads next
			// (trace_hook.h's documented lifetime). A non-conformant hook can
			// reach through the `const` view with a const_cast and write; this is
			// the fault this tool injects to prove the digest comparison below
			// has the power to detect a perturbed forward. Never run for the
			// banked measurement.
			auto* mutable_codes = const_cast<int8_t*>(chain->codes.data());
			for (size_t i = 0; i < chain->codes.size(); ++i) mutable_codes[i] = static_cast<int8_t>(~mutable_codes[i]);
		}
	}
	if (kv != nullptr) {
		FoldBytes(*sink, kv->site.data(), kv->site.size());
		sink->Fold(static_cast<uint64_t>(kv->token_index));
		sink->Fold(static_cast<uint64_t>(kv->head));
		FoldBytes(*sink, kv->x_int.data(), kv->x_int.size() * sizeof(int64_t));
		sink->Fold(static_cast<uint64_t>(kv->m_in));
		sink->Fold(static_cast<uint64_t>(kv->e_in));
		FoldBytes(*sink, kv->codes.data(), kv->codes.size());
		sink->Fold(static_cast<uint64_t>(kv->m_out));
		sink->Fold(static_cast<uint64_t>(kv->e_out));
	}
}

// One full greedy-decode run, manually composed from EmbedEntry/RunLayerLoop/
// RmsNormSite/LogitsSite/ArgmaxLowestIndexTieBreak -- the exact composition
// RunGreedyDecodeLoop's own RunWholeToken step uses internally
// (src/forward/forward_sites.cpp) -- with an optional trace hook threaded
// through every call that accepts one. Mirrors RunGreedyDecodeLoop's own
// stop-id/max-token contract (§9.1) so its output is directly comparable to
// arm A's.
struct ManualRunResult {
	SslmForwardStatus status = SslmForwardStatus::Ok;
	std::vector<int32_t> tokens;
	std::vector<int32_t> logit_rows;  // rows_produced * vocab_size, row-major
};

ManualRunResult RunManualDecode(const SslmModelView& model_view, const std::vector<LayerWeights>& layers,
                                 const int8_t* embed_weights, CarriedScale embed_site_constant,
                                 const int32_t* final_norm_gain, CarriedScale final_norm_site_constant,
                                 const int8_t* head_weights, const std::vector<int32_t>& prompt_tokens,
                                 const std::vector<int32_t>& stop_ids, size_t max_new_tokens,
                                 SslmTraceHookState* hook_state) {
	ManualRunResult result;
	const uint32_t num_hidden_layers = model_view.config.num_hidden_layers;
	const size_t hidden_size = model_view.config.hidden_size;
	const uint32_t num_kv_heads = model_view.config.num_key_value_heads;
	const int64_t context_cap = static_cast<int64_t>(model_view.config.context_cap);
	const size_t vocab_size = model_view.config.vocab_size;
	const size_t kv_bytes = static_cast<size_t>(num_hidden_layers) * static_cast<size_t>(context_cap) *
	                         num_kv_heads * model_view.config.head_dim * 2;

	std::vector<uint8_t> workspace(kv_bytes);
	std::vector<int8_t> hidden_codes(hidden_size);
	SequenceLayerState seq;
	seq.hidden_codes = hidden_codes.data();

	std::vector<int8_t> embed_codes(hidden_size);
	std::vector<int8_t> final_codes(hidden_size);
	std::vector<int64_t> wide_logits(vocab_size);
	std::vector<int32_t> logit_row(vocab_size);

	auto RunWholeToken = [&](int32_t token) -> SslmForwardStatus {
		CarriedScale embed_scale{};
		const SslmForwardStatus est =
		    EmbedEntry(token, static_cast<int32_t>(model_view.config.vocab_size), embed_weights,
		               hidden_size, embed_site_constant, embed_codes.data(), &embed_scale);
		if (est != SslmForwardStatus::Ok) return est;
		for (size_t i = 0; i < hidden_size; ++i) seq.hidden_codes[i] = embed_codes[i];
		seq.hidden_scale = embed_scale;
		seq.layer_index = 0;
		return RunLayerLoop(seq, layers.data(), num_hidden_layers, /*layer_budget=*/num_hidden_layers,
		                     hidden_size, model_view.config.head_dim, num_kv_heads,
		                     model_view.config.intermediate_size, context_cap, model_view.rope_tables,
		                     workspace.data(), workspace.size(), /*site_prefix=*/{},
		                     /*token_index=*/0, hook_state);
	};

	for (size_t i = 0; i + 1 < prompt_tokens.size(); ++i) {
		result.status = RunWholeToken(prompt_tokens[i]);
		if (result.status != SslmForwardStatus::Ok) return result;
	}

	int32_t current_token = prompt_tokens.back();
	while (result.tokens.size() < max_new_tokens) {
		result.status = RunWholeToken(current_token);
		if (result.status != SslmForwardStatus::Ok) return result;

		CarriedScale final_scale{};
		result.status = RmsNormSite(seq.hidden_codes, final_norm_gain, hidden_size, seq.hidden_scale,
		                             final_norm_site_constant, final_codes.data(), &final_scale,
		                             "final_norm", /*token_index=*/0, hook_state);
		if (result.status != SslmForwardStatus::Ok) return result;

		result.status = LogitsSite(final_codes.data(), hidden_size, head_weights, vocab_size,
		                            wide_logits.data(), logit_row.data());
		if (result.status != SslmForwardStatus::Ok) return result;

		const int32_t token = ArgmaxLowestIndexTieBreak(logit_row.data(), vocab_size);
		result.tokens.push_back(token);
		result.logit_rows.insert(result.logit_rows.end(), logit_row.begin(), logit_row.end());

		bool matched = false;
		for (int32_t s : stop_ids) {
			if (s == token) {
				matched = true;
				break;
			}
		}
		if (matched) break;
		current_token = token;
	}
	return result;
}

}  // namespace

int main(int argc, char** argv) {
	if (argc < 3) {
		PrintUsage(argv[0]);
		return 2;
	}
	const std::string model_path = argv[1];
	const std::string tokenizer_path = argv[2];
	size_t max_new_tokens = 6;
	bool corrupt_hook = false;
	for (int i = 3; i < argc; ++i) {
		if (std::strcmp(argv[i], "--max-new") == 0 && i + 1 < argc) {
			max_new_tokens = static_cast<size_t>(std::atoi(argv[++i]));
		} else if (std::strcmp(argv[i], "--corrupt-hook") == 0) {
			corrupt_hook = true;
		} else {
			std::fprintf(stderr, "unrecognized argument: %s\n", argv[i]);
			PrintUsage(argv[0]);
			return 2;
		}
	}

	// --- Load the tokenizer (real artifact, same path as sslm_generate.cpp). ---
	std::vector<uint8_t> tok_bytes;
	if (!ReadFile(tokenizer_path.c_str(), tok_bytes)) {
		std::fprintf(stderr, "FAILED at stage=tokenizer_file_read: could not read \"%s\"\n",
		             tokenizer_path.c_str());
		return 1;
	}
	SslmArtifact tok_artifact;
	SslmError tok_open_err;
	if (SslmArtifact::OpenFromMemory(tok_bytes.data(), tok_bytes.size(), tok_artifact, &tok_open_err) !=
	    SslmStatus::Ok) {
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

	// --- Load the model through the real production entry point. --------------
	std::vector<uint8_t> model_bytes;
	if (!ReadFile(model_path.c_str(), model_bytes)) {
		std::fprintf(stderr, "FAILED at stage=model_file_read: could not read \"%s\"\n", model_path.c_str());
		return 1;
	}
	SslmModelView model_view;
	std::string model_err;
	if (SslmModel::Load(model_bytes.data(), model_bytes.size(), model_view, &model_err) !=
	    SslmModelStatus::Ok) {
		std::fprintf(stderr, "FAILED at stage=model_load: diagnostic=\"%s\"\n", model_err.c_str());
		return 1;
	}

	const uint32_t num_heads = model_view.config.num_attention_heads;
	const uint32_t num_kv_heads = model_view.config.num_key_value_heads;
	const uint32_t num_hidden_layers = model_view.config.num_hidden_layers;
	const size_t hidden_size = model_view.config.hidden_size;

	PreflightScanWscFolds(model_view);
	std::vector<LayerBacking> backings(num_hidden_layers);
	std::vector<LayerWeights> layers(num_hidden_layers);
	for (uint32_t l = 0; l < num_hidden_layers; ++l) {
		std::string marshal_err;
		if (!MarshalLayer(model_view, l, num_heads, num_kv_heads, backings[l], layers[l], &marshal_err)) {
			std::fprintf(stderr, "FAILED at stage=layer_weights_marshal: layer=%u diagnostic=\"%s\"\n", l,
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
	std::vector<int32_t> final_norm_gain = WidenGainToInt32(*final_gain_w);
	bool ok = true;
	CarriedScale embed_site_constant = ReadCarriedScale(model_view.composition_constants, "embed", &ok);
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
			std::fprintf(stderr, "FAILED at stage=head_marshal: tie_word_embeddings=0, no lm_head tensor\n");
			return 1;
		}
		head_weights = reinterpret_cast<const int8_t*>(lm_head_w->data);
	}

	const int64_t context_cap = static_cast<int64_t>(model_view.config.context_cap);
	const size_t kv_bytes = static_cast<size_t>(num_hidden_layers) * static_cast<size_t>(context_cap) *
	                         num_kv_heads * model_view.config.head_dim * 2;

	const std::vector<int32_t> stop_ids = {151645, 151643};

	std::printf(
	    "T-1300 instrumentation-invariance measurement -- model=%s max_new=%zu corrupt_hook=%d\n",
	    model_path.c_str(), max_new_tokens, corrupt_hook ? 1 : 0);
	std::printf("prompt,ab_match,bc_match,hook_calls\n");

	int prompts_run = 0;
	int ab_mismatches = 0;
	int bc_mismatches = 0;

	for (const PromptCase& pc : kPromptSet) {
		const std::string prompt_text = BuildPrompt(pc.question);
		const std::vector<int32_t> prompt_tokens = tokenizer.Encode(prompt_text);
		if (prompt_tokens.empty()) {
			std::fprintf(stderr, "FAILED at stage=tokenizer_encode: prompt=%s encoded to zero tokens\n",
			             pc.name);
			return 1;
		}

		// --- Arm A: production RunGreedyDecodeLoop, no hook capability. --------
		std::vector<uint8_t> ws_a(kv_bytes);
		std::vector<int8_t> hidden_a(hidden_size);
		SequenceLayerState seq_a;
		seq_a.hidden_codes = hidden_a.data();
		std::vector<int32_t> out_tokens_a(max_new_tokens);
		std::vector<int32_t> out_logits_a(max_new_tokens * model_view.config.vocab_size);
		size_t produced_a = 0;
		SslmDecodeStopReason stop_reason_a;
		const SslmForwardStatus status_a = RunGreedyDecodeLoop(
		    seq_a, layers.data(), num_hidden_layers, hidden_size, model_view.config.head_dim,
		    num_kv_heads, model_view.config.intermediate_size, context_cap, model_view.rope_tables,
		    prompt_tokens.data(), prompt_tokens.size(), embed_weights, embed_site_constant,
		    final_norm_gain.data(), final_norm_site_constant, head_weights,
		    static_cast<int32_t>(model_view.config.vocab_size), stop_ids.data(), stop_ids.size(),
		    max_new_tokens, ws_a.data(), ws_a.size(), out_tokens_a.data(), out_logits_a.data(),
		    out_tokens_a.size(), &produced_a, &stop_reason_a, model_view.config.kv_precision);
		if (status_a != SslmForwardStatus::Ok) {
			std::fprintf(stderr, "FAILED at stage=arm_a_decode: prompt=%s status=%s\n", pc.name,
			             SslmForwardStatusName(status_a));
			return 1;
		}

		// --- Arm B: manual composition, no hook. --------------------------------
		ManualRunResult run_b = RunManualDecode(model_view, layers, embed_weights, embed_site_constant,
		                                         final_norm_gain.data(), final_norm_site_constant,
		                                         head_weights, prompt_tokens, stop_ids, max_new_tokens,
		                                         /*hook_state=*/nullptr);
		if (run_b.status != SslmForwardStatus::Ok) {
			std::fprintf(stderr, "FAILED at stage=arm_b_decode: prompt=%s status=%s\n", pc.name,
			             SslmForwardStatusName(run_b.status));
			return 1;
		}

		// --- Arm C: manual composition, hook installed. -------------------------
		HookSink sink;
		sink.corrupt = corrupt_hook;
		SslmTraceHookState hook_state;
		SslmSetTraceHook(hook_state, &ChainHookFn, &sink);
		if (!SslmTraceHookInstalled(hook_state)) {
			std::fprintf(stderr, "FAILED at stage=arm_c_setup: SslmSetTraceHook did not install\n");
			return 1;
		}
		ManualRunResult run_c = RunManualDecode(model_view, layers, embed_weights, embed_site_constant,
		                                         final_norm_gain.data(), final_norm_site_constant,
		                                         head_weights, prompt_tokens, stop_ids, max_new_tokens,
		                                         &hook_state);
		if (run_c.status != SslmForwardStatus::Ok) {
			std::fprintf(stderr, "FAILED at stage=arm_c_decode: prompt=%s status=%s\n", pc.name,
			             SslmForwardStatusName(run_c.status));
			return 1;
		}
		if (sink.call_count == 0) {
			std::fprintf(stderr,
			             "FAILED at stage=arm_c_verify: prompt=%s hook installed but never called -- "
			             "this run would prove nothing\n",
			             pc.name);
			return 1;
		}

		// --- Digest and compare. -------------------------------------------------
		uint8_t tok_digest_a[32], tok_digest_b[32], tok_digest_c[32];
		uint8_t logit_digest_a[32], logit_digest_b[32], logit_digest_c[32];
		ComputeTokenDigest(out_tokens_a.data(), produced_a, tok_digest_a);
		ComputeFinalLogitDigest(out_logits_a.data(), produced_a, model_view.config.vocab_size,
		                         logit_digest_a);
		ComputeTokenDigest(run_b.tokens.data(), run_b.tokens.size(), tok_digest_b);
		ComputeFinalLogitDigest(run_b.logit_rows.data(), run_b.tokens.size(), model_view.config.vocab_size,
		                         logit_digest_b);
		ComputeTokenDigest(run_c.tokens.data(), run_c.tokens.size(), tok_digest_c);
		ComputeFinalLogitDigest(run_c.logit_rows.data(), run_c.tokens.size(), model_view.config.vocab_size,
		                         logit_digest_c);

		const bool ab_match = DigestsEqual(tok_digest_a, tok_digest_b) &&
		                       DigestsEqual(logit_digest_a, logit_digest_b) &&
		                       produced_a == run_b.tokens.size();
		const bool bc_match = DigestsEqual(tok_digest_b, tok_digest_c) &&
		                       DigestsEqual(logit_digest_b, logit_digest_c) &&
		                       run_b.tokens.size() == run_c.tokens.size();

		char hex_a[65], hex_b[65], hex_c[65];
		PrintDigest(tok_digest_a, hex_a);
		PrintDigest(tok_digest_b, hex_b);
		PrintDigest(tok_digest_c, hex_c);
		std::printf("%s,%d,%d,%llu\n", pc.name, ab_match ? 1 : 0, bc_match ? 1 : 0,
		            static_cast<unsigned long long>(sink.call_count));
		std::printf("  arm_a_token_digest=%s\n", hex_a);
		std::printf("  arm_b_token_digest=%s\n", hex_b);
		std::printf("  arm_c_token_digest=%s\n", hex_c);

		prompts_run++;
		if (!ab_match) ab_mismatches++;
		if (!bc_match) bc_mismatches++;
	}

	std::printf("SUMMARY: prompts_run=%d ab_mismatches=%d bc_mismatches=%d corrupt_hook=%d\n",
	            prompts_run, ab_mismatches, bc_mismatches, corrupt_hook ? 1 : 0);

	if (corrupt_hook) {
		// Fault-injection mode: EVERY prompt is expected to mismatch on B vs C.
		// Any prompt that still matches means the injected fault did not reach
		// the digest, and the check's power is not demonstrated.
		if (bc_mismatches != prompts_run) {
			std::fprintf(stderr,
			             "FAILED at stage=fault_injection_check: expected all %d prompts to MISMATCH "
			             "under --corrupt-hook, only %d did -- the comparison's power is not proven\n",
			             prompts_run, bc_mismatches);
			return 1;
		}
		std::printf("fault injection confirmed: all %d prompts MISMATCHED under --corrupt-hook\n",
		            prompts_run);
		return 0;
	}

	if (ab_mismatches != 0 || bc_mismatches != 0) {
		std::fprintf(stderr,
		             "RESULT: MISMATCH -- ab_mismatches=%d bc_mismatches=%d (see per-prompt digests "
		             "above)\n",
		             ab_mismatches, bc_mismatches);
		return 1;
	}
	std::printf("RESULT: all %d prompts identical across arm A (production), arm B (manual, no "
	            "hook), and arm C (manual, hook installed)\n",
	            prompts_run);
	return 0;
}
