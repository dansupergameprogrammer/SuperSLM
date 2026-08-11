// t1902_domain_gate_probe.cpp -- T-1902 scratch instrument, disposable,
// never merges. Answers exactly one question: does the production engine's
// (`main`@727e63e) per-element `LandingRescale`'s own
// `out_magnitude_exceeded_int64` gate -- which the fused K-landing path
// checks unconditionally and turns into a hard
// `SslmForwardStatus::OptionGFusedLandingExponentOutOfDomain` refusal
// (forward_sites.cpp:1500-1509, RunLayerLoop's 16-parameter
// OptionGKLandingMode::kFused overload) -- ever fire on two real capture
// populations: T-1777's 239-document encoder corpus, and T-1800/T-1818's
// 32-prompt decode population. See Claude/Vitruvius/t1822-activation-scale-
// remedy-design-2026-08-07.md §31.4.2 and D-SLM2465 (D:\Wizard's decision
// log) for why this settles whether the T-1891 spike's Arms A/B captures
// (which ran a build with NO such check) are reusable as production
// baselines.
//
// This file is NOT a modification of any existing production tool and does
// not touch main's own tools/sslm_generate.cpp or the spike's
// tools/t1777_pooled_trace_batch.cpp; it is a new, minimal driver reusing
// their own already-reviewed loading/marshaling boilerplate
// (tools/sslm_generate.cpp) and per-position capture shape
// (claude/t1777-retrieval-agreement's tools/t1777_pooled_trace_batch.cpp,
// Step B) verbatim, stripped of everything this question does not need
// (pooled-embedding dumps, self-checks, saturation counters) and calling
// production's real 16-parameter `RunLayerLoop`/`RunGreedyDecodeLoop`
// overloads directly with `OptionGKLandingMode::kFused` /
// `option_g_fused_k_landing=true` -- the exact call shape T-1900's own
// `TestOptionGSelectionDispatch_EndToEndProductionPath` proved is what an
// artifact with the header `flags` bit set actually drives.
//
// Modes:
//   t1902_domain_gate_probe --selftest
//     Calls LandingRescale directly with the witness T-1900's own build log
//     §7 cites (bc=1, m_a=1, r_t=2147483649, e_a=35, e_t=-59 -- inside the
//     T-1898 probe's own [-60,-25) dangerous band, and a member of the
//     "wrong K code" population that probe's Q1 finding proved always sets
//     this exact flag true, 0 counterexamples across 8,084,656
//     observations), asserts the OUT-PARAMETER reads true, and exits
//     nonzero if it does not. This is the harness's own vitality proof
//     (StandardsDocument.md §4/§5.4): it demonstrates, before any real
//     count is trusted, that a bool read off this exact out-parameter CAN
//     surface as a nonzero report -- the identical mechanism the encoder
//     and decode modes below use to count real occurrences.
//
//   t1902_domain_gate_probe --encoder <model.sslm> <tok.sslm> <prompts.tsv>
//     <prompts.tsv> is T-1777's own file format (one document per line,
//     "<label>\t<chat-templated prompt, literal newlines as \"<NL>\">").
//     For each document: tokenize, then walk every position 0..n-1,
//     embedding that position's token and running num_hidden_layers
//     single-layer RunLayerLoop steps over ONE shared per-document
//     workspace/sequence (t1777_pooled_trace_batch.cpp's own "Step B"
//     shape) -- so every layer x every kv_head x every position in the
//     corpus is exercised, fused, exactly as a real prefill would run it.
//     Any RunLayerLoop call returning
//     OptionGFusedLandingExponentOutOfDomain increments the counter, is
//     logged (document label, position, layer), and that document's own
//     walk stops there (matching production: the call is non-Ok, the
//     forward pass halts) -- the next document still runs. A different
//     non-Ok status is logged separately as an unexpected engine error,
//     not folded into the domain-gate count.
//
//   t1902_domain_gate_probe --decode <model.sslm> <tok.sslm>
//     Runs T-1800's own hardcoded 32-prompt population (verbatim: same
//     prompts, same chat template, same system prompt, same stop ids, same
//     max_new=8 budget as tools/t1800_decode_agreement.py) through
//     RunGreedyDecodeLoop once per prompt, counting
//     OptionGFusedLandingExponentOutOfDomain returns the same way.
//
// Requires model_view.option_g_fused_k_landing == true (i.e., the artifact
// was loaded with header flags bit 0 set) -- aborts loudly at startup if
// not, in either --encoder or --decode mode, because a legacy-path run
// would trivially report zero and that zero would be a false pass (T-1902
// brief §3).

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
	             "usage:\n"
	             "  %s --selftest\n"
	             "  %s --encoder <model.sslm> <tokenizer.sslm> <prompts.tsv>\n"
	             "  %s --decode <model.sslm> <tokenizer.sslm>\n",
	             argv0, argv0, argv0);
}

// ---------------------------------------------------------------------
// Shared model-load + marshal boilerplate, reused verbatim in shape from
// tools/sslm_generate.cpp (Stage 1-3 there). Aborts the process (loud,
// StandardsDocument.md 5.6) on any load/marshal failure -- there is no
// meaningful partial result for this probe if the artifact does not load.
// ---------------------------------------------------------------------
struct LoadedModel {
	std::vector<uint8_t> model_bytes;
	SslmModelView view;
	std::vector<LayerBacking> backings;
	std::vector<LayerWeights> layers;
	const int8_t* embed_weights = nullptr;
	const int8_t* head_weights = nullptr;
	std::vector<int32_t> final_norm_gain;
	CarriedScale embed_site_constant{};
	CarriedScale final_norm_site_constant{};
};

LoadedModel LoadAndMarshal(const std::string& model_path) {
	LoadedModel lm;
	if (!ReadFile(model_path.c_str(), lm.model_bytes)) {
		std::fprintf(stderr, "FATAL stage=model_file_read: could not read \"%s\"\n", model_path.c_str());
		std::exit(1);
	}
	std::string model_err;
	const SslmModelStatus load_status =
	    SslmModel::Load(lm.model_bytes.data(), lm.model_bytes.size(), lm.view, &model_err);
	if (load_status != SslmModelStatus::Ok) {
		std::fprintf(stderr, "FATAL stage=model_load: status=%s diagnostic=\"%s\"\n",
		             SslmModelStatusName(load_status), model_err.c_str());
		std::exit(1);
	}
	std::printf("model loaded: %s option_g_fused_k_landing=%d layers=%u kv_heads=%u\n",
	            model_path.c_str(), lm.view.option_g_fused_k_landing ? 1 : 0,
	            lm.view.config.num_hidden_layers, lm.view.config.num_key_value_heads);
	if (!lm.view.option_g_fused_k_landing) {
		std::fprintf(stderr,
		             "FATAL: model_view.option_g_fused_k_landing == false for \"%s\" -- this artifact's "
		             "header flags bit is not set, so a legacy-path run here would trivially report zero "
		             "gate hits, which is a false pass, not a measurement (T-1902 brief S3). Use the "
		             "flags-patched scratch artifact.\n",
		             model_path.c_str());
		std::exit(1);
	}

	const uint32_t num_heads = lm.view.config.num_attention_heads;
	const uint32_t num_kv_heads = lm.view.config.num_key_value_heads;
	const uint32_t num_hidden_layers = lm.view.config.num_hidden_layers;

	PreflightScanWscFolds(lm.view);
	lm.backings.resize(num_hidden_layers);
	lm.layers.resize(num_hidden_layers);
	for (uint32_t l = 0; l < num_hidden_layers; ++l) {
		std::string marshal_err;
		if (!MarshalLayer(lm.view, l, num_heads, num_kv_heads, lm.backings[l], lm.layers[l], &marshal_err)) {
			std::fprintf(stderr, "FATAL stage=layer_weights_marshal: layer=%u diagnostic=\"%s\"\n", l,
			             marshal_err.c_str());
			std::exit(1);
		}
	}

	const SslmTensorView* embed_w = lm.view.weights.Tensor("embed");
	const SslmTensorView* final_gain_w = lm.view.weights.Tensor("final_norm.gain");
	if (!embed_w || !final_gain_w) {
		std::fprintf(stderr, "FATAL stage=head_marshal: missing embed or final_norm.gain tensor\n");
		std::exit(1);
	}
	lm.final_norm_gain = WidenGainToInt32(*final_gain_w);
	bool ok = true;
	lm.embed_site_constant = ReadCarriedScale(lm.view.composition_constants, "embed", &ok);
	lm.final_norm_site_constant = ReadCarriedScale(lm.view.composition_constants, "final_norm", &ok);
	if (!ok) {
		std::fprintf(stderr, "FATAL stage=head_marshal: missing embed/final_norm site constant\n");
		std::exit(1);
	}
	lm.embed_weights = reinterpret_cast<const int8_t*>(embed_w->data);
	if (lm.view.config.tie_word_embeddings) {
		lm.head_weights = lm.embed_weights;
	} else {
		const SslmTensorView* lm_head_w = lm.view.weights.Tensor("lm_head");
		if (!lm_head_w) {
			std::fprintf(stderr, "FATAL stage=head_marshal: tie_word_embeddings=0, no lm_head tensor\n");
			std::exit(1);
		}
		lm.head_weights = reinterpret_cast<const int8_t*>(lm_head_w->data);
	}
	return lm;
}

// ---------------------------------------------------------------------
// --selftest
// ---------------------------------------------------------------------
int RunSelftest() {
	bool exceeded = false;
	uint64_t sat = 0;
	// T-1900 build log §7's own cited "smallest witness inside [-60,-25)":
	// bc=1, m_a=1, r_t=2147483649, e_a=35, e_t=-59.
	const int64_t raw = LandingRescale(/*branch_code=*/1, /*m_a=*/1, /*r_t=*/2147483649LL,
	                                    /*e_a=*/35, /*e_t=*/-59, &sat, &exceeded);
	std::printf("selftest: LandingRescale(1,1,2147483649,35,-59) raw=%lld saturation_count=%llu "
	            "out_magnitude_exceeded_int64=%d\n",
	            static_cast<long long>(raw), static_cast<unsigned long long>(sat), exceeded ? 1 : 0);
	if (!exceeded) {
		std::fprintf(stderr,
		             "SELFTEST FAILED: expected out_magnitude_exceeded_int64==true on this witness; got "
		             "false. The harness's own detection mechanism is not proven able to report a "
		             "nonzero hit -- a zero count from --encoder/--decode would not be trustworthy. "
		             "Aborting.\n");
		return 1;
	}
	std::printf("SELFTEST PASSED: the harness's counting mechanism (reading "
	            "out_magnitude_exceeded_int64 off a real LandingRescale call) correctly reports true on "
	            "a known-triggering witness. Nothing in the engine or the artifact was modified by this "
	            "mode; it is a pure function call.\n");
	return 0;
}

// ---------------------------------------------------------------------
// --encoder
// ---------------------------------------------------------------------
struct Doc {
	std::string label;
	std::string prompt;
};

bool ReadPromptsTsv(const std::string& path, std::vector<Doc>& out, std::string* err) {
	std::ifstream f(path, std::ios::binary);
	if (!f) {
		*err = "could not open prompts file";
		return false;
	}
	std::string line;
	size_t lineno = 0;
	while (std::getline(f, line)) {
		++lineno;
		if (!line.empty() && line.back() == '\r') line.pop_back();
		if (line.empty()) continue;
		const size_t tab = line.find('\t');
		if (tab == std::string::npos) {
			*err = "line " + std::to_string(lineno) + " has no tab separator";
			return false;
		}
		Doc d;
		d.label = line.substr(0, tab);
		d.prompt = line.substr(tab + 1);
		const std::string placeholder = "<NL>";
		size_t pos = 0;
		while ((pos = d.prompt.find(placeholder, pos)) != std::string::npos) {
			d.prompt.replace(pos, placeholder.size(), "\n");
			pos += 1;
		}
		out.push_back(std::move(d));
	}
	return true;
}

int RunEncoder(const std::string& model_path, const std::string& tok_path, const std::string& prompts_path) {
	std::vector<uint8_t> tok_bytes;
	if (!ReadFile(tok_path.c_str(), tok_bytes)) {
		std::fprintf(stderr, "FATAL stage=tokenizer_file_read: could not read \"%s\"\n", tok_path.c_str());
		return 1;
	}
	SslmArtifact tok_artifact;
	SslmError tok_open_err;
	if (SslmArtifact::OpenFromMemory(tok_bytes.data(), tok_bytes.size(), tok_artifact, &tok_open_err) !=
	    SslmStatus::Ok) {
		std::fprintf(stderr, "FATAL stage=tokenizer_artifact_open: status=%s diagnostic=\"%s\"\n",
		             SslmStatusName(tok_open_err.code), tok_open_err.message.c_str());
		return 1;
	}
	TokenizerView tokenizer;
	std::string tok_err;
	if (!TokenizerView::Open(tok_artifact, tokenizer, &tok_err)) {
		std::fprintf(stderr, "FATAL stage=tokenizer_view_open: diagnostic=\"%s\"\n", tok_err.c_str());
		return 1;
	}

	std::vector<Doc> docs;
	std::string read_err;
	if (!ReadPromptsTsv(prompts_path, docs, &read_err)) {
		std::fprintf(stderr, "FATAL stage=prompts_read: %s\n", read_err.c_str());
		return 1;
	}
	std::printf("encoder corpus: %zu documents from \"%s\"\n", docs.size(), prompts_path.c_str());

	LoadedModel lm = LoadAndMarshal(model_path);
	const uint32_t num_kv_heads = lm.view.config.num_key_value_heads;
	const uint32_t num_hidden_layers = lm.view.config.num_hidden_layers;
	const size_t hidden_size = lm.view.config.hidden_size;
	const int64_t context_cap = static_cast<int64_t>(lm.view.config.context_cap);
	const size_t kv_bytes = static_cast<size_t>(num_hidden_layers) * static_cast<size_t>(context_cap) *
	                        num_kv_heads * lm.view.config.head_dim * 2;

	uint64_t total_positions = 0;
	uint64_t total_layer_steps = 0;
	uint64_t domain_gate_hits = 0;
	uint64_t other_errors = 0;
	size_t docs_completed = 0;
	size_t docs_gate_hit = 0;
	size_t docs_other_error = 0;

	for (const Doc& doc : docs) {
		const std::vector<int32_t> prompt_tokens = tokenizer.Encode(doc.prompt);
		if (prompt_tokens.empty()) {
			std::fprintf(stderr, "label=%s: tokenizer produced zero tokens -- skipped (not a gate hit)\n",
			             doc.label.c_str());
			++other_errors;
			++docs_other_error;
			continue;
		}
		if (static_cast<int64_t>(prompt_tokens.size()) > context_cap) {
			std::fprintf(stderr, "label=%s: %zu tokens > context_cap %lld -- skipped (not a gate hit)\n",
			             doc.label.c_str(), prompt_tokens.size(), static_cast<long long>(context_cap));
			++other_errors;
			++docs_other_error;
			continue;
		}

		std::vector<uint8_t> workspace(kv_bytes);
		std::vector<int8_t> hidden_codes(hidden_size);
		SequenceLayerState seq;
		seq.hidden_codes = hidden_codes.data();

		bool doc_hit_gate = false;
		bool doc_had_other_error = false;
		for (size_t pos = 0; pos < prompt_tokens.size(); ++pos) {
			CarriedScale embed_scale{};
			std::vector<int8_t> embed_codes(hidden_size);
			const SslmForwardStatus est =
			    EmbedEntry(prompt_tokens[pos], static_cast<int32_t>(lm.view.config.vocab_size),
			               lm.embed_weights, hidden_size, lm.embed_site_constant, embed_codes.data(),
			               &embed_scale);
			if (est != SslmForwardStatus::Ok) {
				std::fprintf(stderr, "label=%s position=%zu: EmbedEntry status=%s (not a gate hit)\n",
				             doc.label.c_str(), pos, SslmForwardStatusName(est));
				doc_had_other_error = true;
				++other_errors;
				break;
			}
			std::memcpy(seq.hidden_codes, embed_codes.data(), hidden_size);
			seq.hidden_scale = embed_scale;
			seq.layer_index = 0;

			for (uint32_t step = 0; step < num_hidden_layers; ++step) {
				++total_layer_steps;
				const SslmForwardStatus st = RunLayerLoop(
				    seq, lm.layers.data(), num_hidden_layers, /*layer_budget=*/1, hidden_size,
				    lm.view.config.head_dim, num_kv_heads, lm.view.config.intermediate_size, context_cap,
				    lm.view.rope_tables, workspace.data(), workspace.size(), OptionGKLandingMode::kFused);
				if (st == SslmForwardStatus::OptionGFusedLandingExponentOutOfDomain) {
					std::printf(
					    "GATE HIT label=%s position=%zu layer=%u status=%s\n", doc.label.c_str(), pos,
					    step, SslmForwardStatusName(st));
					++domain_gate_hits;
					doc_hit_gate = true;
					break;
				}
				if (st != SslmForwardStatus::Ok) {
					std::fprintf(stderr,
					             "label=%s position=%zu layer=%u: RunLayerLoop status=%s (not the domain "
					             "gate -- a different, unexpected status)\n",
					             doc.label.c_str(), pos, step, SslmForwardStatusName(st));
					doc_had_other_error = true;
					++other_errors;
					break;
				}
			}
			++total_positions;
			if (doc_hit_gate || doc_had_other_error) break;
		}
		if (doc_hit_gate) {
			++docs_gate_hit;
		} else if (doc_had_other_error) {
			++docs_other_error;
		} else {
			++docs_completed;
		}
	}

	std::printf(
	    "\n--- encoder summary ---\n"
	    "documents_in_population: %zu\n"
	    "documents_completed_clean: %zu\n"
	    "documents_stopped_by_domain_gate: %zu\n"
	    "documents_stopped_by_other_error: %zu\n"
	    "total_positions_processed: %llu\n"
	    "total_layer_steps_executed: %llu\n"
	    "OptionGFusedLandingExponentOutOfDomain_count: %llu\n"
	    "other_error_count: %llu\n",
	    docs.size(), docs_completed, docs_gate_hit, docs_other_error,
	    static_cast<unsigned long long>(total_positions), static_cast<unsigned long long>(total_layer_steps),
	    static_cast<unsigned long long>(domain_gate_hits), static_cast<unsigned long long>(other_errors));
	return 0;
}

// ---------------------------------------------------------------------
// --decode : T-1800's own 32-prompt population, verbatim (system prompt,
// chat template, stop ids, max_new=8 -- tools/t1800_decode_agreement.py).
// ---------------------------------------------------------------------
const char* const kSystemPrompt = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.";

const char* const kPrompts[] = {
    "What is the capital of Italy?",
    "What is the capital of Canada?",
    "What is the capital of Egypt?",
    "Name the smallest planet in the solar system.",
    "Name the longest river in the world.",
    "How many legs does a spider have?",
    "How many continents are there on Earth?",
    "What is 7 plus 9?",
    "What is 14 minus 6?",
    "What is 8 times 6?",
    "What is 100 divided by 4?",
    "What is 23 plus 19?",
    "Write one word that means happy.",
    "Write one word that means fast.",
    "Give me a synonym for the word 'large'.",
    "Name a color that is not red, blue, or green.",
    "Say the days of the week starting from Monday.",
    "List three fruits.",
    "List two animals that live in the ocean.",
    "What language is spoken in Brazil?",
    "What is the freezing point of water in Celsius?",
    "What is the boiling point of water in Celsius?",
    "Who wrote the play Romeo and Juliet?",
    "What gas do plants absorb from the air?",
    "What is the chemical symbol for gold?",
    "How many sides does a hexagon have?",
    "What is the opposite of hot?",
    "What is the opposite of up?",
    "Translate the word 'hello' into French.",
    "Complete the sequence: 2, 4, 6, 8, ...",
    "What year did World War II end?",
    "Name the largest ocean on Earth.",
};
constexpr size_t kNumPrompts = sizeof(kPrompts) / sizeof(kPrompts[0]);
constexpr size_t kMaxNewTokens = 8;
const int32_t kStopIds[] = {151645, 151643};

std::string BuildPrompt(const std::string& question) {
	return std::string("<|im_start|>system\n") + kSystemPrompt + "<|im_end|>\n<|im_start|>user\n" +
	       question + "<|im_end|>\n<|im_start|>assistant\n";
}

int RunDecode(const std::string& model_path, const std::string& tok_path) {
	std::vector<uint8_t> tok_bytes;
	if (!ReadFile(tok_path.c_str(), tok_bytes)) {
		std::fprintf(stderr, "FATAL stage=tokenizer_file_read: could not read \"%s\"\n", tok_path.c_str());
		return 1;
	}
	SslmArtifact tok_artifact;
	SslmError tok_open_err;
	if (SslmArtifact::OpenFromMemory(tok_bytes.data(), tok_bytes.size(), tok_artifact, &tok_open_err) !=
	    SslmStatus::Ok) {
		std::fprintf(stderr, "FATAL stage=tokenizer_artifact_open: status=%s diagnostic=\"%s\"\n",
		             SslmStatusName(tok_open_err.code), tok_open_err.message.c_str());
		return 1;
	}
	TokenizerView tokenizer;
	std::string tok_err;
	if (!TokenizerView::Open(tok_artifact, tokenizer, &tok_err)) {
		std::fprintf(stderr, "FATAL stage=tokenizer_view_open: diagnostic=\"%s\"\n", tok_err.c_str());
		return 1;
	}

	LoadedModel lm = LoadAndMarshal(model_path);
	const uint32_t num_kv_heads = lm.view.config.num_key_value_heads;
	const uint32_t num_hidden_layers = lm.view.config.num_hidden_layers;
	const size_t hidden_size = lm.view.config.hidden_size;
	const int64_t context_cap = static_cast<int64_t>(lm.view.config.context_cap);
	const size_t kv_bytes = static_cast<size_t>(num_hidden_layers) * static_cast<size_t>(context_cap) *
	                        num_kv_heads * lm.view.config.head_dim * 2;
	const size_t vocab_size_z = static_cast<size_t>(lm.view.config.vocab_size);

	uint64_t domain_gate_hits = 0;
	uint64_t other_errors = 0;
	size_t prompts_completed = 0;

	std::printf("decode population: %zu prompts, max_new=%zu\n", kNumPrompts, kMaxNewTokens);

	for (size_t i = 0; i < kNumPrompts; ++i) {
		const std::string full_prompt = BuildPrompt(kPrompts[i]);
		const std::vector<int32_t> prompt_tokens = tokenizer.Encode(full_prompt);
		if (prompt_tokens.empty()) {
			std::fprintf(stderr, "prompt[%zu]: tokenizer produced zero tokens -- skipped (not a gate hit)\n", i);
			++other_errors;
			continue;
		}

		std::vector<uint8_t> workspace(kv_bytes);
		std::vector<int8_t> hidden_codes(hidden_size);
		SequenceLayerState seq;
		seq.hidden_codes = hidden_codes.data();
		std::vector<int32_t> out_tokens(kMaxNewTokens);
		std::vector<int32_t> out_logit_rows(kMaxNewTokens * vocab_size_z);
		size_t tokens_produced = 0;
		SslmDecodeStopReason stop_reason = SslmDecodeStopReason::MaxTokensReached;

		const SslmForwardStatus st = RunGreedyDecodeLoop(
		    seq, lm.layers.data(), num_hidden_layers, hidden_size, lm.view.config.head_dim, num_kv_heads,
		    lm.view.config.intermediate_size, context_cap, lm.view.rope_tables, prompt_tokens.data(),
		    prompt_tokens.size(), lm.embed_weights, lm.embed_site_constant, lm.final_norm_gain.data(),
		    lm.final_norm_site_constant, lm.head_weights, static_cast<int32_t>(lm.view.config.vocab_size),
		    kStopIds, 2, kMaxNewTokens, workspace.data(), workspace.size(), out_tokens.data(),
		    out_logit_rows.data(), out_tokens.size(), &tokens_produced, &stop_reason,
		    lm.view.config.kv_precision, lm.view.option_g_fused_k_landing);

		if (st == SslmForwardStatus::OptionGFusedLandingExponentOutOfDomain) {
			std::printf("GATE HIT prompt[%zu]=\"%s\" status=%s\n", i, kPrompts[i], SslmForwardStatusName(st));
			++domain_gate_hits;
		} else if (st != SslmForwardStatus::Ok) {
			std::fprintf(stderr, "prompt[%zu]=\"%s\": RunGreedyDecodeLoop status=%s (not the domain gate)\n",
			             i, kPrompts[i], SslmForwardStatusName(st));
			++other_errors;
		} else {
			++prompts_completed;
		}
	}

	std::printf(
	    "\n--- decode summary ---\n"
	    "prompts_in_population: %zu\n"
	    "prompts_completed_clean: %zu\n"
	    "OptionGFusedLandingExponentOutOfDomain_count: %llu\n"
	    "other_error_count: %llu\n",
	    kNumPrompts, prompts_completed, static_cast<unsigned long long>(domain_gate_hits),
	    static_cast<unsigned long long>(other_errors));
	return 0;
}

}  // namespace

int main(int argc, char** argv) {
	if (argc < 2) {
		PrintUsage(argv[0]);
		return 2;
	}
	const std::string mode = argv[1];
	if (mode == "--selftest") {
		return RunSelftest();
	} else if (mode == "--encoder") {
		if (argc != 5) {
			PrintUsage(argv[0]);
			return 2;
		}
		return RunEncoder(argv[2], argv[3], argv[4]);
	} else if (mode == "--decode") {
		if (argc != 4) {
			PrintUsage(argv[0]);
			return 2;
		}
		return RunDecode(argv[2], argv[3]);
	}
	PrintUsage(argv[0]);
	return 2;
}
