// t1960_pooled_trace_batch_q.cpp -- T-1960 encoder-corpus capture instrument for
// T-1822 Sec32.5's incremental contrast (fused-Q spike, toggle-off vs toggle-on),
// built on T-1954's own certified spike (brunel/t1954-fusedq-spike@fca77bb).
//
// This is a MECHANICAL merge of two already-reviewed sources, not a new design:
//   - The capture loop, both self-checks, the dump format, and K's own
//     `OptionGFusedLandingExponentOutOfDomain` counter are VERBATIM from
//     tools/t1942_pooled_trace_batch.cpp (T-1942, itself a port of T-1777's own
//     t1777_pooled_trace_batch.cpp onto the shipped Option-G production loader).
//   - The Q-landing-constant in-memory injection (DerivedQLanding,
//     LoadDerivedConstants, and the injection loop) is VERBATIM from
//     tools/t1954_selfcheck_q.cpp (T-1954's own certified Gate B mechanism) --
//     the shipped artifact carries no `layer{N}.q` key yet (Sec32.2's own unbuilt
//     artifact-writer obligation), so the same spike-tier in-memory substitute
//     T-1954 already built and gated is reused here, unmodified in mechanism.
//
// The ONLY genuinely new logic is a second domain-gate counter tracking Q's own,
// distinct refusal status (SslmForwardStatus::OptionGFusedQLandingExponentOutOfDomain,
// T-1822 Sec32.3 -- never conflated with K's OptionGFusedLandingExponentOutOfDomain)
// at the same three call sites K's counter already watches.
//
// Q's own fused/legacy toggle is NOT a parameter of this tool -- it is read
// internally by forward_sites.cpp via SSLM_OPTION_G_FUSED_Q_LANDING (T-1954
// Sec2, D-SLM2748: read once per RunLayerLoopImpl call, not cached), exactly
// like every other T-1954 tool. Toggle-off vs toggle-on is selected by the
// CALLER setting that env var before invoking this binary -- the artifact and
// this tool's own arguments are identical in both legs.
//
// Usage:
//   t1960_pooled_trace_batch_q <model.sslm> <tokenizer.sslm> <prompts.tsv>
//       <dump_dir> <derived_q_constants.txt>
// <derived_q_constants.txt>: one line per layer, "L m_out e_out e_t r_t"
// (t1954_derive_q_landing.py's own output format, unchanged).

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <sstream>
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
	             "usage: %s <model.sslm> <tokenizer.sslm> <prompts.tsv> <dump_dir> "
	             "<derived_q_constants.txt>\n",
	             argv0);
}

// Verbatim of t1942_pooled_trace_batch.cpp's own Fnv1a64.
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
		{
			const std::string placeholder = "<NL>";
			size_t pos = 0;
			while ((pos = d.prompt.find(placeholder, pos)) != std::string::npos) {
				d.prompt.replace(pos, placeholder.size(), "\n");
				pos += 1;
			}
		}
		out.push_back(std::move(d));
	}
	return true;
}

// --- T-1954's own DerivedQLanding / LoadDerivedConstants, verbatim. ---------
struct DerivedQLanding {
	int64_t m_out, e_out, e_t, r_t;
};

bool LoadDerivedConstants(const std::string& path, std::map<uint32_t, DerivedQLanding>* out,
                           std::string* err) {
	std::ifstream f(path);
	if (!f) {
		*err = "could not open \"" + path + "\"";
		return false;
	}
	std::string line;
	while (std::getline(f, line)) {
		if (line.empty()) continue;
		std::istringstream iss(line);
		uint32_t l;
		DerivedQLanding d{};
		if (!(iss >> l >> d.m_out >> d.e_out >> d.e_t >> d.r_t)) {
			*err = "malformed line: \"" + line + "\"";
			return false;
		}
		(*out)[l] = d;
	}
	return true;
}

// Processes ONE prompt against an already-loaded, already-Q-injected model.
// Identical to t1942_pooled_trace_batch.cpp's own ProcessOneDocument, plus a
// second counter (*domain_gate_hits_q) tracking Q's own, distinct refusal
// status at the same three call sites.
bool ProcessOneDocument(const SslmModelView& model_view, const std::vector<LayerWeights>& layers,
                         TokenizerView& tokenizer, const int8_t* embed_weights,
                         const CarriedScale& embed_site_constant, const int32_t* final_norm_gain,
                         const CarriedScale& final_norm_site_constant, const int8_t* head_weights,
                         size_t kv_bytes, const std::string& label, const std::string& prompt,
                         const std::string& dump_dir, size_t* domain_gate_hits_k,
                         size_t* domain_gate_hits_q) {
	const uint32_t num_heads = model_view.config.num_attention_heads;
	const uint32_t num_kv_heads = model_view.config.num_key_value_heads;
	const uint32_t num_hidden_layers = model_view.config.num_hidden_layers;
	const size_t hidden_size = model_view.config.hidden_size;
	const size_t num_rows = static_cast<size_t>(num_hidden_layers) + 1;
	const int64_t context_cap = static_cast<int64_t>(model_view.config.context_cap);
	const size_t vocab_size_z = static_cast<size_t>(model_view.config.vocab_size);
	const OptionGKLandingMode k_landing_mode =
	    model_view.option_g_fused_k_landing ? OptionGKLandingMode::kFused : OptionGKLandingMode::kLegacy;

	auto NoteStatus = [&](SslmForwardStatus st) {
		if (st == SslmForwardStatus::OptionGFusedLandingExponentOutOfDomain) ++*domain_gate_hits_k;
		if (st == SslmForwardStatus::OptionGFusedQLandingExponentOutOfDomain) ++*domain_gate_hits_q;
	};

	const std::vector<int32_t> prompt_tokens = tokenizer.Encode(prompt);
	if (prompt_tokens.empty()) {
		std::fprintf(stderr, "FAILED label=%s stage=tokenizer_encode: prompt encoded to zero tokens\n",
		             label.c_str());
		return false;
	}
	const size_t n_positions = prompt_tokens.size();
	if (static_cast<int64_t>(n_positions) > context_cap) {
		std::fprintf(stderr, "FAILED label=%s stage=context_check: %zu tokens > context_cap %lld\n",
		             label.c_str(), n_positions, static_cast<long long>(context_cap));
		return false;
	}

	// --- Step A: production path. ---
	std::vector<uint8_t> prod_workspace(kv_bytes);
	std::vector<int8_t> prod_hidden_codes(hidden_size);
	SequenceLayerState prod_seq;
	prod_seq.hidden_codes = prod_hidden_codes.data();
	std::vector<int32_t> prod_out_token(1);
	std::vector<int32_t> prod_out_logit_row(vocab_size_z);
	size_t prod_tokens_produced = 0;
	SslmDecodeStopReason prod_stop_reason = SslmDecodeStopReason::MaxTokensReached;
	const SslmForwardStatus prod_status = RunGreedyDecodeLoop(
	    prod_seq, layers.data(), num_hidden_layers, hidden_size, model_view.config.head_dim, num_kv_heads,
	    model_view.config.intermediate_size, context_cap, model_view.rope_tables, prompt_tokens.data(),
	    prompt_tokens.size(), embed_weights, embed_site_constant, final_norm_gain, final_norm_site_constant,
	    head_weights, static_cast<int32_t>(model_view.config.vocab_size), /*stop_ids=*/nullptr,
	    /*stop_count=*/0, /*max_new_tokens=*/1, prod_workspace.data(), prod_workspace.size(),
	    prod_out_token.data(), prod_out_logit_row.data(), prod_out_token.size(), &prod_tokens_produced,
	    &prod_stop_reason, model_view.config.kv_precision, model_view.option_g_fused_k_landing);
	NoteStatus(prod_status);
	if (prod_status != SslmForwardStatus::Ok || prod_tokens_produced != 1) {
		std::fprintf(stderr, "FAILED label=%s stage=production_decode: status=%s produced=%zu\n",
		             label.c_str(), SslmForwardStatusName(prod_status), prod_tokens_produced);
		return false;
	}

	auto EmbedWholeTokenInto = [&](SequenceLayerState& seq, int32_t token) -> SslmForwardStatus {
		std::vector<int8_t> embed_codes(hidden_size);
		CarriedScale embed_scale{};
		const SslmForwardStatus est = EmbedEntry(token, static_cast<int32_t>(model_view.config.vocab_size),
		                                          embed_weights, hidden_size, embed_site_constant,
		                                          embed_codes.data(), &embed_scale);
		if (est != SslmForwardStatus::Ok) return est;
		for (size_t i = 0; i < hidden_size; ++i) seq.hidden_codes[i] = embed_codes[i];
		seq.hidden_scale = embed_scale;
		seq.layer_index = 0;
		return SslmForwardStatus::Ok;
	};

	// --- Step B: all-position capture. ---
	std::vector<uint8_t> all_workspace(kv_bytes);
	std::vector<int8_t> all_hidden_codes(hidden_size);
	SequenceLayerState all_seq;
	all_seq.hidden_codes = all_hidden_codes.data();
	std::vector<std::vector<RowSnapshot>> all_rows(n_positions);
	const size_t equiv_position = (n_positions >= 2) ? (n_positions / 2) : 0;
	RowSnapshot equiv_final_from_resumed{};

	for (size_t pos = 0; pos < n_positions; ++pos) {
		SslmForwardStatus est = EmbedWholeTokenInto(all_seq, prompt_tokens[pos]);
		if (est != SslmForwardStatus::Ok) {
			std::fprintf(stderr, "FAILED label=%s stage=capture_embed: position=%zu status=%s\n",
			             label.c_str(), pos, SslmForwardStatusName(est));
			return false;
		}
		all_rows[pos].reserve(num_rows);
		all_rows[pos].push_back(RowSnapshot{
		    std::vector<int8_t>(all_seq.hidden_codes, all_seq.hidden_codes + hidden_size),
		    all_seq.hidden_scale.m, all_seq.hidden_scale.e});
		for (uint32_t step = 0; step < num_hidden_layers; ++step) {
			const SslmForwardStatus st = RunLayerLoop(
			    all_seq, layers.data(), num_hidden_layers, /*layer_budget=*/1, hidden_size,
			    model_view.config.head_dim, num_kv_heads, model_view.config.intermediate_size, context_cap,
			    model_view.rope_tables, all_workspace.data(), all_workspace.size(), k_landing_mode);
			NoteStatus(st);
			if (st != SslmForwardStatus::Ok) {
				std::fprintf(stderr, "FAILED label=%s stage=capture_layer_step: position=%zu layer=%u status=%s\n",
				             label.c_str(), pos, step, SslmForwardStatusName(st));
				return false;
			}
			all_rows[pos].push_back(RowSnapshot{
			    std::vector<int8_t>(all_seq.hidden_codes, all_seq.hidden_codes + hidden_size),
			    all_seq.hidden_scale.m, all_seq.hidden_scale.e});
		}
		if (pos == equiv_position) equiv_final_from_resumed = all_rows[pos].back();
	}

	// --- Self-check 1: last position vs. production, bit-for-bit. ----------
	{
		const RowSnapshot& last_row = all_rows[n_positions - 1].back();
		std::vector<int8_t> final_codes(hidden_size);
		CarriedScale final_scale{};
		SslmForwardStatus st =
		    RmsNormSite(last_row.codes.data(), final_norm_gain, hidden_size, CarriedScale{last_row.m, last_row.e},
		                final_norm_site_constant, final_codes.data(), &final_scale, "final_norm");
		if (st != SslmForwardStatus::Ok) {
			std::fprintf(stderr, "FAILED label=%s stage=self_check_final_norm: status=%s\n", label.c_str(),
			             SslmForwardStatusName(st));
			return false;
		}
		std::vector<int64_t> wide_logits(vocab_size_z);
		std::vector<int32_t> logit_row(vocab_size_z);
		st = LogitsSite(final_codes.data(), hidden_size, head_weights,
		                 static_cast<int32_t>(model_view.config.vocab_size), wide_logits.data(), logit_row.data());
		if (st != SslmForwardStatus::Ok) {
			std::fprintf(stderr, "FAILED label=%s stage=self_check_logits: status=%s\n", label.c_str(),
			             SslmForwardStatusName(st));
			return false;
		}
		const int32_t trace_token = ArgmaxLowestIndexTieBreak(logit_row.data(), vocab_size_z);
		const bool token_match = (trace_token == prod_out_token[0]);
		const bool logits_match =
		    (std::memcmp(logit_row.data(), prod_out_logit_row.data(), vocab_size_z * sizeof(int32_t)) == 0);
		if (!token_match || !logits_match) {
			std::fprintf(stderr,
			             "FAILED label=%s stage=self_check_1: production/trace disagree at last position "
			             "token_match=%d logits_match=%d (no dump written)\n",
			             label.c_str(), token_match ? 1 : 0, logits_match ? 1 : 0);
			return false;
		}
	}

	// --- Self-check 2: resumed-vs-one-shot equivalence at interior position. --
	{
		std::vector<uint8_t> oneshot_workspace(kv_bytes);
		std::vector<int8_t> oneshot_hidden_codes(hidden_size);
		SequenceLayerState oneshot_seq;
		oneshot_seq.hidden_codes = oneshot_hidden_codes.data();
		for (size_t pos = 0; pos <= equiv_position; ++pos) {
			SslmForwardStatus est = EmbedWholeTokenInto(oneshot_seq, prompt_tokens[pos]);
			if (est != SslmForwardStatus::Ok) {
				std::fprintf(stderr, "FAILED label=%s stage=equiv_embed: position=%zu status=%s\n", label.c_str(),
				             pos, SslmForwardStatusName(est));
				return false;
			}
			const SslmForwardStatus st = RunLayerLoop(
			    oneshot_seq, layers.data(), num_hidden_layers, /*layer_budget=*/num_hidden_layers, hidden_size,
			    model_view.config.head_dim, num_kv_heads, model_view.config.intermediate_size, context_cap,
			    model_view.rope_tables, oneshot_workspace.data(), oneshot_workspace.size(), k_landing_mode);
			NoteStatus(st);
			if (st != SslmForwardStatus::Ok) {
				std::fprintf(stderr, "FAILED label=%s stage=equiv_layers: position=%zu status=%s\n", label.c_str(),
				             pos, SslmForwardStatusName(st));
				return false;
			}
		}
		const bool codes_match =
		    std::memcmp(oneshot_seq.hidden_codes, equiv_final_from_resumed.codes.data(), hidden_size) == 0;
		const bool scale_match = oneshot_seq.hidden_scale.m == equiv_final_from_resumed.m &&
		                          oneshot_seq.hidden_scale.e == equiv_final_from_resumed.e;
		if (!codes_match || !scale_match) {
			std::fprintf(stderr,
			             "FAILED label=%s stage=self_check_2: resumed/one-shot disagree at interior "
			             "position=%zu codes_match=%d scale_match=%d (no dump written)\n",
			             label.c_str(), equiv_position, codes_match ? 1 : 0, scale_match ? 1 : 0);
			return false;
		}
	}

	// --- Dump (identical binary layout to t1942_pooled_trace_batch.cpp's own). ---
	const uint64_t fingerprint = Fnv1a64(prompt);
	const std::string dump_path = dump_dir + "/" + label + ".int8.bin";
	std::ofstream f(dump_path, std::ios::binary | std::ios::trunc);
	if (!f) {
		std::fprintf(stderr, "FAILED label=%s stage=dump_write: could not open \"%s\"\n", label.c_str(),
		             dump_path.c_str());
		return false;
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
		std::fprintf(stderr, "FAILED label=%s stage=dump_write: write error on \"%s\"\n", label.c_str(),
		             dump_path.c_str());
		return false;
	}
	std::printf("ok label=%s positions=%zu rows=%zu fingerprint=0x%016llX -> %s\n", label.c_str(), n_positions,
	            num_rows, static_cast<unsigned long long>(fingerprint), dump_path.c_str());
	return true;
}

}  // namespace

int main(int argc, char** argv) {
	if (argc != 6) {
		PrintUsage(argv[0]);
		return 2;
	}
	const std::string model_path = argv[1];
	const std::string tokenizer_path = argv[2];
	const std::string prompts_tsv = argv[3];
	const std::string dump_dir = argv[4];
	const std::string derived_path = argv[5];

	std::map<uint32_t, DerivedQLanding> derived;
	std::string derived_err;
	if (!LoadDerivedConstants(derived_path, &derived, &derived_err)) {
		std::fprintf(stderr, "FAILED stage=derived_constants_load diagnostic=\"%s\"\n", derived_err.c_str());
		return 1;
	}

	std::vector<Doc> docs;
	std::string list_err;
	if (!ReadPromptsTsv(prompts_tsv, docs, &list_err)) {
		std::fprintf(stderr, "FAILED stage=prompts_read: %s\n", list_err.c_str());
		return 1;
	}
	if (docs.empty()) {
		std::fprintf(stderr, "FAILED stage=prompts_read: no documents in \"%s\"\n", prompts_tsv.c_str());
		return 1;
	}

	std::vector<uint8_t> tok_bytes;
	if (!ReadFile(tokenizer_path.c_str(), tok_bytes)) {
		std::fprintf(stderr, "FAILED stage=tokenizer_file_read: could not read \"%s\"\n", tokenizer_path.c_str());
		return 1;
	}
	SslmArtifact tok_artifact;
	SslmError tok_open_err;
	if (SslmArtifact::OpenFromMemory(tok_bytes.data(), tok_bytes.size(), tok_artifact, &tok_open_err) !=
	    SslmStatus::Ok) {
		std::fprintf(stderr, "FAILED stage=tokenizer_artifact_open: status=%s diagnostic=\"%s\"\n",
		             SslmStatusName(tok_open_err.code), tok_open_err.message.c_str());
		return 1;
	}
	TokenizerView tokenizer;
	std::string tok_err;
	if (!TokenizerView::Open(tok_artifact, tokenizer, &tok_err)) {
		std::fprintf(stderr, "FAILED stage=tokenizer_view_open: diagnostic=\"%s\"\n", tok_err.c_str());
		return 1;
	}

	std::vector<uint8_t> model_bytes;
	if (!ReadFile(model_path.c_str(), model_bytes)) {
		std::fprintf(stderr, "FAILED stage=model_file_read: could not read \"%s\"\n", model_path.c_str());
		return 1;
	}
	SslmModelView model_view;
	std::string model_err;
	const SslmModelStatus load_status =
	    SslmModel::Load(model_bytes.data(), model_bytes.size(), model_view, &model_err);
	if (load_status != SslmModelStatus::Ok) {
		std::fprintf(stderr, "FAILED stage=model_load: status=%s diagnostic=\"%s\"\n",
		             SslmModelStatusName(load_status), model_err.c_str());
		return 1;
	}
	const uint32_t num_heads = model_view.config.num_attention_heads;
	const uint32_t num_kv_heads = model_view.config.num_key_value_heads;
	const uint32_t num_hidden_layers = model_view.config.num_hidden_layers;
	const size_t hidden_size = model_view.config.hidden_size;

	PreflightScanWscFolds(model_view);

	std::vector<LayerBacking> backings(num_hidden_layers);
	std::vector<LayerWeights> layers(num_hidden_layers);
	uint32_t layers_injected = 0;
	for (uint32_t l = 0; l < num_hidden_layers; ++l) {
		std::string marshal_err;
		if (!MarshalLayer(model_view, l, num_heads, num_kv_heads, backings[l], layers[l], &marshal_err)) {
			std::fprintf(stderr, "FAILED stage=layer_weights_marshal: layer=%u diagnostic=\"%s\"\n", l,
			             marshal_err.c_str());
			return 1;
		}
		// T-1954's own injection, verbatim mechanism: the ONLY in-memory
		// mutation performed on top of the real production marshal. Every
		// other field MarshalLayer populated is untouched.
		const auto it = derived.find(l);
		if (it == derived.end()) {
			std::fprintf(stderr, "FAILED stage=derived_constants_lookup: no entry for layer=%u\n", l);
			return 1;
		}
		layers[l].q_landing_m_out = it->second.m_out;
		layers[l].q_landing_e_out = it->second.e_out;
		layers[l].q_landing_e_t = it->second.e_t;
		layers[l].q_landing_r_t = it->second.r_t;
		++layers_injected;
	}

	const SslmTensorView* embed_w = model_view.weights.Tensor("embed");
	const SslmTensorView* final_gain_w = model_view.weights.Tensor("final_norm.gain");
	if (!embed_w || !final_gain_w) {
		std::fprintf(stderr, "FAILED stage=head_marshal: missing embed or final_norm.gain tensor\n");
		return 1;
	}
	std::vector<int32_t> final_norm_gain = WidenGainToInt32(*final_gain_w);
	bool ok = true;
	CarriedScale embed_site_constant = ReadCarriedScale(model_view.composition_constants, "embed", &ok);
	CarriedScale final_norm_site_constant = ReadCarriedScale(model_view.composition_constants, "final_norm", &ok);
	if (!ok) {
		std::fprintf(stderr, "FAILED stage=head_marshal: missing embed/final_norm site constant\n");
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
			             "FAILED stage=head_marshal: tie_word_embeddings=0 but no \"lm_head\" WGT1 tensor\n");
			return 1;
		}
		head_weights = reinterpret_cast<const int8_t*>(lm_head_w->data);
	}

	const int64_t context_cap = static_cast<int64_t>(model_view.config.context_cap);
	const size_t kv_bytes = static_cast<size_t>(num_hidden_layers) * static_cast<size_t>(context_cap) *
	                        num_kv_heads * model_view.config.head_dim * 2;

	// T-1968 fix round (Poirot c131eab review, Significant 2/D-SLM2798):
	// same tristate-readback fix as t1960_decode_q.cpp -- see that file's
	// own comment at this exact edit for the full rationale.
	const char* q_toggle_env = std::getenv("SSLM_OPTION_G_FUSED_Q_LANDING");
	const std::string q_toggle_arm_name =
	    (q_toggle_env == nullptr || q_toggle_env[0] == '\0' ||
	     (q_toggle_env[0] == '0' && q_toggle_env[1] == '\0'))
	        ? "legacy"
	    : (std::string(q_toggle_env) == "dynamic") ? "dynamic-fused"
	                                                : "static-fused";
	std::printf(
	    "model loaded once: %s option_g_fused_k_landing=%d "
	    "SSLM_OPTION_G_FUSED_Q_LANDING(read-back)=%s (raw=\"%s\") "
	    "q_landing_injected=%u/%u hidden_size=%zu layers=%u documents=%zu\n",
	    model_path.c_str(), model_view.option_g_fused_k_landing ? 1 : 0, q_toggle_arm_name.c_str(),
	    q_toggle_env ? q_toggle_env : "(unset)", layers_injected, num_hidden_layers, hidden_size,
	    num_hidden_layers, docs.size());

	size_t n_ok = 0, n_failed = 0, domain_gate_hits_k = 0, domain_gate_hits_q = 0;
	for (const Doc& d : docs) {
		const bool ok_doc =
		    ProcessOneDocument(model_view, layers, tokenizer, embed_weights, embed_site_constant,
		                       final_norm_gain.data(), final_norm_site_constant, head_weights, kv_bytes, d.label,
		                       d.prompt, dump_dir, &domain_gate_hits_k, &domain_gate_hits_q);
		if (ok_doc) {
			++n_ok;
		} else {
			++n_failed;
		}
	}
	std::printf("batch_done: %zu ok, %zu failed (of %zu)\n", n_ok, n_failed, docs.size());
	std::printf("OptionGFusedLandingExponentOutOfDomain_count(K): %zu\n", domain_gate_hits_k);
	std::printf("OptionGFusedQLandingExponentOutOfDomain_count(Q): %zu\n", domain_gate_hits_q);
	return (n_failed == 0) ? 0 : 1;
}
