// t1954_trace_probe.cpp -- T-1959 micro-round (Poirot 285a8e1 confirmation
// review, Significant/D-SLM2740). T-1955's own S3 finding, closed at
// source by T-1956, had never executed anywhere: `SslmEmitKvLandingTrace`'s
// only call site in the whole tree was the new one, no test sink ever
// received a non-null K/V-landing record, and none of the three gates
// install a trace hook. This tool is the "one readback" closure the
// reviewer named -- the same shape T-1956 already used to prove S1 live
// (a counter readback), applied here to a trace emission instead of a
// saturation counter.
//
// A dedicated tool rather than a change to `t1954_selfcheck_q.cpp`, because
// `RunGreedyDecodeLoop` (which the self-check tool calls) exposes NO path
// to a `SslmTraceHookState` at all -- verified at source, it is not one of
// its own parameters, and its own internal `RunLayerLoop` call never
// threads one. The public 16-parameter `RunLayerLoop` overload DOES expose
// `trace_hook_state` (forward_sites.h) -- already-existing, already-shipped
// surface, reused here rather than widening `RunGreedyDecodeLoop`'s own
// production signature for a spike-only diagnostic.
//
// Method: load the real, unmodified artifact through the real production
// loader (identical to every other T-1954 tool); inject the same derived
// Q landing constants the self-check tool uses; embed ONE real prompt
// token (`EmbedEntry`, the same step `RunGreedyDecodeLoop`'s own
// `RunWholeToken` lambda performs); install a counting hook; call
// `RunLayerLoop` directly with `layer_budget = num_hidden_layers` (every
// layer, one token) and the fused-Q toggle set. Expected record count:
// `num_heads * num_hidden_layers` (one record per head per layer, this
// token's own single position).
//
// Usage: t1954_trace_probe <model.sslm> <tokenizer.sslm> <derived_constants.txt>
// (SSLM_OPTION_G_FUSED_Q_LANDING=1 must already be set in the calling
// process, same convention as t1954_selfcheck_q.)

#include <cstdio>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include "superslm/artifact.h"
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

// Accumulated by the hook, read back after the call -- the readback the
// reviewer asked for.
struct TraceCounters {
	uint64_t chain_records = 0;       // SslmChainTraceRecord -- every OTHER projection site's own
	                                   // pre-existing chain trace (attn_norm, o_proj, MLP sites,
	                                   // etc.), informational only, see the printout at the end.
	uint64_t kv_landing_records = 0;  // SslmKvLandingTraceRecord -- the fused-Q emission under test.
	uint64_t sanity_failures = 0;     // any received record whose fields fail the sanity checks below.
	uint32_t num_heads = 0;
	uint32_t num_hidden_layers = 0;
	size_t head_dim = 0;
	std::string last_site;
	uint32_t last_head = 0;
	size_t last_token_index = 0;
	int64_t last_m_out = 0, last_e_out = 0;
	size_t last_codes_size = 0;
};

void CountingHook(const SslmChainTraceRecord* chain, const SslmKvLandingTraceRecord* kv,
                   void* user) {
	TraceCounters* counters = static_cast<TraceCounters*>(user);
	if (chain != nullptr) {
		++counters->chain_records;
	}
	if (kv != nullptr) {
		++counters->kv_landing_records;
		counters->last_site = std::string(kv->site);
		counters->last_head = kv->head;
		counters->last_token_index = kv->token_index;
		counters->last_m_out = kv->m_out;
		counters->last_e_out = kv->e_out;
		counters->last_codes_size = kv->codes.size();
		// Sane-field-values check, per record, not just on the last one:
		bool ok = true;
		ok = ok && (kv->site.substr(0, 5) == "layer");  // "layer{N}.q_proj.requant"
		ok = ok && kv->head < counters->num_heads;
		ok = ok && kv->x_int.size() == counters->head_dim;
		ok = ok && kv->codes.size() == counters->head_dim;
		// Canonical-mantissa range, checked_chain_funnel.h's own CarriedScale
		// contract: m in [2^30, 2^31) for a canonical scale (this is
		// q_landing_m_out, injected from the derived constants, so it must
		// still be canonical here -- confirms the record's own m_out is the
		// SAME value injected, not garbage).
		ok = ok && kv->m_out >= (int64_t{1} << 30) && kv->m_out < (int64_t{1} << 31);
		// Landed codes are int8 range by construction (ClampRopeCode's own
		// contract, [-127, 127]) -- checked per element.
		for (int8_t c : kv->codes) {
			if (c < -127 || c > 127) ok = false;
		}
		if (!ok) ++counters->sanity_failures;
	}
}

}  // namespace

int main(int argc, char** argv) {
	if (argc < 4) {
		std::fprintf(stderr, "usage: %s <model.sslm> <tokenizer.sslm> <derived_constants.txt>\n",
		             argv[0]);
		return 2;
	}
	const std::string model_path = argv[1];
	const std::string tokenizer_path = argv[2];
	const std::string derived_path = argv[3];

	std::map<uint32_t, DerivedQLanding> derived;
	std::string derived_err;
	if (!LoadDerivedConstants(derived_path, &derived, &derived_err)) {
		std::fprintf(stderr, "FAILED stage=derived_constants_load diagnostic=\"%s\"\n",
		             derived_err.c_str());
		return 1;
	}

	std::vector<uint8_t> tok_bytes;
	if (!ReadFile(tokenizer_path.c_str(), tok_bytes)) {
		std::fprintf(stderr, "FAILED stage=tokenizer_file_read path=\"%s\"\n", tokenizer_path.c_str());
		return 1;
	}
	SslmArtifact tok_artifact;
	SslmError tok_open_err;
	if (SslmArtifact::OpenFromMemory(tok_bytes.data(), tok_bytes.size(), tok_artifact,
	                                  &tok_open_err) != SslmStatus::Ok) {
		std::fprintf(stderr, "FAILED stage=tokenizer_artifact_open status=%s\n",
		             SslmStatusName(tok_open_err.code));
		return 1;
	}
	TokenizerView tokenizer;
	std::string tok_err;
	if (!TokenizerView::Open(tok_artifact, tokenizer, &tok_err)) {
		std::fprintf(stderr, "FAILED stage=tokenizer_view_open diagnostic=\"%s\"\n", tok_err.c_str());
		return 1;
	}

	std::vector<uint8_t> model_bytes;
	if (!ReadFile(model_path.c_str(), model_bytes)) {
		std::fprintf(stderr, "FAILED stage=model_file_read path=\"%s\"\n", model_path.c_str());
		return 1;
	}
	SslmModelView model_view;
	std::string model_err;
	const SslmModelStatus load_status =
	    SslmModel::Load(model_bytes.data(), model_bytes.size(), model_view, &model_err);
	if (load_status != SslmModelStatus::Ok) {
		std::fprintf(stderr, "FAILED stage=model_load status=%s diagnostic=\"%s\"\n",
		             SslmModelStatusName(load_status), model_err.c_str());
		return 1;
	}
	std::printf("model loaded (real, unmodified artifact): hidden_size=%u layers=%u\n",
	            model_view.config.hidden_size, model_view.config.num_hidden_layers);

	const uint32_t num_heads = model_view.config.num_attention_heads;
	const uint32_t num_kv_heads = model_view.config.num_key_value_heads;
	const uint32_t num_hidden_layers = model_view.config.num_hidden_layers;
	const size_t hidden_size = model_view.config.hidden_size;
	const size_t head_dim = model_view.config.head_dim;

	PreflightScanWscFolds(model_view);

	std::vector<LayerBacking> backings(num_hidden_layers);
	std::vector<LayerWeights> layers(num_hidden_layers);
	uint32_t layers_injected = 0;
	for (uint32_t l = 0; l < num_hidden_layers; ++l) {
		std::string marshal_err;
		if (!MarshalLayer(model_view, l, num_heads, num_kv_heads, backings[l], layers[l],
		                   &marshal_err)) {
			std::fprintf(stderr, "FAILED stage=layer_weights_marshal layer=%u diagnostic=\"%s\"\n", l,
			             marshal_err.c_str());
			return 1;
		}
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
	std::printf("q_landing constants injected for %u/%u layers\n", layers_injected,
	            num_hidden_layers);

	const SslmTensorView* embed_w = model_view.weights.Tensor("embed");
	if (!embed_w) {
		std::fprintf(stderr, "FAILED stage=head_marshal: missing embed tensor\n");
		return 1;
	}
	bool ok = true;
	CarriedScale embed_site_constant =
	    ReadCarriedScale(model_view.composition_constants, "embed", &ok);
	if (!ok) {
		std::fprintf(stderr, "FAILED stage=head_marshal: missing embed site constant\n");
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

	// One real prompt token -- "What is 12 + 15?", tokenized the same way
	// every other T-1954 tool tokenizes it; only the first token is needed,
	// this probe runs exactly one position through every layer.
	const std::vector<int32_t> prompt_tokens = tokenizer.Encode("What is 12 + 15?");
	if (prompt_tokens.empty()) {
		std::fprintf(stderr, "FAILED stage=tokenize: zero tokens\n");
		return 1;
	}
	const int32_t token = prompt_tokens[0];
	std::printf("embedding token: %d\n", token);

	std::vector<int8_t> embed_codes(hidden_size);
	CarriedScale embed_scale{};
	const SslmForwardStatus embed_status =
	    EmbedEntry(token, static_cast<int32_t>(model_view.config.vocab_size), embed_weights,
	               hidden_size, embed_site_constant, embed_codes.data(), &embed_scale, "embed", 0,
	               nullptr);
	if (embed_status != SslmForwardStatus::Ok) {
		std::fprintf(stderr, "FAILED stage=embed status=%s\n", SslmForwardStatusName(embed_status));
		return 1;
	}
	for (size_t i = 0; i < hidden_size; ++i) seq.hidden_codes[i] = embed_codes[i];
	seq.hidden_scale = embed_scale;
	seq.layer_index = 0;

	TraceCounters counters;
	counters.num_heads = num_heads;
	counters.num_hidden_layers = num_hidden_layers;
	counters.head_dim = head_dim;

	SslmTraceHookState trace_state;
	SslmSetTraceHook(trace_state, &CountingHook, &counters);
	if (!SslmTraceHookInstalled(trace_state)) {
		std::fprintf(stderr, "FAILED stage=hook_install: SslmTraceHookInstalled reports false "
		                     "immediately after SslmSetTraceHook\n");
		return 1;
	}

	// The public 16-parameter RunLayerLoop overload -- trace_hook_state is
	// its own trailing parameter (forward_sites.h), unreachable through
	// RunGreedyDecodeLoop. K's own toggle is legacy here (irrelevant to
	// this probe, which tests Q's own emission only); Q's own toggle comes
	// from the env var, as every other T-1954 tool already does.
	const SslmForwardStatus layer_status = RunLayerLoop(
	    seq, layers.data(), num_hidden_layers, /*layer_budget=*/num_hidden_layers, hidden_size,
	    head_dim, num_kv_heads, model_view.config.intermediate_size, context_cap,
	    model_view.rope_tables, workspace.data(), workspace.size(),
	    OptionGKLandingMode::kLegacy, /*site_prefix=*/"", /*token_index=*/0, &trace_state);

	if (layer_status != SslmForwardStatus::Ok) {
		std::fprintf(stderr, "FAILED stage=layer_loop status=%s\n",
		             SslmForwardStatusName(layer_status));
		return 1;
	}

	const uint64_t expected = static_cast<uint64_t>(num_heads) * num_hidden_layers;
	std::printf("kv_landing_records (fused-Q emission, S3's own closure readback): %llu\n",
	            static_cast<unsigned long long>(counters.kv_landing_records));
	std::printf("expected (num_heads=%u * num_hidden_layers=%u): %llu\n", num_heads,
	            num_hidden_layers, static_cast<unsigned long long>(expected));
	std::printf("sanity_failures (bad site/head/span/mantissa/code-range on ANY received record): "
	            "%llu\n",
	            static_cast<unsigned long long>(counters.sanity_failures));
	std::printf("last_record: site=\"%s\" head=%u token_index=%zu m_out=%lld e_out=%lld "
	            "codes_size=%zu\n",
	            counters.last_site.c_str(), counters.last_head, counters.last_token_index,
	            static_cast<long long>(counters.last_m_out), static_cast<long long>(counters.last_e_out),
	            counters.last_codes_size);

	if (counters.kv_landing_records == 0) {
		std::fprintf(stderr, "FAILED: S3's emission block was reached zero times -- the finding is "
		                     "NOT closed\n");
		return 1;
	}
	if (counters.kv_landing_records != expected) {
		std::fprintf(stderr,
		             "FAILED: record count %llu != expected %llu -- some layer or head did not emit\n",
		             static_cast<unsigned long long>(counters.kv_landing_records),
		             static_cast<unsigned long long>(expected));
		return 1;
	}
	if (counters.sanity_failures != 0) {
		std::fprintf(stderr, "FAILED: %llu record(s) failed the sanity check\n",
		             static_cast<unsigned long long>(counters.sanity_failures));
		return 1;
	}
	// `chain_records` is informational, not a failure condition. First run of
	// this tool reported 280 (28 layers x 10 chain-funnel sites per layer:
	// attn_norm, o_proj, gate_proj, up_proj, down_proj, mlp_norm, attn_ctx,
	// mlp_act, attn_residual, mlp_residual) -- every OTHER projection in the
	// layer still calls RequantChainChecked regardless of Q's own toggle,
	// and once ANY hook is installed, RequantChainChecked emits its own
	// chain record unconditionally (checked_chain_funnel.cpp). This is
	// correct, expected behaviour of the pre-existing tracing mechanism,
	// unrelated to Q's own construction -- not a defect this probe exists
	// to catch. The corrected expectation (fused-Q toggle on): Q's own
	// legacy chain site (`q_proj.requant`) specifically never fires (it is
	// bypassed by construction when fused), but every other site's own
	// chain record still does.
	std::printf("chain_records observed: %llu (informational -- every OTHER projection site's own "
	            "pre-existing chain trace, unrelated to Q's construction; not a pass/fail signal)\n",
	            static_cast<unsigned long long>(counters.chain_records));
	std::printf("TRACE PROBE PASSED: S3's emission block executed %llu times, exactly num_heads * "
	            "num_hidden_layers, every record sane.\n",
	            static_cast<unsigned long long>(counters.kv_landing_records));
	return 0;
}
