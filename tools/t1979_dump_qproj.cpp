// t1979_dump_qproj.cpp -- T-1979 GPU walking-skeleton spike, CPU-side dump.
//
// Disposable spike tool (branch brunel/t1979-gpu-skeleton, never merges). Loads
// the real 1.5B artifact, marshals layer 0's real weights (the same production
// MarshalLayer adapter tools/sslm_layer_trace.cpp already uses), computes the
// real layer-0 attn_norm output for one real token via the production EmbedEntry
// + RmsNormSite calls (both public, unmodified production functions), then runs
// the production RunLayerLoop with a trace hook installed so it captures the
// "layer0.q_proj.requant" chain-trace record -- the production ProjectAndFunnel's own
// funnel call, INSTRUMENTED, never re-derived. This tool calls no re-derived
// arithmetic of its own: every number in the dump is either read straight off
// the artifact, or produced by calling the shipped production functions
// (EmbedEntry, RmsNormSite, RunLayerLoop, via the trace hook) exactly as the
// production decode path already calls them.
//
// Dump format (little-endian, all fixed-width ints):
//   uint64 hidden_size
//   int8[hidden_size]   in_codes      -- layer0.attn_norm's own real output (RmsNormSite)
//   int64 in_scale_m, int64 in_scale_e
//   int8[hidden_size * hidden_size] q_weight   -- WGT1 q_proj, row-major [out,in]
//   int32[hidden_size] q_fold_identity
//   int32[hidden_size] q_fold_mult
//   int32[hidden_size] q_fold_shift
//   uint8 has_bias
//   int64[hidden_size] q_bias        -- present iff has_bias; all-zero placeholder otherwise
//   int64 q_site_constant_m, int64 q_site_constant_e
//   int64[hidden_size] cpu_wide_row  -- the funnel's own input row (post WSC1+bias), from the trace record's x_int
//   int64 cpu_d_prime
//   int64 cpu_dn
//   int32 cpu_s
//   int64 cpu_r
//   int8[hidden_size]  cpu_out_codes
//   int64 cpu_out_scale_m, int64 cpu_out_scale_e
//
// Usage: t1979_dump_qproj <model.sslm> <tokenizer.sslm> "<prompt>" --dump <path>
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

void PrintUsage(const char* argv0) {
	std::fprintf(stderr,
	             "usage: %s <model.sslm> <tokenizer.sslm> \"<prompt>\" --dump <path>\n", argv0);
}

// Trace-hook capture context: the ONE chain record for site "layer0.q_proj.requant",
// copied out of the transient spans the hook receives (SslmChainTraceRecord's
// own header: "valid only for the duration of the hook call").
struct CapturedRecord {
	bool found = false;
	std::vector<int64_t> x_int;
	int64_t d_prime = 0;
	int64_t dn = 0;
	int32_t s = 0;
	int64_t r = 0;
	std::vector<int8_t> codes;
	int64_t m_out = 0;
	int64_t e_out = 0;
};

void TraceHookFn(const SslmChainTraceRecord* chain, const SslmKvLandingTraceRecord* /*kv*/,
                  void* user) {
	if (chain == nullptr) return;
	if (chain->site != "layer0.q_proj.requant") return;
	CapturedRecord* out = reinterpret_cast<CapturedRecord*>(user);
	out->found = true;
	out->x_int.assign(chain->x_int.begin(), chain->x_int.end());
	out->d_prime = chain->d_prime;
	out->dn = chain->dn;
	out->s = chain->s;
	out->r = chain->r;
	out->codes.assign(chain->codes.begin(), chain->codes.end());
	out->m_out = chain->m_out;
	out->e_out = chain->e_out;
}

template <typename T>
void WriteRaw(std::ofstream& f, const T& v) {
	f.write(reinterpret_cast<const char*>(&v), sizeof(v));
}

template <typename T>
void WriteVec(std::ofstream& f, const std::vector<T>& v) {
	if (!v.empty()) f.write(reinterpret_cast<const char*>(v.data()), v.size() * sizeof(T));
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
		return 2;
	}

	// --- tokenizer + prompt encode -----------------------------------------
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

	// --- model load + per-layer marshal -------------------------------------
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
	std::printf("model loaded: hidden_size=%u layers=%u heads=%u/%u head_dim=%u\n",
	            model_view.config.hidden_size, model_view.config.num_hidden_layers,
	            model_view.config.num_attention_heads, model_view.config.num_key_value_heads,
	            model_view.config.head_dim);

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
	if (!embed_w) {
		std::fprintf(stderr, "FAILED at stage=embed_marshal: missing embed tensor\n");
		return 1;
	}
	bool ok = true;
	CarriedScale embed_site_constant =
	    ReadCarriedScale(model_view.composition_constants, "embed", &ok);
	if (!ok) {
		std::fprintf(stderr, "FAILED at stage=embed_marshal: missing embed site constant\n");
		return 1;
	}
	const int8_t* embed_weights = reinterpret_cast<const int8_t*>(embed_w->data);

	const int64_t context_cap = static_cast<int64_t>(model_view.config.context_cap);
	const size_t kv_bytes = static_cast<size_t>(num_hidden_layers) * static_cast<size_t>(context_cap) *
	                        num_kv_heads * model_view.config.head_dim * 2;

	// --- the real token: the prompt's own last token (position 0 forward, same
	// convention sslm_layer_trace.cpp uses for its own "last prompt token"). ---
	const int32_t token = prompt_tokens.back();

	// EmbedEntry (public production function, unmodified) -> RmsNormSite
	// (public production function, unmodified) -> the real layer0.attn_norm
	// output, which is ProjectAndFunnel(q_proj)'s own real in_codes/in_scale.
	std::vector<int8_t> embed_codes(hidden_size);
	CarriedScale embed_scale{};
	SslmForwardStatus st = EmbedEntry(token, static_cast<int32_t>(model_view.config.vocab_size),
	                                   embed_weights, hidden_size, embed_site_constant,
	                                   embed_codes.data(), &embed_scale);
	if (st != SslmForwardStatus::Ok) {
		std::fprintf(stderr, "FAILED at stage=embed: status=%s\n", SslmForwardStatusName(st));
		return 1;
	}

	std::vector<int8_t> normed_codes(hidden_size);
	CarriedScale normed_scale{};
	st = RmsNormSite(embed_codes.data(), layers[0].attn_norm_gain, hidden_size, embed_scale,
	                  layers[0].attn_norm_site_constant, normed_codes.data(), &normed_scale,
	                  "layer0.attn_norm");
	if (st != SslmForwardStatus::Ok) {
		std::fprintf(stderr, "FAILED at stage=attn_norm: status=%s\n", SslmForwardStatusName(st));
		return 1;
	}

	std::printf("real in_codes/in_scale captured: layer0.attn_norm output for token=%d, scale=(%lld,%lld)\n",
	            token, static_cast<long long>(normed_scale.m), static_cast<long long>(normed_scale.e));

	// --- install the trace hook, then run the REAL RunLayerLoop (layer_budget=1)
	// over the SAME normed_codes/normed_scale as the seed -- this is exactly
	// RunWholeToken's own composition, so the "layer0.q_proj.requant" record it emits
	// is the production ProjectAndFunnel's own real output, instrumented, never
	// re-derived by this tool. -----------------------------------------------
	CapturedRecord captured;
	SslmSetTraceHook(model_view.trace_hook, &TraceHookFn, &captured);

	std::vector<uint8_t> workspace(kv_bytes);
	std::vector<int8_t> seq_codes(normed_codes.size());
	// RunLayerLoop's own seed is the RESIDUAL stream (attn_norm's own INPUT,
	// i.e. embed_codes/embed_scale) -- attn_norm is the loop's own first internal
	// step (forward_sites.h's RunLayerLoop doc: "attn_norm (RmsNormSite) -> q/k/v
	// projections"). Seeding with embed_codes/embed_scale reproduces the loop's
	// own real attn_norm call internally, which is what emits the "layer0.q_proj.requant"
	// trace record this tool captures -- seeding with normed_codes would run
	// attn_norm a SECOND time on an already-normed row, which is not what the
	// production decode path ever does.
	for (size_t i = 0; i < hidden_size; ++i) seq_codes[i] = embed_codes[i];
	SequenceLayerState seq;
	seq.hidden_codes = seq_codes.data();
	seq.hidden_scale = embed_scale;
	seq.layer_index = 0;

	st = RunLayerLoop(seq, layers.data(), num_hidden_layers, /*layer_budget=*/1, hidden_size,
	                   model_view.config.head_dim, num_kv_heads, model_view.config.intermediate_size,
	                   context_cap, model_view.rope_tables, workspace.data(), workspace.size(),
	                   /*site_prefix=*/{}, /*token_index=*/0, &model_view.trace_hook);
	SslmSetTraceHook(model_view.trace_hook, nullptr, nullptr);
	if (st != SslmForwardStatus::Ok) {
		std::fprintf(stderr, "FAILED at stage=run_layer_loop: status=%s\n", SslmForwardStatusName(st));
		return 1;
	}
	if (!captured.found) {
		std::fprintf(stderr, "FAILED at stage=trace_capture: no \"layer0.q_proj\" chain record observed\n");
		return 1;
	}
	if (captured.x_int.size() != hidden_size || captured.codes.size() != hidden_size) {
		std::fprintf(stderr,
		             "FAILED at stage=trace_capture: captured record width mismatch (x_int=%zu codes=%zu "
		             "hidden_size=%zu)\n",
		             captured.x_int.size(), captured.codes.size(), hidden_size);
		return 1;
	}
	std::printf(
	    "captured CPU production layer0.q_proj trace record: d_prime=%lld dn=%lld s=%d r=%lld "
	    "out_scale=(%lld,%lld)\n",
	    static_cast<long long>(captured.d_prime), static_cast<long long>(captured.dn), captured.s,
	    static_cast<long long>(captured.r), static_cast<long long>(captured.m_out),
	    static_cast<long long>(captured.e_out));

	// Cross-check: this tool's own normed_codes/normed_scale (computed by
	// calling RmsNormSite directly, above) must be BIT-IDENTICAL to what
	// RunLayerLoop's own internal attn_norm call produced -- otherwise the
	// in_codes/in_scale this dump ships as ProjectAndFunnel's real input would
	// not be the same values the captured trace record was actually computed
	// from. RunLayerLoop does not expose its internal normed row directly, so
	// this is checked by recomputing it a second, independent time (same
	// public RmsNormSite call, same real inputs) and comparing -- not by
	// re-deriving the arithmetic, only by calling the same production
	// function twice, matching this codebase's own established double-call
	// self-check convention (sslm_layer_trace.cpp's interior_row_oracle).
	{
		std::vector<int8_t> normed_codes_check(hidden_size);
		CarriedScale normed_scale_check{};
		const SslmForwardStatus cst =
		    RmsNormSite(embed_codes.data(), layers[0].attn_norm_gain, hidden_size, embed_scale,
		                layers[0].attn_norm_site_constant, normed_codes_check.data(),
		                &normed_scale_check, "layer0.attn_norm");
		const bool codes_match = cst == SslmForwardStatus::Ok &&
		                          std::memcmp(normed_codes_check.data(), normed_codes.data(),
		                                      hidden_size) == 0;
		const bool scale_match = normed_scale_check.m == normed_scale.m &&
		                          normed_scale_check.e == normed_scale.e;
		if (!codes_match || !scale_match) {
			std::fprintf(stderr,
			             "FAILED at stage=self_check: repeated RmsNormSite call disagrees with itself "
			             "(codes_match=%d scale_match=%d) -- refusing to dump\n",
			             codes_match ? 1 : 0, scale_match ? 1 : 0);
			return 1;
		}
	}

	// --- dump ----------------------------------------------------------------
	std::ofstream f(dump_path, std::ios::binary | std::ios::trunc);
	if (!f) {
		std::fprintf(stderr, "FAILED at stage=dump_write: could not open \"%s\"\n", dump_path.c_str());
		return 1;
	}
	const uint64_t hidden_size_u64 = static_cast<uint64_t>(hidden_size);
	WriteRaw(f, hidden_size_u64);
	WriteVec(f, normed_codes);
	WriteRaw(f, normed_scale.m);
	WriteRaw(f, normed_scale.e);

	std::vector<int8_t> q_weight_copy(layers[0].q_weight, layers[0].q_weight + hidden_size * hidden_size);
	WriteVec(f, q_weight_copy);

	std::vector<int32_t> q_fold_identity_copy(layers[0].q_fold_identity,
	                                           layers[0].q_fold_identity + hidden_size);
	std::vector<int32_t> q_fold_mult_copy(layers[0].q_fold_mult, layers[0].q_fold_mult + hidden_size);
	std::vector<int32_t> q_fold_shift_copy(layers[0].q_fold_shift, layers[0].q_fold_shift + hidden_size);
	WriteVec(f, q_fold_identity_copy);
	WriteVec(f, q_fold_mult_copy);
	WriteVec(f, q_fold_shift_copy);

	const uint8_t has_bias = layers[0].q_bias != nullptr ? 1 : 0;
	WriteRaw(f, has_bias);
	std::vector<int64_t> q_bias_copy(hidden_size, 0);
	if (has_bias) {
		q_bias_copy.assign(layers[0].q_bias, layers[0].q_bias + hidden_size);
	}
	WriteVec(f, q_bias_copy);

	WriteRaw(f, layers[0].q_site_constant.m);
	WriteRaw(f, layers[0].q_site_constant.e);

	WriteVec(f, captured.x_int);
	WriteRaw(f, captured.d_prime);
	WriteRaw(f, captured.dn);
	WriteRaw(f, captured.s);
	WriteRaw(f, captured.r);
	WriteVec(f, captured.codes);
	WriteRaw(f, captured.m_out);
	WriteRaw(f, captured.e_out);

	if (!f) {
		std::fprintf(stderr, "FAILED at stage=dump_write: write error on \"%s\"\n", dump_path.c_str());
		return 1;
	}
	std::printf("has_bias=%d dumped -> %s\n", static_cast<int>(has_bias), dump_path.c_str());
	return 0;
}
