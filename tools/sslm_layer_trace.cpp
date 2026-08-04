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
// Trailing section (T-1689, source-attribution design
// Claude/Vitruvius/superslm-t1683-source-attribution-design-2026-08-02.md
// S5): appended AFTER the existing 29-row body so every existing consumer of
// this dump format keeps working unmodified (append-only convention) --
// uint64 num_hidden_layers, then num_hidden_layers uint64 values: the
// PER-LAYER, CUMULATIVE-COUNTER DELTA of SequenceLayerState::
// kv_saturation_count (forward_sites.h) across the last prompt token's own
// per-layer walk. Snapshot points: once after the embedding row is captured
// (before the last token's own 28-layer walk begins -- prefill's own
// saturation events, from every earlier prompt token, are excluded by this
// baseline) and once after each RunLayerLoop(layer_budget=1) call below;
// deltas[i] = snapshot[i+1] - snapshot[i], the count of NEW saturation
// events (K and V landing writes combined, every head) at layer i+1
// specifically. No production code changes: kv_saturation_count is a public
// field RunLayerLoop already writes (design S2.1); this tool only reads it.
//
// Second trailing section (T-1691, source-attribution design S7 step 4):
// appended AFTER T-1689's own saturation-delta trailer -- uint64
// num_target_rows, then num_target_rows uint64 values (the DUMP ROW NUMBER
// each captured block belongs to -- row N is the state after RunLayerLoop
// has committed 0-indexed layer N-1, the same `layer` parameter KeyRow/
// ValueRow take at that row), then uint64 context_length (the number of
// PRIOR positions captured, 0..context_length-1 -- never the current,
// freshly-landed position at that row, which the parity shadow computes
// itself, S7 step 5), uint64 num_key_value_heads, uint64 head_dim, then
// num_target_rows * context_length * num_key_value_heads * head_dim * 2
// int8 codes: per captured row, per position (0..context_length-1), a
// num_key_value_heads*head_dim K block immediately followed by a
// num_key_value_heads*head_dim V block. Target rows are the design's own
// band -- {5, 21, 22, ..., 28} -- and a geometry with fewer than 28 layers
// (this file's own smaller test fixtures) simply never reaches the rows
// past its own num_hidden_layers, so num_target_rows can be less than 9;
// no row is ever skipped independently of that bound. No production code
// changes: KeyRow/ValueRow are already public accessors RunLayerLoop's own
// landing write and attention interior already use (design S7 step 4).
//
// On a self-check mismatch (production vs. manual-replay token id or logit
// row), this tool exits non-zero with a loud diagnostic and writes no dump --
// matching this codebase's existing fail-loud convention
// (sslm_generate.cpp's own "FAILED at stage=..." lines).
//
// Third, OPTIONAL trailing capability (T-1691, build-sequencing session
// 2026-08-03, Brunel): `--site-dump <path>`, a SEPARATE output file (never
// appended to the dump above -- the existing 29-row body and both existing
// trailers are byte-for-byte unaffected by this flag, present or absent).
// Installs a trace hook (include/superslm/trace_hook.h's own
// SslmSetTraceHook/SslmTraceHookState -- an EXISTING public seam every
// RequantChainChecked call site already threads a `trace_hook_state`
// parameter to and emits a record through, unconditionally, when a hook is
// installed; this tool is the first caller to actually install one). No
// production code changes: this is a tool-side caller of an
// already-shipped, already-wired public API
// (src/forward/checked_chain_funnel.cpp:341-352 already builds and emits
// the SAME record this reads). Captures every SslmChainTraceRecord (site
// name, the wide row AS READ AT THE SITE -- the site's own real input --
// and the site's own real output codes) for the dump-row band this file's
// header now also documents: all 28 rows (widened 2026-08-04, Brunel, for
// the onset/amplification pass -- originally {5, 26, 27, 28}, the layer-5
// control plus the design's own divergence band, T-1691's build-sequencing
// scope from the earlier session -- narrower than the KV-trailer's own
// {5,21..28}, which this flag does not touch or share state with). Filtering is done
// by TOGGLING hook installation around each target row's own
// RunLayerLoop(1) call -- SslmSetTraceHook(state, nullptr, nullptr) between
// target rows -- rather than by inspecting the record's own site string,
// so a record can never leak from a non-target layer's own call even if a
// site name were ever reused across layers.
//
// Fourth, OPTIONAL trailing capability (T-1691, RoPE site-comparison session,
// Brunel, D-SLM749): the site-dump's own chain-record section now also
// carries "rope_apply.q.hN"/"rope_apply.k.hN" records -- RopeApplySite gained
// a trace-hook parameter (the D-SLM749 production-code change) and emits
// through the SAME SslmChainTraceRecord/SslmEmitChainTrace seam every other
// site already uses, so this tool's existing SiteCaptureHook picks them up
// with no change of its own. What DOES need a change here: the K/V-landing
// sibling record type, SslmKvLandingTraceRecord (trace_hook.h) -- RunLayerLoop
// now emits one per head per token for both k_proj and v_proj's own
// PRE-ROTATION landed value (the other half of D-SLM749), through
// SslmEmitKvLandingTrace, which this file's hook previously discarded
// (`const SslmKvLandingTraceRecord* /*kv*/`). The site-dump file gains a
// SECOND section, appended after the existing chain-record section (append-
// only, matching the existing chain/T-1689/T-1691-trailer convention): uint64
// num_kv_records, then per record: the same name/x_int/codes shape as a chain
// record, plus uint32 head, int64 m_in, int64 e_in (trace_hook.h's own
// SslmKvLandingTraceRecord fields) in place of d_prime/dn/s/r/m_out/e_out --
// no code changed in the existing chain-record section's own read/write path.
#include <cstdint>
#include <cstdio>
#include <cstdlib>
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

// --- T-1691 site-dump capture (Brunel, build-sequencing session, per-site
// input/output for the gating step this session's own commission opened
// with) -------------------------------------------------------------------

// One captured SslmChainTraceRecord, copied out of the hook call's own
// non-owning spans -- the record's own header comment states those spans
// are "valid only for the duration of the hook call that carries them; a
// hook that needs the data after returning copies it." This struct is that
// copy.
struct CapturedSite {
	std::string site;
	std::vector<int64_t> x_int;  // the wide row, as read at the site (input)
	std::vector<int8_t> codes;   // the site's own output codes
	int64_t d_prime = 0;
	int64_t dn = 0;
	int32_t s = 0;
	int64_t r = 0;
	int64_t m_out = 0;
	int64_t e_out = 0;
};

// The K/V-landing sibling of CapturedSite (D-SLM749, T-1691 RoPE site-
// comparison session): one captured SslmKvLandingTraceRecord, copied out of
// the hook call's own non-owning spans for the same "valid only for the
// duration of the hook call" reason CapturedSite exists.
struct CapturedKvLanding {
	std::string site;
	uint32_t head = 0;
	std::vector<int64_t> x_int;  // the head's folded segment (kacc/vacc), as landed
	int64_t m_in = 0;
	int64_t e_in = 0;
	std::vector<int8_t> codes;   // the pre-rotation landed codes
};

struct SiteCaptureContext {
	bool active = false;  // toggled by the caller around each target row's own call
	std::vector<CapturedSite> records;
	std::vector<CapturedKvLanding> kv_records;
};

// The hook callback (SslmTraceHookFn's own signature, trace_hook.h). Only
// copies a record when `active` -- set by the caller immediately before and
// after each target row's own RunLayerLoop(1) call, so a record from a
// non-target layer is never appended even though the hook stays installed
// throughout (installation is a per-tool-run constant; activity is
// per-layer). Read-only with respect to the record: no field is written
// back, matching trace_hook.h's own "read-only by construction" contract.
// `chain`/`kv` are never both non-null (trace_hook.h's own "exactly one of
// the two record pointers non-null per call" contract) -- the two branches
// below are not mutually exclusive by accident, they mirror that contract.
void SiteCaptureHook(const SslmChainTraceRecord* chain, const SslmKvLandingTraceRecord* kv,
                      void* user) {
	auto* ctx = static_cast<SiteCaptureContext*>(user);
	if (!ctx->active) return;
	if (chain != nullptr) {
		CapturedSite rec;
		rec.site = std::string(chain->site);
		rec.x_int.assign(chain->x_int.begin(), chain->x_int.end());
		rec.codes.assign(chain->codes.begin(), chain->codes.end());
		rec.d_prime = chain->d_prime;
		rec.dn = chain->dn;
		rec.s = chain->s;
		rec.r = chain->r;
		rec.m_out = chain->m_out;
		rec.e_out = chain->e_out;
		ctx->records.push_back(std::move(rec));
	}
	if (kv != nullptr) {
		CapturedKvLanding rec;
		rec.site = std::string(kv->site);
		rec.head = kv->head;
		rec.x_int.assign(kv->x_int.begin(), kv->x_int.end());
		rec.m_in = kv->m_in;
		rec.e_in = kv->e_in;
		rec.codes.assign(kv->codes.begin(), kv->codes.end());
		ctx->kv_records.push_back(std::move(rec));
	}
}

// Format: uint64 num_records, then per record: uint64 site_name_len, the
// name's own bytes, uint64 n_x (x_int's own length), n_x int64 x_int
// values, uint64 n_codes (codes' own length -- not assumed equal to n_x;
// a projection site's own output width differs from its input width),
// n_codes int8 codes, then int64 d_prime, int64 dn, int32 s, int64 r,
// int64 m_out, int64 e_out.
//
// SECOND section (D-SLM749, T-1691 RoPE site-comparison session), appended
// after the section above -- append-only, the section above is byte-for-byte
// unchanged whether or not any KV-landing record exists: uint64
// num_kv_records, then per record: uint64 site_name_len, the name's own
// bytes, uint32 head, uint64 n_x, n_x int64 x_int values, int64 m_in, int64
// e_in, uint64 n_codes, n_codes int8 codes. A self-contained file, never
// appended to or read alongside the dump above.
bool WriteSiteDump(const char* path, const std::vector<CapturedSite>& records,
                    const std::vector<CapturedKvLanding>& kv_records) {
	std::ofstream f(path, std::ios::binary | std::ios::trunc);
	if (!f) return false;
	const uint64_t n = static_cast<uint64_t>(records.size());
	f.write(reinterpret_cast<const char*>(&n), sizeof(n));
	for (const CapturedSite& rec : records) {
		const uint64_t name_len = static_cast<uint64_t>(rec.site.size());
		f.write(reinterpret_cast<const char*>(&name_len), sizeof(name_len));
		f.write(rec.site.data(), static_cast<std::streamsize>(name_len));
		const uint64_t n_x = static_cast<uint64_t>(rec.x_int.size());
		f.write(reinterpret_cast<const char*>(&n_x), sizeof(n_x));
		f.write(reinterpret_cast<const char*>(rec.x_int.data()),
		        static_cast<std::streamsize>(n_x * sizeof(int64_t)));
		const uint64_t n_codes = static_cast<uint64_t>(rec.codes.size());
		f.write(reinterpret_cast<const char*>(&n_codes), sizeof(n_codes));
		f.write(reinterpret_cast<const char*>(rec.codes.data()),
		        static_cast<std::streamsize>(n_codes));
		f.write(reinterpret_cast<const char*>(&rec.d_prime), sizeof(rec.d_prime));
		f.write(reinterpret_cast<const char*>(&rec.dn), sizeof(rec.dn));
		f.write(reinterpret_cast<const char*>(&rec.s), sizeof(rec.s));
		f.write(reinterpret_cast<const char*>(&rec.r), sizeof(rec.r));
		f.write(reinterpret_cast<const char*>(&rec.m_out), sizeof(rec.m_out));
		f.write(reinterpret_cast<const char*>(&rec.e_out), sizeof(rec.e_out));
	}
	const uint64_t n_kv = static_cast<uint64_t>(kv_records.size());
	f.write(reinterpret_cast<const char*>(&n_kv), sizeof(n_kv));
	for (const CapturedKvLanding& rec : kv_records) {
		const uint64_t name_len = static_cast<uint64_t>(rec.site.size());
		f.write(reinterpret_cast<const char*>(&name_len), sizeof(name_len));
		f.write(rec.site.data(), static_cast<std::streamsize>(name_len));
		f.write(reinterpret_cast<const char*>(&rec.head), sizeof(rec.head));
		const uint64_t n_x = static_cast<uint64_t>(rec.x_int.size());
		f.write(reinterpret_cast<const char*>(&n_x), sizeof(n_x));
		f.write(reinterpret_cast<const char*>(rec.x_int.data()),
		        static_cast<std::streamsize>(n_x * sizeof(int64_t)));
		f.write(reinterpret_cast<const char*>(&rec.m_in), sizeof(rec.m_in));
		f.write(reinterpret_cast<const char*>(&rec.e_in), sizeof(rec.e_in));
		const uint64_t n_codes = static_cast<uint64_t>(rec.codes.size());
		f.write(reinterpret_cast<const char*>(&n_codes), sizeof(n_codes));
		f.write(reinterpret_cast<const char*>(rec.codes.data()),
		        static_cast<std::streamsize>(n_codes));
	}
	return static_cast<bool>(f);
}

bool WriteDump(const char* path, const std::vector<LayerSnapshot>& rows, uint64_t hidden_size,
               uint64_t prompt_fingerprint, const std::vector<uint64_t>& saturation_deltas,
               const std::vector<uint32_t>& kv_trailer_rows, uint64_t kv_trailer_context_length,
               uint64_t kv_trailer_num_kv_heads, uint64_t kv_trailer_head_dim,
               const std::vector<int8_t>& kv_trailer_codes) {
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
	// T-1689 trailer (append-only, S5/this file's header comment): the
	// existing 29-row body above is byte-identical to what every prior
	// consumer of this format already reads; this section is new bytes
	// AFTER it, never a change to the body's own layout.
	const uint64_t num_layers = static_cast<uint64_t>(saturation_deltas.size());
	f.write(reinterpret_cast<const char*>(&num_layers), sizeof(num_layers));
	f.write(reinterpret_cast<const char*>(saturation_deltas.data()),
	        static_cast<std::streamsize>(saturation_deltas.size() * sizeof(uint64_t)));

	// T-1691 trailer (append-only, design S7 step 4, this file's header
	// comment): new bytes AFTER T-1689's own trailer above -- neither the
	// 29-row body nor the saturation trailer moves.
	const uint64_t num_target_rows = static_cast<uint64_t>(kv_trailer_rows.size());
	f.write(reinterpret_cast<const char*>(&num_target_rows), sizeof(num_target_rows));
	std::vector<uint64_t> target_rows_u64(kv_trailer_rows.begin(), kv_trailer_rows.end());
	f.write(reinterpret_cast<const char*>(target_rows_u64.data()),
	        static_cast<std::streamsize>(target_rows_u64.size() * sizeof(uint64_t)));
	f.write(reinterpret_cast<const char*>(&kv_trailer_context_length),
	        sizeof(kv_trailer_context_length));
	f.write(reinterpret_cast<const char*>(&kv_trailer_num_kv_heads), sizeof(kv_trailer_num_kv_heads));
	f.write(reinterpret_cast<const char*>(&kv_trailer_head_dim), sizeof(kv_trailer_head_dim));
	f.write(reinterpret_cast<const char*>(kv_trailer_codes.data()),
	        static_cast<std::streamsize>(kv_trailer_codes.size()));
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
	std::string site_dump_path;
	for (int i = 4; i < argc; ++i) {
		if (std::strcmp(argv[i], "--dump") == 0 && i + 1 < argc) {
			dump_path = argv[++i];
		} else if (std::strcmp(argv[i], "--site-dump") == 0 && i + 1 < argc) {
			site_dump_path = argv[++i];
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
		                   model_view.rope_tables, trace_workspace.data(), trace_workspace.size());
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

	// T-1689 (design S5): the saturation-delta baseline, taken here -- AFTER
	// the embedding row above, BEFORE the last token's own 28-layer walk
	// starts. Every saturation event from prefill (every earlier prompt
	// token's own 28-layer walk, above) is already reflected in
	// trace_seq.kv_saturation_count at this point and is excluded from every
	// delta below by this baseline subtraction -- the census counts ONLY
	// the last token's own per-layer landing writes, matching design S5's
	// "per-layer, cumulative" delta definition.
	std::vector<uint64_t> saturation_deltas;
	saturation_deltas.reserve(num_hidden_layers);
	uint64_t prev_saturation_count = trace_seq.kv_saturation_count;

	// T-1691 step 4 (design S7 step 4): the target-layer band, stated as DUMP
	// ROW NUMBERS (this file's header comment). `kv_trailer_context_length`
	// is the number of PRIOR positions -- fixed for the whole last-token
	// walk, taken here (before that walk's first layer runs), matching
	// `position` itself being "constant across every layer of this token"
	// (forward_sites.cpp's own comment, quoted in the design).
	//
	// Widened to all 28 rows (parity-extension pass, Brunel, 2026-08-04):
	// originally {5, 21..28} (the design's own divergence band plus the
	// layer-5 control). The reversal band (rows 12-20, unexplained, upstream
	// of the divergence onset) was never covered by this trailer, which
	// blocks the attention-interior and RoPE drives (both need the real
	// prior-position K/V codes this trailer carries) from reaching it even
	// though the site-dump band above already does. Independent of (and
	// still narrower in KIND than) the site-dump band, which this array does
	// not touch or share state with.
	constexpr uint32_t kKvTrailerTargetRows[] = {1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14,
	                                              15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28};
	const int64_t kv_trailer_context_length = trace_seq.context_length;
	std::vector<uint32_t> kv_trailer_rows_captured;
	std::vector<int8_t> kv_trailer_codes;
	uint64_t actual_kv_trailer_elements = 0;

	// T-1691 site-dump (widened T-1691 onset/amplification pass, Brunel,
	// 2026-08-04): originally the divergence band plus the layer-5 control
	// only ({5, 26, 27, 28} -- Dan's own stated scope for the earlier
	// build-sequencing pass). The onset/amplification question this pass
	// answers needs the site-level record at EVERY layer, not only the
	// locus band, so this array now covers all 28 rows (design's own full
	// depth) -- independent of (and still narrower in KIND than) the KV
	// trailer's own {5,21..28} band above, which this flag does not touch
	// or share state with. `trace_hook_state` is installed once (if
	// requested) and stays installed for the tool's whole run; `site_ctx`'s
	// own `active` flag is what actually gates which layers' records reach
	// `site_ctx.records`, toggled around each target row's own call below.
	constexpr uint32_t kSiteDumpTargetRows[] = {1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14,
	                                             15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28};
	auto IsSiteDumpTargetRow = [&](uint32_t row_number) {
		for (uint32_t target : kSiteDumpTargetRows) {
			if (target == row_number) return true;
		}
		return false;
	};
	SiteCaptureContext site_ctx;
	SslmTraceHookState trace_hook_state;
	const bool site_dump_requested = !site_dump_path.empty();
	if (site_dump_requested) {
		SslmSetTraceHook(trace_hook_state, &SiteCaptureHook, &site_ctx);
	}

	for (uint32_t step = 0; step < num_hidden_layers; ++step) {
		// This iteration's own dump row number, computed here (rather than
		// after the call, as the KV-trailer block below still does) because
		// the site-dump capture must be toggled ON before RunLayerLoop runs
		// this layer's own sites and OFF immediately after -- a record from
		// any other layer must never reach `site_ctx.records`, whether or
		// not this row is a KV-trailer target too.
		const uint32_t site_dump_row_number = step + 1;
		if (site_dump_requested) {
			site_ctx.active = IsSiteDumpTargetRow(site_dump_row_number);
		}
		const SslmForwardStatus st = RunLayerLoop(
		    trace_seq, layers.data(), num_hidden_layers, /*layer_budget=*/1, hidden_size,
		    model_view.config.head_dim, num_kv_heads, model_view.config.intermediate_size, context_cap,
		    model_view.rope_tables, trace_workspace.data(), trace_workspace.size(),
		    /*site_prefix=*/{}, /*token_index=*/0,
		    site_dump_requested ? &trace_hook_state : nullptr);
		if (site_dump_requested) {
			site_ctx.active = false;
		}
		if (st != SslmForwardStatus::Ok) {
			std::fprintf(stderr, "FAILED at stage=trace_layer_step: layer=%u status=%s\n", step,
			             SslmForwardStatusName(st));
			return 1;
		}
		rows.push_back(LayerSnapshot{
		    std::vector<int8_t>(trace_seq.hidden_codes, trace_seq.hidden_codes + hidden_size),
		    trace_seq.hidden_scale.m, trace_seq.hidden_scale.e});
		// T-1689: this layer's own new saturation events -- the delta since
		// the previous snapshot (the baseline above, or the previous
		// iteration's own snapshot). kv_saturation_count is never reset by
		// RunLayerLoop (design S2.1), so this subtraction is exactly the
		// count of events NEW at this layer, never a re-count of an earlier
		// one.
		const uint64_t cur_saturation_count = trace_seq.kv_saturation_count;
		saturation_deltas.push_back(cur_saturation_count - prev_saturation_count);
		prev_saturation_count = cur_saturation_count;

		// T-1691 step 4: this dump row's own number is `step + 1` (row 0 is
		// the embedding, pushed above the loop). `step` itself is the
		// 0-indexed `layer` parameter KeyRow/ValueRow take -- the exact
		// value RunLayerLoop's own body just used internally for this same
		// layer's landing write (forward_sites.cpp: `const uint32_t l =
		// seq.layer_index;`, taken BEFORE this call's own commit, which is
		// `step` here since `layer_budget == 1`).
		const uint32_t row_number = step + 1;
		bool is_target_row = false;
		for (uint32_t target : kKvTrailerTargetRows) {
			if (target == row_number) {
				is_target_row = true;
				break;
			}
		}
		if (is_target_row) {
			kv_trailer_rows_captured.push_back(row_number);
			for (int64_t pos = 0; pos < kv_trailer_context_length; ++pos) {
				for (size_t h = 0; h < num_kv_heads; ++h) {
					const int8_t* const k_row = KeyRow(trace_workspace.data(), step, context_cap,
					                                    num_kv_heads, model_view.config.head_dim, h, pos);
					kv_trailer_codes.insert(kv_trailer_codes.end(), k_row,
					                        k_row + model_view.config.head_dim);
					actual_kv_trailer_elements += model_view.config.head_dim;
				}
				for (size_t h = 0; h < num_kv_heads; ++h) {
					const int8_t* const v_row = ValueRow(trace_workspace.data(), step, context_cap,
					                                      num_kv_heads, model_view.config.head_dim, h, pos);
					kv_trailer_codes.insert(kv_trailer_codes.end(), v_row,
					                        v_row + model_view.config.head_dim);
					actual_kv_trailer_elements += model_view.config.head_dim;
				}
			}
		}
	}

	// T-1691 step 4, design S7 step 4's own two owed cells (D-SLM727b): a
	// count-reconciliation guard, checked against an INDEPENDENTLY-derived
	// expected count (num_target_rows actually reached * context_length *
	// num_kv_heads * head_dim * 2) -- never against the loop's own running
	// total, which a dropped- or duplicated-position defect in the loop
	// above would corrupt identically on both sides. Guard vitality: the
	// env var below (default unset, exercised only by
	// tools/test_t1691_kv_context_trailer.py's own guard-vitality cell)
	// deliberately perturbs `actual` by one element, confirming this
	// comparison -- the real, shipped comparison, not a copy of it -- fires
	// and rejects before any dump is written.
	const uint64_t expected_kv_trailer_elements =
	    static_cast<uint64_t>(kv_trailer_rows_captured.size()) *
	    static_cast<uint64_t>(kv_trailer_context_length) * static_cast<uint64_t>(num_kv_heads) *
	    static_cast<uint64_t>(model_view.config.head_dim) * 2;
	if (const char* force_mismatch = std::getenv("SSLM_T1691_KV_TRAILER_FORCE_MISMATCH")) {
		if (std::strcmp(force_mismatch, "1") == 0) {
			actual_kv_trailer_elements += 1;
		}
	}
	if (actual_kv_trailer_elements != expected_kv_trailer_elements) {
		std::fprintf(stderr,
		             "FAILED at stage=kv_trailer_count_guard: actual=%llu expected=%llu (rows=%zu "
		             "context_length=%lld num_kv_heads=%u head_dim=%u) -- a dropped or duplicated "
		             "position/head/row is the likely cause (no dump written)\n",
		             static_cast<unsigned long long>(actual_kv_trailer_elements),
		             static_cast<unsigned long long>(expected_kv_trailer_elements),
		             kv_trailer_rows_captured.size(), static_cast<long long>(kv_trailer_context_length),
		             static_cast<unsigned>(num_kv_heads), static_cast<unsigned>(model_view.config.head_dim));
		return 1;
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
		    oracle_workspace.data(), oracle_workspace.size());
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
		    model_view.rope_tables, oracle_workspace.data(), oracle_workspace.size());
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

	// --- Step 5 (design S4.1 step 5, T-1689 trailer extension): dump. --------
	const uint64_t fingerprint = Fnv1a64(prompt);
	if (!WriteDump(dump_path.c_str(), rows, static_cast<uint64_t>(hidden_size), fingerprint,
	               saturation_deltas, kv_trailer_rows_captured,
	               static_cast<uint64_t>(kv_trailer_context_length), static_cast<uint64_t>(num_kv_heads),
	               static_cast<uint64_t>(model_view.config.head_dim), kv_trailer_codes)) {
		std::fprintf(stderr, "FAILED at stage=dump_write: could not write \"%s\"\n", dump_path.c_str());
		return 1;
	}
	uint64_t total_saturation = 0;
	for (uint64_t d : saturation_deltas) total_saturation += d;
	std::printf("layer_trace_dumped: %zu rows x %zu hidden_size, prompt_fingerprint=0x%016llX, "
	            "%zu kv_saturation deltas (total=%llu), kv_context_trailer: %zu/%zu target rows "
	            "reached (context_length=%lld, %llu codes) -> %s\n",
	            rows.size(), hidden_size, static_cast<unsigned long long>(fingerprint),
	            saturation_deltas.size(), static_cast<unsigned long long>(total_saturation),
	            kv_trailer_rows_captured.size(), sizeof(kKvTrailerTargetRows) / sizeof(uint32_t),
	            static_cast<long long>(kv_trailer_context_length),
	            static_cast<unsigned long long>(actual_kv_trailer_elements), dump_path.c_str());

	// T-1691 site-dump (this session): written only when requested, to its
	// own separate file, after the self-check above has already confirmed
	// the trace walk agrees with production -- a site-dump is never written
	// from a run whose own hidden-state walk was not itself already proven
	// correct.
	if (site_dump_requested) {
		if (!WriteSiteDump(site_dump_path.c_str(), site_ctx.records, site_ctx.kv_records)) {
			std::fprintf(stderr, "FAILED at stage=site_dump_write: could not write \"%s\"\n",
			             site_dump_path.c_str());
			return 1;
		}
		std::printf("site_dump_written: %zu records across %zu target rows -> %s\n",
		            site_ctx.records.size(), sizeof(kSiteDumpTargetRows) / sizeof(uint32_t),
		            site_dump_path.c_str());
		std::printf("site_dump_written_kv: %zu kv-landing records -> %s\n",
		            site_ctx.kv_records.size(), site_dump_path.c_str());
	}
	return 0;
}
