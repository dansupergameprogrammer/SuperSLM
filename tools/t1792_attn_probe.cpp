// t1792_attn_probe.cpp -- T-1792: a copy of T-1789's own t1789_attention_probe.cpp (branch
// claude/t1789-attention-solve@eb06008, read in full before reuse), with exactly ONE change:
// `checkpoint_layers` below is widened from the six-layer sample {0,1,2,9,18,27} to ALL 28
// layers {0..27} -- T-1792's own commission is to measure whether the W(8) construction helps
// at model scale rather than only at the two shallow layers the six-layer sample happened to
// weight at 1/3 combined. Every self-check, dump format, and manual-replay discipline below is
// otherwise byte-identical to T-1789's own probe; only the layer SET being captured changes.
//
// T-1789's own header follows unchanged:
//
// t1789_attention_probe.cpp -- T-1789: a copy of T-1788's own t1788_weight_activation_probe.cpp
// (branch claude/t1788-weight-activation-quant@cb591df, read in full before reuse), with exactly
// ONE addition: the per-layer constants dump also carries attn_norm_site_constant (m,e) and
// q_site_constant (m,e) so the analysis script can re-derive normed_scale/q_scale on FOREIGN
// (hybrid, float-injected) inputs rather than only reconciling them from production dumps.
// Everything below this header block is T-1788's own probe, unchanged except that addition.
//
// Original header:
// t1788_weight_activation_probe.cpp -- T-1788: measure whether Q/K weight quantization and
// RMSNorm's own activation quantization explain the 80.1% residual T-1787 (D-SLM1296) left
// unexplained after removing Query quantization, the Key requantization chain, and RoPE.
//
// T-1787's own handoff named these as the most likely unmeasured candidates, and explicitly
// stated why its own method could not isolate them: every T-1787 substitution reused the SAME
// production int8 Q/K weight matrices and the SAME production int8 RmsNorm output in every
// variant -- "exact" there meant removing an ACTIVATION-side rounding step downstream of both,
// never a weight-side or RmsNorm-output-side one. This probe captures what a DIFFERENT
// experiment needs: the true (independent, checkpoint) weight comparison requires the engine's
// own DEQUANTIZED activation (normed_code * normed_scale, in real units) so it can be multiplied
// against the checkpoint's own float32 weight; the RmsNorm-own-quantization comparison requires
// the RAW pre-fold GEMM accumulator (not just the post-fold-post-bias "wide" value T-1787 already
// captured), because RmsNorm's own round is upstream of the fold and the fold is itself a
// deterministic function of the raw accumulator this probe now also dumps.
//
// WHAT'S CAPTURED, PER (layer, position) AT THE 6 CHECKPOINT LAYERS, EVERY PREFILL POSITION PLUS
// THE LAST TOKEN (RmsNorm and K's own projection are needed at every cached position; Q's own
// projection is needed only at the last position, single-query causal attention, matching every
// prior ticket in this chain):
//
//   h[] (int8, hidden_size)      -- RmsNorm's own INPUT (seq.hidden_codes at that position),
//                                    smaller to dump than the "wide" pre-round array and gives a
//                                    stronger self-check: reconstructing wide[] from h[]/g[]/
//                                    FloorDivI64/ISqrt in Python and self-checking
//                                    RequantTokenCodeWide(wide[i],r,s)==normed[i] validates this
//                                    probe's OWN Python-side FloorDivI64/ISqrt replica too, not
//                                    just the funnel's r/s derivation.
//   normed[] (int8, hidden_size) -- RmsNorm's own OUTPUT (self-check target above).
//   normed_scale (m,e)           -- RmsNorm's own out_scale (CombineCarriedScale(site_constant,
//                                    d_prime_factor)) -- UNCHANGED whether RmsNorm's own round is
//                                    removed or not (r,s, hence out_scale, are derived from wide[]
//                                    BEFORE the round -- the round does not feed back into them).
//   r, s                         -- RmsNorm's own funnel r/s (MaxAbsReduceWide/NormalizeScale/
//                                    CarriedScaleReciprocal on wide[]), called FOR REAL, so a
//                                    Python replica of those three functions self-checks against
//                                    a live value, not merely a value this probe alone asserts.
//
// PER LAYER (position-independent -- Q's own projection matters only at the last token, K's/V's
// only insofar as this file needs K's; V is untouched, out of this ticket's own scope):
//
//   raw Q accumulator (int64, hidden_size)          -- GemmInt8AccumulateRow(normed, q_weight),
//                                                       PRE fold, PRE bias -- last position only.
//   Q folded-pre-bias accumulator (int64, hidden_size) -- ApplyWeightScaleFold applied to the
//                                                       above, called FOR REAL -- self-check target
//                                                       for this probe's Python-side exact-fold
//                                                       replica (SaturatingRoundingDoublingHighMul
//                                                       + RoundingDivideByPOT, both closed-form).
//   q_wide[] (int64, hidden_size)                    -- POST fold, POST bias (T-1787's own
//                                                       "q_wide", unchanged in kind) -- last
//                                                       position only.
//   q_r, q_s                                         -- Q's OWN funnel r/s (dynamic, derived from
//                                                       q_wide's own magnitude) -- last position.
//   q_ln2, q_b, q_c (production)                     -- self-check target for this ticket's own
//                                                       ported CombineCarriedScale/
//                                                       IExpScaleConstants, at PRODUCTION's own
//                                                       (unperturbed) q_scale -- last position,
//                                                       per (layer, kv_head).
//
//   raw K accumulator (int64, head_dim), per (position, kv_head) -- same shape as Q's, but at
//     EVERY cached position (K feeds every row of the score matrix, not just the last).
//   K folded-pre-bias accumulator (int64, head_dim), per (position, kv_head).
//   kacc[] (int64, head_dim), per (position, kv_head)  -- T-1787's own "kacc" (post fold, post
//     bias), unchanged in kind.
//
// PER LAYER, CONSTANTS (position- and prompt-independent -- dumped once, not once per prompt):
//   q_fold_identity/mult/shift[hidden_size], k_fold_identity/mult/shift[kv_hidden_size],
//   attn_norm_gain[hidden_size], q_site_constant(m,e), kv_landing_r_t_k/e_t_k[num_kv_heads],
//   iexp_softmax_khead_m/e[num_kv_heads], q_weight[hidden_size*hidden_size] (int8),
//   k_weight[kv_hidden_size*hidden_size] (int8).
//
// SELF-CHECK DISCIPLINE, unchanged from T-1786/T-1787: every quantity this probe dumps is either
// (a) read directly off production's own internal state via the SAME compiled call sequence
// production uses (h[], normed[], normed_scale, raw/folded/wide accumulators, r/s, q_ln2/q_b/q_c,
// weights, fold constants -- all captured from a REAL call to RmsNormSite/GemmInt8AccumulateRow/
// ApplyWeightScaleFold/RequantChainChecked/IExpScaleConstants, or read directly off the marshaled
// LayerWeights), or (b) cross-checked bit-for-bit against production's own later, externally-
// observable output (the K self-check below, identical in kind to T-1786's/T-1787's own). No
// value in this dump is asserted without either provenance.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <span>
#include <string>
#include <vector>

#include "superslm/artifact.h"
#include "superslm/checked_chain_funnel.h"
#include "superslm/forward_sites.h"
#include "superslm/intmath.h"
#include "superslm/matmul.h"
#include "superslm/model.h"
#include "superslm/silu_lut.h"
#include "superslm/silu_lut_canonical.h"
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

int64_t ReadRopeTableEntryI64Copy(const uint8_t* base, uint64_t index) {
	const uint8_t* p = base + index * 8;
	uint64_t v = 0;
	for (int i = 0; i < 8; ++i) v |= static_cast<uint64_t>(p[i]) << (8 * i);
	return static_cast<int64_t>(v);
}

SslmForwardStatus ApplyBiasReconcileRowCopy(int64_t* acc, size_t out_channels, const int64_t* bias,
                                             int64_t in_scale_m, int64_t in_scale_e) {
	const SslmForwardStatus gate = CheckRoundingDivideByPotExponentDomain(kBiasQFormat, in_scale_e);
	if (gate != SslmForwardStatus::Ok) return gate;
	const int64_t r_a = CarriedScaleReciprocal(in_scale_m);
	bool any_out_of_domain = false;
	for (size_t i = 0; i < out_channels; ++i) {
		if (CheckBiasAccumulateMagnitudeDomain(acc[i], bias[i], kBiasQFormat, r_a, in_scale_e) !=
		    SslmForwardStatus::Ok) {
			any_out_of_domain = true;
		}
	}
	if (any_out_of_domain) return SslmForwardStatus::BiasReconcileProductOutOfDomain;
	for (size_t i = 0; i < out_channels; ++i) {
		acc[i] += BiasReconcile(bias[i], kBiasQFormat, r_a, in_scale_e);
	}
	return SslmForwardStatus::Ok;
}

// Extends T-1787's own ProjectAndFunnelCopyCapture with TWO more optional capture points: the
// RAW pre-fold accumulator and the folded-pre-bias accumulator (T-1787 captured only the final
// post-fold-post-bias "wide" value).
SslmForwardStatus ProjectAndFunnelCopyCapture3(const int8_t* in_codes, CarriedScale in_scale,
                                                const int8_t* weight, size_t in_channels,
                                                size_t out_channels, const int32_t* identity,
                                                const int32_t* mult, const int32_t* shift,
                                                CarriedScale site_constant, const int64_t* bias,
                                                int8_t* out_codes, CarriedScale* out_scale,
                                                std::vector<int64_t>* out_raw,
                                                std::vector<int64_t>* out_folded_prebias,
                                                std::vector<int64_t>* out_wide) {
	std::vector<int64_t> acc(out_channels);
	GemmInt8AccumulateRow(in_codes, weight, in_channels, out_channels, acc.data());
	if (out_raw != nullptr) *out_raw = acc;
	for (size_t i = 0; i < out_channels; ++i) {
		acc[i] = ApplyWeightScaleFold(acc[i], identity[i], mult[i], shift[i]);
	}
	if (out_folded_prebias != nullptr) *out_folded_prebias = acc;
	if (bias != nullptr) {
		const SslmForwardStatus bias_status =
		    ApplyBiasReconcileRowCopy(acc.data(), out_channels, bias, in_scale.m, in_scale.e);
		if (bias_status != SslmForwardStatus::Ok) return bias_status;
	}
	if (out_wide != nullptr) *out_wide = acc;
	const CarriedScale incoming[1] = {in_scale};
	const ChainResult result = RequantChainChecked(
	    acc.data(), out_channels, std::span<const CarriedScale>{incoming, 1}, site_constant, out_codes,
	    out_scale);
	return result.status;
}

struct NormCapture {
	uint32_t layer = 0;
	int64_t position = 0;
	int64_t normed_m = 0, normed_e = 0;
	int64_t r = 0;
	int32_t s = 0;
	std::vector<int8_t> h;       // hidden_size, RmsNorm's own input
	std::vector<int8_t> normed;  // hidden_size, RmsNorm's own output (self-check target)
};

struct QCapture3 {
	uint32_t layer = 0;
	int64_t q_r = 0;
	int32_t q_s = 0;
	int64_t site_m = 0, site_e = 0;
	std::vector<int64_t> raw;            // hidden_size, pre-fold
	std::vector<int64_t> folded_prebias; // hidden_size, post-fold pre-bias
	std::vector<int64_t> wide;           // hidden_size, post-fold post-bias
};

struct KCapture3 {
	uint32_t layer = 0;
	int64_t position = 0;
	uint32_t kv_head = 0;
	int64_t normed_m = 0, normed_e = 0;
	int64_t r_t = 0, e_t = 0;
	std::vector<int64_t> raw;             // head_dim, pre-fold
	std::vector<int64_t> folded_prebias;  // head_dim, post-fold pre-bias
	std::vector<int64_t> kacc;            // head_dim, post-fold post-bias
};

struct CapturedProbRow {
	uint32_t layer = 0, head = 0;
	int64_t width = 0, q_ln2 = 0, q_b = 0, q_c = 0;
	std::vector<int64_t> scores, probs;
};

void PrintUsage(const char* argv0) {
	std::fprintf(stderr,
	             "usage: %s <model.sslm> <tokenizer.sslm> \"<prompt>\" --dump-prefix <path> "
	             "[--weights-dump <path>]\n",
	             argv0);
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
	std::string dump_prefix, weights_dump_path;
	for (int i = 4; i < argc; ++i) {
		if (std::strcmp(argv[i], "--dump-prefix") == 0 && i + 1 < argc) {
			dump_prefix = argv[++i];
		} else if (std::strcmp(argv[i], "--weights-dump") == 0 && i + 1 < argc) {
			weights_dump_path = argv[++i];
		} else {
			std::fprintf(stderr, "unrecognized argument: %s\n", argv[i]);
			PrintUsage(argv[0]);
			return 2;
		}
	}
	if (dump_prefix.empty()) {
		std::fprintf(stderr, "FAILED at stage=args: --dump-prefix <path> is required\n");
		return 2;
	}

	std::vector<uint8_t> tok_bytes;
	if (!ReadFile(tokenizer_path.c_str(), tok_bytes)) {
		std::fprintf(stderr, "FAILED at stage=tokenizer_file_read\n");
		return 1;
	}
	SslmArtifact tok_artifact;
	SslmError tok_open_err;
	if (SslmArtifact::OpenFromMemory(tok_bytes.data(), tok_bytes.size(), tok_artifact, &tok_open_err) !=
	    SslmStatus::Ok) {
		std::fprintf(stderr, "FAILED at stage=tokenizer_artifact_open\n");
		return 1;
	}
	TokenizerView tokenizer;
	std::string tok_err;
	if (!TokenizerView::Open(tok_artifact, tokenizer, &tok_err)) {
		std::fprintf(stderr, "FAILED at stage=tokenizer_view_open: %s\n", tok_err.c_str());
		return 1;
	}
	const std::vector<int32_t> prompt_tokens = tokenizer.Encode(prompt);
	if (prompt_tokens.empty()) {
		std::fprintf(stderr, "FAILED at stage=tokenizer_encode\n");
		return 1;
	}

	std::vector<uint8_t> model_bytes;
	if (!ReadFile(model_path.c_str(), model_bytes)) {
		std::fprintf(stderr, "FAILED at stage=model_file_read\n");
		return 1;
	}
	SslmModelView model_view;
	std::string model_err;
	if (SslmModel::Load(model_bytes.data(), model_bytes.size(), model_view, &model_err) !=
	    SslmModelStatus::Ok) {
		std::fprintf(stderr, "FAILED at stage=model_load: %s\n", model_err.c_str());
		return 1;
	}
	std::printf("model loaded: hidden_size=%u layers=%u heads=%u/%u head_dim=%u context_cap=%u "
	            "prompt_tokens=%zu\n",
	            model_view.config.hidden_size, model_view.config.num_hidden_layers,
	            model_view.config.num_attention_heads, model_view.config.num_key_value_heads,
	            model_view.config.head_dim, model_view.config.context_cap, prompt_tokens.size());

	const uint32_t num_heads = model_view.config.num_attention_heads;
	const uint32_t num_kv_heads = model_view.config.num_key_value_heads;
	const uint32_t num_hidden_layers = model_view.config.num_hidden_layers;
	const size_t hidden_size = model_view.config.hidden_size;
	const size_t head_dim = model_view.config.head_dim;
	const size_t intermediate_size = model_view.config.intermediate_size;
	const size_t kv_hidden_size = num_kv_heads * head_dim;
	const size_t group = num_heads / num_kv_heads;

	PreflightScanWscFolds(model_view);

	std::vector<LayerBacking> backings(num_hidden_layers);
	std::vector<LayerWeights> layers(num_hidden_layers);
	for (uint32_t l = 0; l < num_hidden_layers; ++l) {
		std::string marshal_err;
		if (!MarshalLayer(model_view, l, num_heads, num_kv_heads, backings[l], layers[l], &marshal_err)) {
			std::fprintf(stderr, "FAILED at stage=layer_weights_marshal: layer=%u: %s\n", l,
			             marshal_err.c_str());
			return 1;
		}
	}

	const SslmTensorView* embed_w = model_view.weights.Tensor("embed");
	if (!embed_w) {
		std::fprintf(stderr, "FAILED at stage=head_marshal: missing embed tensor\n");
		return 1;
	}
	bool ok = true;
	CarriedScale embed_site_constant = ReadCarriedScale(model_view.composition_constants, "embed", &ok);
	if (!ok) {
		std::fprintf(stderr, "FAILED at stage=head_marshal: missing embed site constant\n");
		return 1;
	}
	const int8_t* embed_weights = reinterpret_cast<const int8_t*>(embed_w->data);

	const int64_t context_cap = static_cast<int64_t>(model_view.config.context_cap);
	const size_t kv_bytes = static_cast<size_t>(num_hidden_layers) * static_cast<size_t>(context_cap) *
	                        num_kv_heads * head_dim * 2;

	std::vector<uint8_t> trace_workspace(kv_bytes);
	std::vector<int8_t> trace_hidden_codes(hidden_size);
	SequenceLayerState trace_seq;
	trace_seq.hidden_codes = trace_hidden_codes.data();

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

	// T-1792: ALL 28 layers, not the six-layer checkpoint sample. Everything downstream of this
	// vector (self-checks, dump loops) is unchanged and iterates whatever it holds.
	std::vector<uint32_t> checkpoint_layers(num_hidden_layers);
	for (uint32_t l = 0; l < num_hidden_layers; ++l) checkpoint_layers[l] = l;
	auto is_checkpoint = [&](uint32_t l) {
		return std::find(checkpoint_layers.begin(), checkpoint_layers.end(), l) != checkpoint_layers.end();
	};

	std::vector<NormCapture> all_norm;
	std::vector<KCapture3> all_k;

	// Capture RmsNorm's own input/output at a checkpoint layer, at the CURRENT position, as a
	// SIDE computation (calls RmsNormSite for real), self-checking wide[]/r/s via
	// RequantTokenCodeWide against the just-captured `normed` output.
	auto CaptureNorm = [&](uint32_t l, int64_t position) -> bool {
		NormCapture nc;
		nc.layer = l;
		nc.position = position;
		nc.h.assign(trace_seq.hidden_codes, trace_seq.hidden_codes + hidden_size);
		nc.normed.assign(hidden_size, 0);
		CarriedScale normed_scale{};
		const SslmForwardStatus st =
		    RmsNormSite(trace_seq.hidden_codes, layers[l].attn_norm_gain, hidden_size, CarriedScale{},
		                layers[l].attn_norm_site_constant, nc.normed.data(), &normed_scale);
		if (st != SslmForwardStatus::Ok) {
			std::fprintf(stderr, "FAILED at stage=norm_capture: layer=%u position=%lld status=%s\n", l,
			             static_cast<long long>(position), SslmForwardStatusName(st));
			return false;
		}
		nc.normed_m = normed_scale.m;
		nc.normed_e = normed_scale.e;
		// Reconstruct wide[] independently (FloorDivI64/ISqrt, RmsNormSite's own formula) and
		// derive r/s for real, then self-check.
		int64_t sumsq = 0;
		for (size_t i = 0; i < hidden_size; ++i) {
			const int64_t hi = static_cast<int64_t>(nc.h[i]);
			sumsq += hi * hi;
		}
		constexpr int kNormFracBitsLocal = 16;
		int64_t root = ISqrt(FloorDivI64(sumsq << (2 * kNormFracBitsLocal), static_cast<int64_t>(hidden_size)));
		root = root > 1 ? root : 1;
		std::vector<int64_t> wide(hidden_size);
		for (size_t i = 0; i < hidden_size; ++i) {
			const int64_t hi = static_cast<int64_t>(nc.h[i]);
			wide[i] = FloorDivI64(hi << (2 * kNormFracBitsLocal), root) *
			          static_cast<int64_t>(layers[l].attn_norm_gain[i]);
		}
		const int64_t d_prime = MaxAbsReduceWide(wide.data(), hidden_size);
		const NormalizedScale ns = NormalizeScale(d_prime);
		nc.r = CarriedScaleReciprocal(ns.dn);
		nc.s = ns.s;
		for (size_t i = 0; i < hidden_size; ++i) {
			const int8_t replica = RequantTokenCodeWide(wide[i], nc.r, nc.s);
			if (replica != nc.normed[i]) {
				std::fprintf(stderr,
				             "FAILED at stage=norm_self_check: layer=%u position=%lld i=%zu "
				             "replica=%d production=%d\n",
				             l, static_cast<long long>(position), i, static_cast<int>(replica),
				             static_cast<int>(nc.normed[i]));
				return false;
			}
		}
		all_norm.push_back(std::move(nc));
		return true;
	};

	// K's own raw/folded-prebias/final accumulator at a checkpoint layer, current position, for
	// every kv_head -- computed as a SIDE call using the SAME `normed` just captured/self-checked
	// above (re-derived here via RmsNormSite again -- cheap, deterministic, and this second call
	// is itself another instance of the self-check discipline: if RmsNormSite were somehow
	// nondeterministic this would immediately diverge from `all_norm`'s own captured normed[]).
	auto CaptureK = [&](uint32_t l, int64_t position) -> bool {
		std::vector<int8_t> normed(hidden_size);
		CarriedScale normed_scale{};
		SslmForwardStatus st = RmsNormSite(trace_seq.hidden_codes, layers[l].attn_norm_gain, hidden_size,
		                                    CarriedScale{}, layers[l].attn_norm_site_constant,
		                                    normed.data(), &normed_scale);
		if (st != SslmForwardStatus::Ok) return false;
		for (uint32_t h = 0; h < num_kv_heads; ++h) {
			KCapture3 kc;
			kc.layer = l;
			kc.position = position;
			kc.kv_head = h;
			kc.normed_m = normed_scale.m;
			kc.normed_e = normed_scale.e;
			kc.r_t = layers[l].kv_landing_r_t_k[h];
			kc.e_t = layers[l].kv_landing_e_t_k[h];
			std::vector<int64_t> raw(head_dim);
			GemmInt8AccumulateRow(normed.data(), layers[l].k_weight + h * head_dim * hidden_size,
			                      hidden_size, head_dim, raw.data());
			kc.raw = raw;
			std::vector<int64_t> folded(head_dim);
			for (size_t d = 0; d < head_dim; ++d) {
				folded[d] = ApplyWeightScaleFold(raw[d], layers[l].k_fold_identity[h * head_dim + d],
				                                 layers[l].k_fold_mult[h * head_dim + d],
				                                 layers[l].k_fold_shift[h * head_dim + d]);
			}
			kc.folded_prebias = folded;
			if (layers[l].k_bias != nullptr) {
				st = ApplyBiasReconcileRowCopy(folded.data(), head_dim, layers[l].k_bias + h * head_dim,
				                                normed_scale.m, normed_scale.e);
				if (st != SslmForwardStatus::Ok) return false;
			}
			kc.kacc = folded;
			all_k.push_back(std::move(kc));
		}
		return true;
	};

	// Prefill: every token except the last, capturing norm+K at checkpoint layers before
	// production's own real forward step for that layer runs.
	for (size_t tok_idx = 0; tok_idx + 1 < prompt_tokens.size(); ++tok_idx) {
		SslmForwardStatus st = EmbedWholeToken(prompt_tokens[tok_idx]);
		if (st != SslmForwardStatus::Ok) {
			std::fprintf(stderr, "FAILED at stage=prefill_embed: position=%zu status=%s\n", tok_idx,
			             SslmForwardStatusName(st));
			return 1;
		}
		for (uint32_t l = 0; l < num_hidden_layers; ++l) {
			const int64_t position = trace_seq.context_length;
			if (is_checkpoint(l)) {
				if (!CaptureNorm(l, position)) return 1;
				if (!CaptureK(l, position)) return 1;
			}
			st = RunLayerLoop(trace_seq, layers.data(), num_hidden_layers, /*layer_budget=*/1, hidden_size,
			                   head_dim, num_kv_heads, intermediate_size, context_cap, model_view.rope_tables,
			                   trace_workspace.data(), trace_workspace.size());
			if (st != SslmForwardStatus::Ok) {
				std::fprintf(stderr, "FAILED at stage=prefill_layer: layer=%u position=%lld status=%s\n", l,
				             static_cast<long long>(position), SslmForwardStatusName(st));
				return 1;
			}
		}
		std::printf("prefill position=%zu captured\n", tok_idx);
	}

	{
		const SslmForwardStatus st = EmbedWholeToken(prompt_tokens.back());
		if (st != SslmForwardStatus::Ok) {
			std::fprintf(stderr, "FAILED at stage=last_token_embed: status=%s\n", SslmForwardStatusName(st));
			return 1;
		}
	}
	const int64_t width_at_last_token = trace_seq.context_length + 1;
	std::printf("last prompt token: context_length=%lld width=%lld\n",
	            static_cast<long long>(trace_seq.context_length), static_cast<long long>(width_at_last_token));

	std::vector<CapturedProbRow> all_rows;
	std::vector<QCapture3> all_q;

	for (uint32_t step = 0; step < num_hidden_layers; ++step) {
		if (is_checkpoint(step)) {
			const int64_t last_position = trace_seq.context_length;
			if (!CaptureNorm(step, last_position)) return 1;
			if (!CaptureK(step, last_position)) return 1;

			std::vector<int8_t> manual_hidden_codes(trace_hidden_codes);
			SequenceLayerState manual_seq = trace_seq;
			manual_seq.hidden_codes = manual_hidden_codes.data();

			// Manual replay of this one layer through the softmax, extending T-1786/T-1787's own
			// ManualRunOneLayer with the Q raw/folded_prebias capture.
			const size_t num_heads_l = hidden_size / head_dim;
			std::vector<int8_t> normed(hidden_size), q_codes(hidden_size), o_codes(hidden_size);
			std::vector<int8_t> q_rot(hidden_size), k_rot(hidden_size), ctx_codes(hidden_size);
			std::vector<int8_t> gate_codes(intermediate_size), up_codes(intermediate_size);
			std::vector<int8_t> act_codes(intermediate_size), down_codes(hidden_size);
			std::vector<int8_t> stream_next(hidden_size), attn_stream(hidden_size);
			CarriedScale normed_scale{}, q_scale{}, ctx_scale{}, o_scale{};
			CarriedScale mlp_normed_scale{}, gate_scale{}, up_scale{}, act_scale{}, down_scale{};
			CarriedScale stream_scale{}, attn_stream_scale{};
			SslmForwardStatus st;

			st = RmsNormSite(manual_seq.hidden_codes, layers[step].attn_norm_gain, hidden_size,
			                 manual_seq.hidden_scale, layers[step].attn_norm_site_constant, normed.data(),
			                 &normed_scale);
			if (st != SslmForwardStatus::Ok) {
				std::fprintf(stderr, "FAILED at stage=manual_rmsnorm: layer=%u status=%s\n", step,
				             SslmForwardStatusName(st));
				return 1;
			}

			QCapture3 qc;
			qc.layer = step;
			qc.site_m = layers[step].q_site_constant.m;
			qc.site_e = layers[step].q_site_constant.e;
			st = ProjectAndFunnelCopyCapture3(normed.data(), normed_scale, layers[step].q_weight, hidden_size,
			                                   hidden_size, layers[step].q_fold_identity,
			                                   layers[step].q_fold_mult, layers[step].q_fold_shift,
			                                   layers[step].q_site_constant, layers[step].q_bias,
			                                   q_codes.data(), &q_scale, &qc.raw, &qc.folded_prebias, &qc.wide);
			if (st != SslmForwardStatus::Ok) {
				std::fprintf(stderr, "FAILED at stage=manual_q_project: layer=%u status=%s\n", step,
				             SslmForwardStatusName(st));
				return 1;
			}
			{
				const int64_t d_prime = MaxAbsReduceWide(qc.wide.data(), hidden_size);
				const NormalizedScale ns = NormalizeScale(d_prime);
				qc.q_r = CarriedScaleReciprocal(ns.dn);
				qc.q_s = ns.s;
				for (size_t i = 0; i < hidden_size; ++i) {
					const int8_t replica = RequantTokenCodeWide(qc.wide[i], qc.q_r, qc.q_s);
					if (replica != q_codes[i]) {
						std::fprintf(stderr,
						             "FAILED at stage=q_self_check: layer=%u i=%zu replica=%d "
						             "production=%d\n",
						             step, i, static_cast<int>(replica), static_cast<int>(q_codes[i]));
						return 1;
					}
				}
			}
			all_q.push_back(qc);

			{
				const size_t kv_hidden_size_l = num_kv_heads * head_dim;
				std::vector<int64_t> kacc(kv_hidden_size_l), vacc(kv_hidden_size_l);
				GemmInt8AccumulateRow(normed.data(), layers[step].k_weight, hidden_size, kv_hidden_size_l,
				                      kacc.data());
				GemmInt8AccumulateRow(normed.data(), layers[step].v_weight, hidden_size, kv_hidden_size_l,
				                      vacc.data());
				for (size_t i = 0; i < kv_hidden_size_l; ++i) {
					kacc[i] = ApplyWeightScaleFold(kacc[i], layers[step].k_fold_identity[i],
					                               layers[step].k_fold_mult[i], layers[step].k_fold_shift[i]);
					vacc[i] = ApplyWeightScaleFold(vacc[i], layers[step].v_fold_identity[i],
					                               layers[step].v_fold_mult[i], layers[step].v_fold_shift[i]);
				}
				if (layers[step].k_bias != nullptr) {
					st = ApplyBiasReconcileRowCopy(kacc.data(), kv_hidden_size_l, layers[step].k_bias,
					                                normed_scale.m, normed_scale.e);
					if (st != SslmForwardStatus::Ok) return 1;
				}
				if (layers[step].v_bias != nullptr) {
					st = ApplyBiasReconcileRowCopy(vacc.data(), kv_hidden_size_l, layers[step].v_bias,
					                                normed_scale.m, normed_scale.e);
					if (st != SslmForwardStatus::Ok) return 1;
				}
				for (size_t h = 0; h < num_kv_heads; ++h) {
					int8_t* const k_row =
					    MutableKeyRow(trace_workspace.data(), step, context_cap, num_kv_heads, head_dim, h,
					                  last_position);
					int8_t* const v_row =
					    MutableValueRow(trace_workspace.data(), step, context_cap, num_kv_heads, head_dim, h,
					                    last_position);
					for (size_t d = 0; d < head_dim; ++d) {
						const size_t i = h * head_dim + d;
						k_row[d] = static_cast<int8_t>(ClampRopeCode(
						    LandingRescale(kacc[i], normed_scale.m, layers[step].kv_landing_r_t_k[h],
						                   normed_scale.e, layers[step].kv_landing_e_t_k[h],
						                   &manual_seq.kv_saturation_count)));
						v_row[d] = static_cast<int8_t>(ClampRopeCode(
						    LandingRescale(vacc[i], normed_scale.m, layers[step].kv_landing_r_t_v[h],
						                   normed_scale.e, layers[step].kv_landing_e_t_v[h],
						                   &manual_seq.kv_saturation_count)));
					}
				}
			}

			for (size_t h = 0; h < num_heads_l; ++h) {
				st = RopeApplySite(q_codes.data() + h * head_dim, head_dim, last_position, context_cap,
				                   model_view.rope_tables, q_rot.data() + h * head_dim);
				if (st != SslmForwardStatus::Ok) return 1;
				const size_t kv_head = h / group;
				const int8_t* const k_row_before_rotate =
				    KeyRow(trace_workspace.data(), step, context_cap, num_kv_heads, head_dim, kv_head,
				           last_position);
				st = RopeApplySite(k_row_before_rotate, head_dim, last_position, context_cap,
				                   model_view.rope_tables, k_rot.data() + h * head_dim);
				if (st != SslmForwardStatus::Ok) return 1;
			}
			for (size_t h = 0; h < num_heads_l; ++h) {
				const size_t kv_head = h / group;
				int8_t* const k_row = MutableKeyRow(trace_workspace.data(), step, context_cap, num_kv_heads,
				                                    head_dim, kv_head, last_position);
				for (size_t d = 0; d < head_dim; ++d) k_row[d] = k_rot[h * head_dim + d];
			}

			std::vector<int64_t> ctx_wide(hidden_size);
			{
				std::vector<int64_t> khead_q_ln2(num_kv_heads), khead_q_b(num_kv_heads), khead_q_c(num_kv_heads);
				std::vector<bool> khead_derived(num_kv_heads, false);
				for (size_t h = 0; h < num_heads_l; ++h) {
					std::vector<int64_t> scores(width_at_last_token), probs(width_at_last_token),
					    ctx_acc(head_dim);
					const size_t kv_head = h / group;
					if (!khead_derived[kv_head]) {
						const int64_t sm_khead_m = layers[step].iexp_softmax_khead_m[kv_head];
						const int64_t sm_khead_e = layers[step].iexp_softmax_khead_e[kv_head];
						const CarriedScale sm = CombineCarriedScale(q_scale, CarriedScale{sm_khead_m, sm_khead_e});
						int64_t derived_q_ln2 = 0, derived_q_b = 0, derived_q_c = 0;
						const IExpScaleDomain scale_domain =
						    IExpScaleConstants(sm.m, sm.e, kIExpLn2Q, 30, kIExpBQ, 30, kIExpCaQ, 30,
						                       &derived_q_ln2, &derived_q_b, &derived_q_c);
						if (scale_domain != IExpScaleDomain::kOk) return 1;
						khead_q_ln2[kv_head] = derived_q_ln2;
						khead_q_b[kv_head] = derived_q_b;
						khead_q_c[kv_head] = derived_q_c;
						khead_derived[kv_head] = true;
					}
					// production's own q_ln2/q_b/q_c for THIS (layer,head) are already stashed
					// below in `row` (all_rows) -- the self-check target for this ticket's own
					// ported CombineCarriedScale/IExpScaleConstants (candidate 0, production
					// q_scale, no substitution).
					st = CheckSoftmaxRowWidthDomain(khead_q_b[kv_head], khead_q_c[kv_head], width_at_last_token);
					if (st != SslmForwardStatus::Ok) return 1;
					const int8_t* const k_rows_base =
					    KeyRow(trace_workspace.data(), step, context_cap, num_kv_heads, head_dim, kv_head, 0);
					GemmInt8AccumulateRow(q_rot.data() + h * head_dim, k_rows_base, head_dim,
					                      static_cast<size_t>(width_at_last_token), scores.data());
					const bool well_formed =
					    SoftmaxRowQ15(scores.data(), static_cast<size_t>(width_at_last_token),
					                 khead_q_ln2[kv_head], khead_q_b[kv_head], khead_q_c[kv_head], probs.data());
					if (!well_formed) return 1;

					CapturedProbRow row;
					row.layer = step;
					row.head = static_cast<uint32_t>(h);
					row.width = width_at_last_token;
					row.q_ln2 = khead_q_ln2[kv_head];
					row.q_b = khead_q_b[kv_head];
					row.q_c = khead_q_c[kv_head];
					row.scores = scores;
					row.probs = probs;
					all_rows.push_back(std::move(row));

					const int8_t* const v_rows_base =
					    ValueRow(trace_workspace.data(), step, context_cap, num_kv_heads, head_dim, kv_head, 0);
					GemmProbQ15Accumulate(probs.data(), v_rows_base, static_cast<size_t>(width_at_last_token),
					                      head_dim, ctx_acc.data());
					for (size_t d = 0; d < head_dim; ++d) {
						ctx_wide[h * head_dim + d] = ApplyWeightScaleFold(
						    ctx_acc[d], layers[step].ctx_fold_identity[h], layers[step].ctx_fold_mult[h],
						    layers[step].ctx_fold_shift[h]);
					}
				}
				const ChainResult ctx_result =
				    RequantChainChecked(ctx_wide.data(), hidden_size, std::span<const CarriedScale>{},
				                        layers[step].ctx_fold_site_constant, ctx_codes.data(), &ctx_scale);
				if (ctx_result.status != SslmForwardStatus::Ok) return 1;
			}

			st = ProjectAndFunnelCopyCapture3(ctx_codes.data(), ctx_scale, layers[step].o_weight, hidden_size,
			                                   hidden_size, layers[step].o_fold_identity,
			                                   layers[step].o_fold_mult, layers[step].o_fold_shift,
			                                   layers[step].o_site_constant, nullptr, o_codes.data(), &o_scale,
			                                   nullptr, nullptr, nullptr);
			if (st != SslmForwardStatus::Ok) return 1;

			st = ResidualReconcileSite(o_codes.data(), o_scale, manual_seq.hidden_codes, manual_seq.hidden_scale,
			                           hidden_size, layers[step].attn_residual_site_constant, attn_stream.data(),
			                           &attn_stream_scale);
			if (st != SslmForwardStatus::Ok) return 1;

			st = RmsNormSite(attn_stream.data(), layers[step].mlp_norm_gain, hidden_size, attn_stream_scale,
			                 layers[step].mlp_norm_site_constant, normed.data(), &mlp_normed_scale);
			if (st != SslmForwardStatus::Ok) return 1;

			st = ProjectAndFunnelCopyCapture3(normed.data(), mlp_normed_scale, layers[step].gate_weight,
			                                   hidden_size, intermediate_size, layers[step].gate_fold_identity,
			                                   layers[step].gate_fold_mult, layers[step].gate_fold_shift,
			                                   layers[step].gate_site_constant, nullptr, gate_codes.data(),
			                                   &gate_scale, nullptr, nullptr, nullptr);
			if (st != SslmForwardStatus::Ok) return 1;
			st = ProjectAndFunnelCopyCapture3(normed.data(), mlp_normed_scale, layers[step].up_weight,
			                                   hidden_size, intermediate_size, layers[step].up_fold_identity,
			                                   layers[step].up_fold_mult, layers[step].up_fold_shift,
			                                   layers[step].up_site_constant, nullptr, up_codes.data(),
			                                   &up_scale, nullptr, nullptr, nullptr);
			if (st != SslmForwardStatus::Ok) return 1;

			st = MlpActSite(gate_codes.data(), gate_scale, up_codes.data(), up_scale, intermediate_size,
			                kSiluLutCanonicalTable, layers[step].mlp_act_site_constant, act_codes.data(),
			                &act_scale);
			if (st != SslmForwardStatus::Ok) return 1;

			st = ProjectAndFunnelCopyCapture3(act_codes.data(), act_scale, layers[step].down_weight,
			                                   intermediate_size, hidden_size, layers[step].down_fold_identity,
			                                   layers[step].down_fold_mult, layers[step].down_fold_shift,
			                                   layers[step].down_site_constant, nullptr, down_codes.data(),
			                                   &down_scale, nullptr, nullptr, nullptr);
			if (st != SslmForwardStatus::Ok) return 1;

			st = ResidualReconcileSite(down_codes.data(), down_scale, attn_stream.data(), attn_stream_scale,
			                           hidden_size, layers[step].mlp_residual_site_constant, stream_next.data(),
			                           &stream_scale);
			if (st != SslmForwardStatus::Ok) return 1;

			for (size_t i = 0; i < hidden_size; ++i) manual_seq.hidden_codes[i] = stream_next[i];
			manual_seq.hidden_scale = stream_scale;
			manual_seq.layer_index = step + 1;

			const SslmForwardStatus pst =
			    RunLayerLoop(trace_seq, layers.data(), num_hidden_layers, /*layer_budget=*/1, hidden_size,
			                 head_dim, num_kv_heads, intermediate_size, context_cap, model_view.rope_tables,
			                 trace_workspace.data(), trace_workspace.size());
			if (pst != SslmForwardStatus::Ok) return 1;

			const bool codes_match = std::memcmp(manual_seq.hidden_codes, trace_seq.hidden_codes, hidden_size) == 0;
			const bool scale_match = manual_seq.hidden_scale.m == trace_seq.hidden_scale.m &&
			                         manual_seq.hidden_scale.e == trace_seq.hidden_scale.e;
			if (!codes_match || !scale_match) {
				std::fprintf(stderr, "FAILED at stage=self_check: layer=%u codes_match=%d scale_match=%d\n",
				             step, codes_match ? 1 : 0, scale_match ? 1 : 0);
				return 1;
			}
			std::printf("self_check: layer=%u manual replay and production agree bit-for-bit (%zu rows)\n",
			            step, all_rows.size());
		} else {
			const SslmForwardStatus pst =
			    RunLayerLoop(trace_seq, layers.data(), num_hidden_layers, /*layer_budget=*/1, hidden_size,
			                 head_dim, num_kv_heads, intermediate_size, context_cap, model_view.rope_tables,
			                 trace_workspace.data(), trace_workspace.size());
			if (pst != SslmForwardStatus::Ok) return 1;
		}
	}

	// Dump 1: per-position normed capture (RmsNorm's own I/O + r/s).
	{
		std::ofstream f(dump_prefix + "_normed.txt");
		f << all_norm.size() << " " << hidden_size << "\n";
		for (const auto& n : all_norm) {
			f << n.layer << " " << n.position << " " << n.normed_m << " " << n.normed_e << " " << n.r << " "
			  << n.s;
			for (int8_t v : n.h) f << " " << static_cast<int>(v);
			for (int8_t v : n.normed) f << " " << static_cast<int>(v);
			f << "\n";
		}
	}
	// Dump 2: Q's raw/folded_prebias/wide at the last token, per checkpoint layer.
	{
		std::ofstream f(dump_prefix + "_qacc.txt");
		f << all_q.size() << " " << hidden_size << "\n";
		for (const auto& q : all_q) {
			f << q.layer << " " << q.q_r << " " << q.q_s << " " << q.site_m << " " << q.site_e;
			for (int64_t v : q.raw) f << " " << v;
			for (int64_t v : q.folded_prebias) f << " " << v;
			for (int64_t v : q.wide) f << " " << v;
			f << "\n";
		}
	}
	// Dump 3: K's raw/folded_prebias/kacc, every (layer,position,kv_head).
	{
		std::ofstream f(dump_prefix + "_kacc.txt");
		f << all_k.size() << " " << head_dim << "\n";
		for (const auto& k : all_k) {
			f << k.layer << " " << k.position << " " << k.kv_head << " " << k.normed_m << " " << k.normed_e
			  << " " << k.r_t << " " << k.e_t;
			for (int64_t v : k.raw) f << " " << v;
			for (int64_t v : k.folded_prebias) f << " " << v;
			for (int64_t v : k.kacc) f << " " << v;
			f << "\n";
		}
	}
	// Dump 4: scores (T-1787-compatible format).
	{
		std::ofstream f(dump_prefix + "_scores.txt");
		f << all_rows.size() << " " << width_at_last_token << "\n";
		for (const auto& row : all_rows) {
			f << row.layer << " " << row.head << " " << row.width << " " << row.q_ln2 << " " << row.q_b
			  << " " << row.q_c;
			for (int64_t s : row.scores) f << " " << s;
			for (int64_t p : row.probs) f << " " << p;
			f << "\n";
		}
	}
	// Dump 5: RoPE tables.
	{
		const SslmTensorView* cos_t = model_view.rope_tables.Tensor("cos");
		const SslmTensorView* sin_t = model_view.rope_tables.Tensor("sin");
		const size_t pairs = head_dim / 2;
		std::ofstream f(dump_prefix + "_rope.txt");
		f << width_at_last_token << " " << pairs << "\n";
		for (int64_t pos = 0; pos < width_at_last_token; ++pos) {
			f << pos;
			for (size_t i = 0; i < pairs; ++i) {
				const int64_t cos_q30 = ReadRopeTableEntryI64Copy(cos_t->data, static_cast<uint64_t>(pos) * pairs + i);
				const int64_t sin_q30 = ReadRopeTableEntryI64Copy(sin_t->data, static_cast<uint64_t>(pos) * pairs + i);
				f << " " << cos_q30 << " " << sin_q30;
			}
			f << "\n";
		}
	}
	// Dump 6: per-layer constants (once regardless of prompt -- but this tool is invoked once per
	// prompt, so this dump is written every time; the analysis script reads only P1's copy, since
	// the constants are provably prompt-independent -- same checkpoint, same layer marshal).
	{
		std::ofstream f(dump_prefix + "_layerconst.txt");
		f << checkpoint_layers.size() << " " << hidden_size << " " << kv_hidden_size << " " << num_kv_heads
		  << "\n";
		for (uint32_t l : checkpoint_layers) {
			f << l;
			for (size_t i = 0; i < hidden_size; ++i) f << " " << layers[l].q_fold_identity[i];
			for (size_t i = 0; i < hidden_size; ++i) f << " " << layers[l].q_fold_mult[i];
			for (size_t i = 0; i < hidden_size; ++i) f << " " << layers[l].q_fold_shift[i];
			for (size_t i = 0; i < kv_hidden_size; ++i) f << " " << layers[l].k_fold_identity[i];
			for (size_t i = 0; i < kv_hidden_size; ++i) f << " " << layers[l].k_fold_mult[i];
			for (size_t i = 0; i < kv_hidden_size; ++i) f << " " << layers[l].k_fold_shift[i];
			for (size_t h = 0; h < num_kv_heads; ++h) f << " " << layers[l].iexp_softmax_khead_m[h];
			for (size_t h = 0; h < num_kv_heads; ++h) f << " " << layers[l].iexp_softmax_khead_e[h];
			f << " " << (layers[l].q_bias != nullptr ? 1 : 0);
			if (layers[l].q_bias != nullptr) {
				for (size_t i = 0; i < hidden_size; ++i) f << " " << layers[l].q_bias[i];
			}
			f << " " << (layers[l].k_bias != nullptr ? 1 : 0);
			if (layers[l].k_bias != nullptr) {
				for (size_t i = 0; i < kv_hidden_size; ++i) f << " " << layers[l].k_bias[i];
			}
			// attn_norm_gain -- lets the analysis script reconstruct RmsNorm's own pre-round
			// wide[] from the dumped h[]/normed[] pair (Dump 1) via RmsNormSite's own formula
			// (FloorDivI64/ISqrt, forward_sites.cpp:222-258), needed to compute the RMSNorm-exact
			// candidate's continuous (un-rounded) activation.
			for (size_t i = 0; i < hidden_size; ++i) f << " " << layers[l].attn_norm_gain[i];
			// T-1789 addition: the two site constants the analysis script needs to RE-DERIVE
			// normed_scale (RmsNormSite: combine over empty incoming + site_constant + d' factor)
			// and q_scale on foreign (hybrid) inputs. Appended at the END of the line so every
			// T-1788-format reader's own fixed-prefix parse is unaffected.
			f << " " << layers[l].attn_norm_site_constant.m << " " << layers[l].attn_norm_site_constant.e
			  << " " << layers[l].q_site_constant.m << " " << layers[l].q_site_constant.e;
			f << "\n";
		}
	}
	// Dump 7 (optional, only written once -- caller passes --weights-dump only for P1): raw int8
	// q_weight/k_weight per checkpoint layer, binary, [out*in_channels] layout matching
	// GemmInt8AccumulateRow's own indexing (matmul.cpp:145, `weights + j*in_channels`).
	if (!weights_dump_path.empty()) {
		std::ofstream f(weights_dump_path, std::ios::binary);
		for (uint32_t l : checkpoint_layers) {
			f.write(reinterpret_cast<const char*>(layers[l].q_weight),
			        static_cast<std::streamsize>(hidden_size * hidden_size));
			f.write(reinterpret_cast<const char*>(layers[l].k_weight),
			        static_cast<std::streamsize>(kv_hidden_size * hidden_size));
		}
	}

	std::printf("\ndumps written: %s_{normed,qacc,kacc,scores,rope,layerconst}.txt%s\n", dump_prefix.c_str(),
	            weights_dump_path.empty() ? "" : (" + " + weights_dump_path).c_str());
	return 0;
}
