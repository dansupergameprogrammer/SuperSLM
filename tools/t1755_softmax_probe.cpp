// t1755_softmax_probe.cpp -- T-1755 (E6 re-measurement, D-SLM77's softmax C7/C8/C9 pins).
//
// Scratch, single-purpose instrument, not wired into any suite -- same convention
// T-1740's tools/t1740_pooled_trace.cpp uses ("do not modify existing reviewed
// tools/production code; if a new capture is needed, add a new scratch file").
//
// Question: does the shipped i-exp softmax construction (SoftmaxRowQ15, intmath.h,
// citing C7/C8/C9 at its own composition comment "e[k] = IExpEvaluate(IExpConstruct(
// s[k], q_ln2, q_b, q_c))  (C7/C8/C9)") still land within D-SLM77's measured
// 1.83%-perturbation / max-delta-2 band against an ideal (infinite-precision) softmax,
// on REAL captured scores from the compiled engine's own attention computation --
// now that S3.7 (D-SLM582, merged main@9fc75b0, 2026-08-01) has replaced the
// previously-hardcoded width=1/dead-intermediate attention path (D-SLM503, T-1434)
// with real causal-width, real per-kv-head-derived (q_ln2, q_b, q_c) triples.
//
// No existing instrument captures the softmax row inputs/outputs: sslm_layer_trace.cpp
// (T-1685) captures only per-layer HIDDEN-STATE output (post-MLP-residual), never the
// interior score/probability row SoftmaxRowQ15 computes and discards. `scores`/`probs`
// are local to RunLayerLoop's own attention-half block (forward_sites.cpp) and are never
// written to `seq` or `workspace`.
//
// What this tool does instead of modifying that reviewed, shipped function: it manually
// replays ONE full layer's forward (attention half + MLP half, in RunLayerLoop's own
// documented order) at a handful of checkpoint layers, using ONLY the same public-API
// site functions RunLayerLoop itself calls (RmsNormSite, ProjectAndFunnel,
// ApplyBiasReconcileRow, RopeApplySite, KeyRow/ValueRow/MutableKeyRow/MutableValueRow,
// LandingRescale, ClampRopeCode, ApplyWeightScaleFold, CombineCarriedScale,
// IExpScaleConstants, CheckSoftmaxRowWidthDomain, SoftmaxRowQ15, GemmProbQ15Accumulate,
// RequantChainChecked (via ProjectAndFunnel), ResidualReconcileSite, MlpActSite), in the
// exact order forward_sites.cpp's own RunLayerLoop body uses -- capturing the score/prob
// row this replay computes, which RunLayerLoop's own call to the identical functions
// would also have computed, byte for byte, had it exposed them.
//
// Self-check (same shape sslm_layer_trace.cpp/T-1685 and t1740_pooled_trace.cpp/T-1740
// use): at every checkpoint layer, the manual replay's OWN resulting hidden_codes/scale
// is compared, bit-for-bit, against PRODUCTION's own RunLayerLoop(layer_budget=1) call
// for that same layer, from the same pre-layer state. Only self-checked rows are used in
// the measurement below; a mismatch aborts with no report written.
//
// ProjectAndFunnel and ApplyBiasReconcileRow are defined with external linkage in
// src/forward/forward_sites.cpp but declared only in that translation unit (no header
// declares them, unlike every other RunLayerLoop-internal site function) -- redeclared
// here to link against the SAME compiled definitions RunLayerLoop itself calls (never a
// second, independent implementation).

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

// ProjectAndFunnel and ApplyBiasReconcileRow (forward_sites.cpp) turned out to be
// declared inside that file's own anonymous namespace -- internal linkage, not
// reachable from another translation unit despite not being marked `static`. Reusing
// the compiled definitions is therefore not possible; these are exact copies of their
// bodies (forward_sites.cpp, read at source: ApplyBiasReconcileRow lines 803-821,
// ProjectAndFunnel lines 832-859 at this worktree's HEAD), calling only public-header
// functions, never a re-derivation of their arithmetic.
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

SslmForwardStatus ProjectAndFunnelCopy(const int8_t* in_codes, CarriedScale in_scale,
                                        const int8_t* weight, size_t in_channels, size_t out_channels,
                                        const int32_t* identity, const int32_t* mult,
                                        const int32_t* shift, CarriedScale site_constant,
                                        const int64_t* bias, int8_t* out_codes,
                                        CarriedScale* out_scale) {
	std::vector<int64_t> acc(out_channels);
	GemmInt8AccumulateRow(in_codes, weight, in_channels, out_channels, acc.data());
	for (size_t i = 0; i < out_channels; ++i) {
		acc[i] = ApplyWeightScaleFold(acc[i], identity[i], mult[i], shift[i]);
	}
	if (bias != nullptr) {
		const SslmForwardStatus bias_status =
		    ApplyBiasReconcileRowCopy(acc.data(), out_channels, bias, in_scale.m, in_scale.e);
		if (bias_status != SslmForwardStatus::Ok) return bias_status;
	}
	const CarriedScale incoming[1] = {in_scale};
	const ChainResult result = RequantChainChecked(
	    acc.data(), out_channels, std::span<const CarriedScale>{incoming, 1}, site_constant, out_codes,
	    out_scale);
	return result.status;
}

// One self-checked softmax row: the exact (scores, width, q_ln2, q_b, q_c) triple
// SoftmaxRowQ15 consumed, and the exact Q15 probability row it produced.
struct CapturedRow {
	uint32_t layer = 0;
	uint32_t query_head = 0;
	int64_t width = 0;
	int64_t q_ln2 = 0, q_b = 0, q_c = 0;
	std::vector<int64_t> scores;  // width elements, pre-ShiftByMax (SoftmaxRowQ15's own input)
	std::vector<int64_t> probs;   // width elements, Q15 (SoftmaxRowQ15's own output)
};

void PrintUsage(const char* argv0) {
	std::fprintf(stderr,
	             "usage: %s <model.sslm> <tokenizer.sslm> \"<prompt>\" --dump <path>\n", argv0);
}

// Manual replay of exactly ONE layer's forward -- attention half + MLP half, in
// RunLayerLoop's own documented order (forward_sites.cpp) -- capturing every query
// head's softmax row along the way. `seq`/`workspace` are advanced in place, exactly as
// RunLayerLoop(layer_budget=1) would advance them.
SslmForwardStatus ManualRunOneLayer(SequenceLayerState& seq, const LayerWeights& lw,
                                     size_t hidden_size, size_t head_dim, size_t num_key_value_heads,
                                     size_t intermediate_size, int64_t context_cap,
                                     const SslmTensorManifest& rope_tables, uint8_t* workspace,
                                     std::vector<CapturedRow>* out_rows) {
	const size_t num_heads = hidden_size / head_dim;
	const size_t group = num_heads / num_key_value_heads;
	const int64_t position = seq.context_length;
	const size_t width = static_cast<size_t>(seq.context_length) + 1;
	const uint32_t l = seq.layer_index;

	std::vector<int8_t> normed(hidden_size), q_codes(hidden_size), o_codes(hidden_size);
	std::vector<int8_t> q_rot(hidden_size), k_rot(hidden_size), ctx_codes(hidden_size);
	std::vector<int8_t> gate_codes(intermediate_size), up_codes(intermediate_size);
	std::vector<int8_t> act_codes(intermediate_size), down_codes(hidden_size);
	std::vector<int8_t> stream_next(hidden_size), attn_stream(hidden_size);
	CarriedScale normed_scale{}, q_scale{}, ctx_scale{}, o_scale{};
	CarriedScale mlp_normed_scale{}, gate_scale{}, up_scale{}, act_scale{}, down_scale{};
	CarriedScale stream_scale{}, attn_stream_scale{};
	SslmForwardStatus st;

	st = RmsNormSite(seq.hidden_codes, lw.attn_norm_gain, hidden_size, seq.hidden_scale,
	                 lw.attn_norm_site_constant, normed.data(), &normed_scale);
	if (st != SslmForwardStatus::Ok) return st;

	st = ProjectAndFunnelCopy(normed.data(), normed_scale, lw.q_weight, hidden_size, hidden_size,
	                      lw.q_fold_identity, lw.q_fold_mult, lw.q_fold_shift, lw.q_site_constant,
	                      lw.q_bias, q_codes.data(), &q_scale);
	if (st != SslmForwardStatus::Ok) return st;

	{
		const size_t kv_hidden_size = num_key_value_heads * head_dim;
		std::vector<int64_t> kacc(kv_hidden_size), vacc(kv_hidden_size);
		GemmInt8AccumulateRow(normed.data(), lw.k_weight, hidden_size, kv_hidden_size, kacc.data());
		GemmInt8AccumulateRow(normed.data(), lw.v_weight, hidden_size, kv_hidden_size, vacc.data());
		for (size_t i = 0; i < kv_hidden_size; ++i) {
			kacc[i] = ApplyWeightScaleFold(kacc[i], lw.k_fold_identity[i], lw.k_fold_mult[i],
			                               lw.k_fold_shift[i]);
			vacc[i] = ApplyWeightScaleFold(vacc[i], lw.v_fold_identity[i], lw.v_fold_mult[i],
			                               lw.v_fold_shift[i]);
		}
		if (lw.k_bias != nullptr) {
			st = ApplyBiasReconcileRowCopy(kacc.data(), kv_hidden_size, lw.k_bias, normed_scale.m,
			                           normed_scale.e);
			if (st != SslmForwardStatus::Ok) return st;
		}
		if (lw.v_bias != nullptr) {
			st = ApplyBiasReconcileRowCopy(vacc.data(), kv_hidden_size, lw.v_bias, normed_scale.m,
			                           normed_scale.e);
			if (st != SslmForwardStatus::Ok) return st;
		}
		for (size_t h = 0; h < num_key_value_heads; ++h) {
			int8_t* const k_row =
			    MutableKeyRow(workspace, l, context_cap, num_key_value_heads, head_dim, h, position);
			int8_t* const v_row = MutableValueRow(workspace, l, context_cap, num_key_value_heads,
			                                      head_dim, h, position);
			for (size_t d = 0; d < head_dim; ++d) {
				const size_t i = h * head_dim + d;
				k_row[d] = static_cast<int8_t>(ClampRopeCode(
				    LandingRescale(kacc[i], normed_scale.m, lw.kv_landing_r_t_k[h], normed_scale.e,
				                   lw.kv_landing_e_t_k[h], &seq.kv_saturation_count)));
				v_row[d] = static_cast<int8_t>(ClampRopeCode(
				    LandingRescale(vacc[i], normed_scale.m, lw.kv_landing_r_t_v[h], normed_scale.e,
				                   lw.kv_landing_e_t_v[h], &seq.kv_saturation_count)));
			}
		}
	}

	for (size_t h = 0; h < num_heads; ++h) {
		st = RopeApplySite(q_codes.data() + h * head_dim, head_dim, position, context_cap, rope_tables,
		                   q_rot.data() + h * head_dim);
		if (st != SslmForwardStatus::Ok) return st;
		const size_t kv_head = h / group;
		const int8_t* const k_row_before_rotate =
		    KeyRow(workspace, l, context_cap, num_key_value_heads, head_dim, kv_head, position);
		st = RopeApplySite(k_row_before_rotate, head_dim, position, context_cap, rope_tables,
		                   k_rot.data() + h * head_dim);
		if (st != SslmForwardStatus::Ok) return st;
	}
	for (size_t h = 0; h < num_heads; ++h) {
		const size_t kv_head = h / group;
		int8_t* const k_row =
		    MutableKeyRow(workspace, l, context_cap, num_key_value_heads, head_dim, kv_head, position);
		for (size_t d = 0; d < head_dim; ++d) k_row[d] = k_rot[h * head_dim + d];
	}

	{
		std::vector<int64_t> khead_q_ln2(num_key_value_heads), khead_q_b(num_key_value_heads),
		    khead_q_c(num_key_value_heads);
		std::vector<bool> khead_derived(num_key_value_heads, false);
		std::vector<int64_t> ctx_wide(hidden_size);
		for (size_t h = 0; h < num_heads; ++h) {
			std::vector<int64_t> scores(width), probs(width), ctx_acc(head_dim);
			const size_t kv_head = h / group;

			if (!khead_derived[kv_head]) {
				const int64_t sm_khead_m = lw.iexp_softmax_khead_m[kv_head];
				const int64_t sm_khead_e = lw.iexp_softmax_khead_e[kv_head];
				const CarriedScale sm = CombineCarriedScale(q_scale, CarriedScale{sm_khead_m, sm_khead_e});
				int64_t derived_q_ln2 = 0, derived_q_b = 0, derived_q_c = 0;
				const IExpScaleDomain scale_domain =
				    IExpScaleConstants(sm.m, sm.e, kIExpLn2Q, 30, kIExpBQ, 30, kIExpCaQ, 30,
				                       &derived_q_ln2, &derived_q_b, &derived_q_c);
				if (scale_domain != IExpScaleDomain::kOk) {
					return SslmForwardStatus::IExpScaleDerivationOutOfDomain;
				}
				khead_q_ln2[kv_head] = derived_q_ln2;
				khead_q_b[kv_head] = derived_q_b;
				khead_q_c[kv_head] = derived_q_c;
				khead_derived[kv_head] = true;
			}

			st = CheckSoftmaxRowWidthDomain(khead_q_b[kv_head], khead_q_c[kv_head], width);
			if (st != SslmForwardStatus::Ok) return st;

			const int8_t* const k_rows_base =
			    KeyRow(workspace, l, context_cap, num_key_value_heads, head_dim, kv_head, 0);
			GemmInt8AccumulateRow(q_rot.data() + h * head_dim, k_rows_base, head_dim, width,
			                      scores.data());
			const bool well_formed = SoftmaxRowQ15(scores.data(), width, khead_q_ln2[kv_head],
			                                       khead_q_b[kv_head], khead_q_c[kv_head], probs.data());
			if (!well_formed) return SslmForwardStatus::SoftmaxKernelRefusedAfterGateAccepted;

			if (out_rows != nullptr) {
				CapturedRow row;
				row.layer = l;
				row.query_head = static_cast<uint32_t>(h);
				row.width = static_cast<int64_t>(width);
				row.q_ln2 = khead_q_ln2[kv_head];
				row.q_b = khead_q_b[kv_head];
				row.q_c = khead_q_c[kv_head];
				row.scores = scores;
				row.probs = probs;
				out_rows->push_back(std::move(row));
			}

			const int8_t* const v_rows_base =
			    ValueRow(workspace, l, context_cap, num_key_value_heads, head_dim, kv_head, 0);
			GemmProbQ15Accumulate(probs.data(), v_rows_base, width, head_dim, ctx_acc.data());
			for (size_t d = 0; d < head_dim; ++d) {
				ctx_wide[h * head_dim + d] = ApplyWeightScaleFold(
				    ctx_acc[d], lw.ctx_fold_identity[h], lw.ctx_fold_mult[h], lw.ctx_fold_shift[h]);
			}
		}
		const ChainResult ctx_result =
		    RequantChainChecked(ctx_wide.data(), hidden_size, std::span<const CarriedScale>{},
		                        lw.ctx_fold_site_constant, ctx_codes.data(), &ctx_scale);
		if (ctx_result.status != SslmForwardStatus::Ok) return ctx_result.status;
	}

	st = ProjectAndFunnelCopy(ctx_codes.data(), ctx_scale, lw.o_weight, hidden_size, hidden_size,
	                      lw.o_fold_identity, lw.o_fold_mult, lw.o_fold_shift, lw.o_site_constant,
	                      /*bias=*/nullptr, o_codes.data(), &o_scale);
	if (st != SslmForwardStatus::Ok) return st;

	st = ResidualReconcileSite(o_codes.data(), o_scale, seq.hidden_codes, seq.hidden_scale, hidden_size,
	                           lw.attn_residual_site_constant, attn_stream.data(), &attn_stream_scale);
	if (st != SslmForwardStatus::Ok) return st;

	st = RmsNormSite(attn_stream.data(), lw.mlp_norm_gain, hidden_size, attn_stream_scale,
	                 lw.mlp_norm_site_constant, normed.data(), &mlp_normed_scale);
	if (st != SslmForwardStatus::Ok) return st;

	st = ProjectAndFunnelCopy(normed.data(), mlp_normed_scale, lw.gate_weight, hidden_size,
	                      intermediate_size, lw.gate_fold_identity, lw.gate_fold_mult,
	                      lw.gate_fold_shift, lw.gate_site_constant, /*bias=*/nullptr, gate_codes.data(),
	                      &gate_scale);
	if (st != SslmForwardStatus::Ok) return st;
	st = ProjectAndFunnelCopy(normed.data(), mlp_normed_scale, lw.up_weight, hidden_size, intermediate_size,
	                      lw.up_fold_identity, lw.up_fold_mult, lw.up_fold_shift, lw.up_site_constant,
	                      /*bias=*/nullptr, up_codes.data(), &up_scale);
	if (st != SslmForwardStatus::Ok) return st;

	st = MlpActSite(gate_codes.data(), gate_scale, up_codes.data(), up_scale, intermediate_size,
	                kSiluLutCanonicalTable, lw.mlp_act_site_constant, act_codes.data(), &act_scale);
	if (st != SslmForwardStatus::Ok) return st;

	st = ProjectAndFunnelCopy(act_codes.data(), act_scale, lw.down_weight, intermediate_size, hidden_size,
	                      lw.down_fold_identity, lw.down_fold_mult, lw.down_fold_shift,
	                      lw.down_site_constant, /*bias=*/nullptr, down_codes.data(), &down_scale);
	if (st != SslmForwardStatus::Ok) return st;

	st = ResidualReconcileSite(down_codes.data(), down_scale, attn_stream.data(), attn_stream_scale,
	                           hidden_size, lw.mlp_residual_site_constant, stream_next.data(),
	                           &stream_scale);
	if (st != SslmForwardStatus::Ok) return st;

	for (size_t i = 0; i < hidden_size; ++i) seq.hidden_codes[i] = stream_next[i];
	seq.hidden_scale = stream_scale;
	seq.layer_index = l + 1;
	// Note: this manual replay never advances context_length (RunLayerLoop's own
	// commit point advances it only when layer_index == num_hidden_layers, i.e. only
	// on a token's LAST layer). `manual_seq` is a throwaway copy discarded right
	// after its self-check in main() -- its context_length is never read again, so
	// this omission (relevant only when the checkpoint layer IS the model's last,
	// e.g. layer 27 of 28) changes nothing this program uses.
	return SslmForwardStatus::Ok;
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

	PreflightScanWscFolds(model_view);

	std::vector<LayerBacking> backings(num_hidden_layers);
	std::vector<LayerWeights> layers(num_hidden_layers);
	for (uint32_t l = 0; l < num_hidden_layers; ++l) {
		std::string marshal_err;
		if (!MarshalLayer(model_view, l, num_heads, num_kv_heads, backings[l], layers[l],
		                   &marshal_err)) {
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

	// Prefill: every prompt token except the last, via PRODUCTION RunLayerLoop
	// (full budget) -- exactly RunGreedyDecodeLoop's own composition. Builds real
	// K/V state and a real context_length for the last token's own forward.
	for (size_t i = 0; i + 1 < prompt_tokens.size(); ++i) {
		SslmForwardStatus st = EmbedWholeToken(prompt_tokens[i]);
		if (st != SslmForwardStatus::Ok) {
			std::fprintf(stderr, "FAILED at stage=prefill_embed: position=%zu status=%s\n", i,
			             SslmForwardStatusName(st));
			return 1;
		}
		st = RunLayerLoop(trace_seq, layers.data(), num_hidden_layers,
		                   /*layer_budget=*/num_hidden_layers, hidden_size, head_dim, num_kv_heads,
		                   intermediate_size, context_cap, model_view.rope_tables,
		                   trace_workspace.data(), trace_workspace.size());
		if (st != SslmForwardStatus::Ok) {
			std::fprintf(stderr, "FAILED at stage=prefill_layers: position=%zu status=%s\n", i,
			             SslmForwardStatusName(st));
			return 1;
		}
	}
	{
		const SslmForwardStatus st = EmbedWholeToken(prompt_tokens.back());
		if (st != SslmForwardStatus::Ok) {
			std::fprintf(stderr, "FAILED at stage=last_token_embed: status=%s\n",
			             SslmForwardStatusName(st));
			return 1;
		}
	}

	const int64_t width_at_last_token = trace_seq.context_length + 1;
	std::printf("last prompt token: context_length=%lld width=%lld\n",
	            static_cast<long long>(trace_seq.context_length),
	            static_cast<long long>(width_at_last_token));

	// Checkpoint layers: spread across depth (matches this project's own "8 prompts
	// (5 informative)" scale for D-SLM77's original cell -- 4 layers x 12 heads = 48
	// self-checked rows at this one real width).
	const std::vector<uint32_t> checkpoint_layers = {0, 9, 18, 27};
	std::vector<CapturedRow> all_rows;

	for (uint32_t step = 0; step < num_hidden_layers; ++step) {
		const bool is_checkpoint =
		    std::find(checkpoint_layers.begin(), checkpoint_layers.end(), step) !=
		    checkpoint_layers.end();

		if (is_checkpoint) {
			// Manual replay on an independent copy of the pre-layer state, sharing
			// the SAME workspace (both computations are deterministic functions of
			// the same inputs and land bit-identical K/V; the manual replay runs
			// first, production's own identical write follows and does not change
			// what was already landed).
			std::vector<int8_t> manual_hidden_codes(trace_hidden_codes);
			SequenceLayerState manual_seq = trace_seq;
			manual_seq.hidden_codes = manual_hidden_codes.data();

			std::vector<CapturedRow> step_rows;
			const SslmForwardStatus mst =
			    ManualRunOneLayer(manual_seq, layers[step], hidden_size, head_dim, num_kv_heads,
			                      intermediate_size, context_cap, model_view.rope_tables,
			                      trace_workspace.data(), &step_rows);
			if (mst != SslmForwardStatus::Ok) {
				std::fprintf(stderr, "FAILED at stage=manual_replay: layer=%u status=%s\n", step,
				             SslmForwardStatusName(mst));
				return 1;
			}

			const SslmForwardStatus pst =
			    RunLayerLoop(trace_seq, layers.data(), num_hidden_layers, /*layer_budget=*/1,
			                 hidden_size, head_dim, num_kv_heads, intermediate_size, context_cap,
			                 model_view.rope_tables, trace_workspace.data(), trace_workspace.size());
			if (pst != SslmForwardStatus::Ok) {
				std::fprintf(stderr, "FAILED at stage=production_step: layer=%u status=%s\n", step,
				             SslmForwardStatusName(pst));
				return 1;
			}

			const bool codes_match =
			    std::memcmp(manual_seq.hidden_codes, trace_seq.hidden_codes, hidden_size) == 0;
			const bool scale_match = manual_seq.hidden_scale.m == trace_seq.hidden_scale.m &&
			                         manual_seq.hidden_scale.e == trace_seq.hidden_scale.e;
			if (!codes_match || !scale_match) {
				std::fprintf(stderr,
				             "FAILED at stage=self_check: layer=%u codes_match=%d scale_match=%d -- "
				             "manual replay diverges from production, captured rows discarded, no "
				             "dump written\n",
				             step, codes_match ? 1 : 0, scale_match ? 1 : 0);
				return 1;
			}
			std::printf("self_check: layer=%u manual replay and production agree bit-for-bit "
			            "(%zu rows captured)\n",
			            step, step_rows.size());
			for (auto& row : step_rows) all_rows.push_back(std::move(row));
		} else {
			const SslmForwardStatus pst =
			    RunLayerLoop(trace_seq, layers.data(), num_hidden_layers, /*layer_budget=*/1,
			                 hidden_size, head_dim, num_kv_heads, intermediate_size, context_cap,
			                 model_view.rope_tables, trace_workspace.data(), trace_workspace.size());
			if (pst != SslmForwardStatus::Ok) {
				std::fprintf(stderr, "FAILED at stage=production_step: layer=%u status=%s\n", step,
				             SslmForwardStatusName(pst));
				return 1;
			}
		}
	}

	// Dump: text, one row per line -- layer, head, width, q_ln2, q_b, q_c, then
	// width scores and width probs, space-separated. Analysis (ideal-softmax
	// comparison) is a separate script per this project's own convention
	// (t1740_pooled_fidelity_report.py's own split from the capture tool).
	std::ofstream f(dump_path);
	if (!f) {
		std::fprintf(stderr, "FAILED at stage=dump_open: could not open \"%s\"\n", dump_path.c_str());
		return 1;
	}
	f << all_rows.size() << "\n";
	for (const auto& row : all_rows) {
		f << row.layer << " " << row.query_head << " " << row.width << " " << row.q_ln2 << " "
		  << row.q_b << " " << row.q_c << "\n";
		for (int64_t k = 0; k < row.width; ++k) f << row.scores[k] << (k + 1 < row.width ? ' ' : '\n');
		for (int64_t k = 0; k < row.width; ++k) f << row.probs[k] << (k + 1 < row.width ? ' ' : '\n');
	}
	std::printf("dump written: %s (%zu self-checked softmax rows, width=%lld, layers={0,9,18,27})\n",
	            dump_path.c_str(), all_rows.size(), static_cast<long long>(width_at_last_token));
	return 0;
}
