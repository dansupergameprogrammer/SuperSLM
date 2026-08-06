// t1786_stage_dump_probe.cpp -- T-1786: dump the compiled engine's REAL Q15 attention
// probabilities, self-checked bit-for-bit against production (identical discipline to T-1778's
// tools/t1778_engine_attn_probe.cpp, branch claude/t1778-float-attn-reference@5df48fc, read in
// full and copied verbatim below with ONE addition: per (layer,head) this probe ALSO dumps the
// row's own RAW post-GEMM scores (the exact `scores` array SoftmaxRowQ15 is called on, before
// ShiftByMax) and the per-(layer,kv-head) derived (q_ln2, q_b, q_c) triple SoftmaxRowQ15
// consumes. Nothing else changes: the manual replay, the self-check discipline, and the
// checkpoint-layer convention are unmodified from T-1778's own probe.
//
// WHY the extra fields, and why this is not the T-1776/T-1769 mistake: T-1786's own commission
// is to attribute the measured probability-space TVD (T-1778 D-SLM1234-1240, 0.155-0.204,
// against the FIXED, independent float32 reference T-1778 built) across the stages of
// SoftmaxRowQ15's own construction. That requires, per row, the exact inputs each stage
// consumes (scores, q_ln2, q_b, q_c) so a SEPARATE, offline analysis script can replay the
// SAME row bit-for-bit and then substitute float-exact arithmetic for ONE stage at a time,
// holding the others at these exact captured production values. The float reference this is
// graded against (T-1786's own copy of T-1778's float instrument, tools/t1786_float_attn_
// reference.py) takes NONE of these dumped values as input -- it is the checkpoint's own
// stock HuggingFace forward, computed and dumped independently, before this probe's own dump
// is ever read. Dumping q_ln2/q_b/q_c here does not feed them into the reference; it only lets
// the offline analysis reproduce the REAL arm exactly before perturbing one stage of it.
//
// Self-check identical in kind and unmodified from T-1778: at every checkpoint layer, the
// manual replay's own resulting hidden_codes/hidden_scale is compared bit-for-bit against
// PRODUCTION's own RunLayerLoop(layer_budget=1) call for that layer, from the same pre-layer
// state. Both `probs` and the newly-added `scores`/`(q_ln2,q_b,q_c)` are validated
// transitively, the same argument T-1778 uses for `probs`: they are read directly from the
// identical compiled SoftmaxRowQ15 call (and its own IExpScaleConstants call immediately
// before it) at the identical point in the call sequence production's own RunLayerLoop uses,
// and the replay's LATER output (hidden_codes) is proven bit-identical to production.

#include <algorithm>
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

// -- verbatim copies of forward_sites.cpp's internal-linkage helpers, identical to T-1778's
// own probe (ProjectAndFunnel/ApplyBiasReconcileRow are internal-linkage in
// forward_sites.cpp; bodies copied verbatim, calling only public-header primitives). --
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

// One captured (layer, head) softmax row: the REAL arm's raw scores (SoftmaxRowQ15's own
// `scores` argument, pre-ShiftByMax), its own derived (q_ln2, q_b, q_c) triple (shared by
// every head of the same kv-head), and the REAL arm's own Q15 probabilities exactly as
// SoftmaxRowQ15 emits them.
struct CapturedProbRow {
	uint32_t layer = 0;
	uint32_t head = 0;
	int64_t width = 0;
	int64_t q_ln2 = 0;
	int64_t q_b = 0;
	int64_t q_c = 0;
	std::vector<int64_t> scores;  // raw post-GEMM, pre-ShiftByMax
	std::vector<int64_t> probs;   // Q15, sums to 32768 (or within rounding of it)
};

void PrintUsage(const char* argv0) {
	std::fprintf(stderr,
	             "usage: %s <model.sslm> <tokenizer.sslm> \"<prompt>\" --dump <path>\n", argv0);
}

// Manual replay of exactly ONE layer's forward through the softmax, capturing every query
// head's REAL Q15 probs plus its raw scores and derived (q_ln2,q_b,q_c) (CapturedProbRow), and
// continuing the REAL arm's computation (unchanged from RunLayerLoop's own body) so
// `seq`/`workspace` advance identically to production -- required for the self-check below.
SslmForwardStatus ManualRunOneLayer(SequenceLayerState& seq, const LayerWeights& lw,
                                     size_t hidden_size, size_t head_dim, size_t num_key_value_heads,
                                     size_t intermediate_size, int64_t context_cap,
                                     const SslmTensorManifest& rope_tables, uint8_t* workspace,
                                     std::vector<CapturedProbRow>* out_rows) {
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

	std::vector<int64_t> ctx_wide(hidden_size);
	{
		std::vector<int64_t> khead_q_ln2(num_key_value_heads), khead_q_b(num_key_value_heads),
		    khead_q_c(num_key_value_heads);
		std::vector<bool> khead_derived(num_key_value_heads, false);
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
				CapturedProbRow row;
				row.layer = l;
				row.head = static_cast<uint32_t>(h);
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

	// --- REAL arm continues, identical to RunLayerLoop's own remaining body -- this is
	// what the self-check validates, which transitively validates `probs`/`scores`/`q_ln2`
	// etc. above. ---
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

	// REAL arm advances seq (used for the self-check against production).
	for (size_t i = 0; i < hidden_size; ++i) seq.hidden_codes[i] = stream_next[i];
	seq.hidden_scale = stream_scale;
	seq.layer_index = l + 1;
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

	// Same checkpoint-layer convention as T-1762/T-1763/T-1768/T-1778: {0,9,18,27} is
	// D-SLM1135-1137's own cell; {1,2} added (T-1763's own extension) to distinguish "layer 0
	// specifically" from "the first few layers generally".
	const std::vector<uint32_t> checkpoint_layers = {0, 1, 2, 9, 18, 27};
	std::vector<CapturedProbRow> all_rows;

	for (uint32_t step = 0; step < num_hidden_layers; ++step) {
		const bool is_checkpoint =
		    std::find(checkpoint_layers.begin(), checkpoint_layers.end(), step) !=
		    checkpoint_layers.end();

		if (is_checkpoint) {
			std::vector<int8_t> manual_hidden_codes(trace_hidden_codes);
			SequenceLayerState manual_seq = trace_seq;
			manual_seq.hidden_codes = manual_hidden_codes.data();

			std::vector<CapturedProbRow> step_rows;
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

	// Dump: layer head width q_ln2 q_b q_c scores[width] probs[width], one row per
	// (layer,head), text format for easy cross-language re-verification (matches this
	// campaign's own scratch-tool convention).
	std::ofstream f(dump_path);
	if (!f) {
		std::fprintf(stderr, "FAILED at stage=dump_open: could not open \"%s\"\n", dump_path.c_str());
		return 1;
	}
	f << all_rows.size() << " " << width_at_last_token << "\n";
	for (const auto& row : all_rows) {
		f << row.layer << " " << row.head << " " << row.width << " " << row.q_ln2 << " "
		  << row.q_b << " " << row.q_c;
		for (int64_t s : row.scores) f << " " << s;
		for (int64_t p : row.probs) f << " " << p;
		f << "\n";
	}
	std::printf("\ndump written: %s (%zu softmax rows, width=%lld, layers={0,1,2,9,18,27})\n",
	            dump_path.c_str(), all_rows.size(), static_cast<long long>(width_at_last_token));
	return 0;
}
