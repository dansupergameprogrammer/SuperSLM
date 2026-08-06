// t1768_qln2_derivation_probe.cpp -- T-1768: establish where layer 0's derived `q_ln2 = 6`
// (T-1763, D-SLM1146) comes from, at source and by execution -- is it a property of layer 0's
// own score statistics feeding a correct rule, or a rule degenerating on layer 0's inputs.
//
// Extends T-1763's own self-checked instrument (tools/t1763_layer0_diagnosis_probe.cpp,
// branch claude/t1763-layer0-diagnosis@10cd1e2, read not owned -- this is an independent copy,
// not a commit to that branch) with ONE addition: a ChainDiag capture at the exact point
// T-1763's own instrument already calls the production derivation function,
// `IExpScaleConstants` (forward_sites.cpp:1362-1364) -- recording every intermediate the
// derivation actually consumes (normed_scale, q_scale, the per-kv-head baked-in softmax
// carried scale, their CombineCarriedScale product, and DynamicScaleReciprocal's own r_m)
// so the 20-200x gap between layer 0's q_ln2=6 and every other checkpointed layer's q_ln2 can
// be attributed to a specific upstream quantity by execution, not inferred from the output
// value alone.
//
// T-1763's own three checked extensions, unchanged here (see that file's own header for the
// full description this comment does not repeat):
//
//   1. Checkpoint layers {0,1,2,9,18,27} instead of T-1762's {0,9,18,27} -- distinguishes
//      "layer 0 specifically" from "the first few layers generally" (T-1762 could not,
//      its nearest checkpoint to 0 was layer 9).
//   2. Per-row diagnostics captured alongside the existing (q_ln2,q_b,q_c,scores,probs)
//      tuple: score dynamic range (max-min over the row, i.e. the ShiftByMax input range),
//      and M = q_b*q_b + q_c (SoftmaxRowQ15's own per-element peak bound, intmath.h
//      lines 620-648) alongside the row's own actual peak evaluated exponent, to check
//      whether the i-exp construction's domain is being approached at layer 0.
//   3. Pre-layer input code distribution (max |code|, mean |code|, saturated-at-127 count)
//      captured on trace_seq.hidden_codes BEFORE each checkpoint layer's forward runs --
//      distinguishes "layer 0 reads raw embeddings" from "layer 0 reads a residual stream
//      like every other layer" by measuring the actual distribution, not assuming one.
//   4. Downstream propagation: BOTH arms (real Q15 softmax, ideal double-precision softmax)
//      are carried all the way through the REST of the same layer's forward (o_proj,
//      attention residual, mlp norm, gate/up/down, mlp residual) using the identical
//      downstream weights/site-constants for both arms -- producing two full post-layer
//      hidden_codes/hidden_scale arrays that answer whether layer 0's attn_ctx code
//      disagreement propagates into the residual stream or is absorbed by it.
//
// Self-check identical in kind to T-1762/T-1755: at every checkpoint layer, the REAL arm's
// manually-replayed hidden_codes/scale is compared bit-for-bit against PRODUCTION's own
// RunLayerLoop(layer_budget=1) call for that layer, from the same pre-layer state. Only
// self-checked layers are reported; a mismatch aborts with no report written. The IDEAL
// arm has no production counterpart (production never computes an ideal-softmax-routed
// layer) -- its correctness rests on sharing every downstream function call, in the same
// order, with the self-checked real arm; the only place the two arms' inputs differ is
// which probability vector fed the V-matmul, exactly as T-1762 argues for its own ideal
// ctx_codes arm.
//
// ProjectAndFunnel and ApplyBiasReconcileRow are internal-linkage in forward_sites.cpp
// (T-1762's own finding, re-confirmed by re-reading forward_sites.cpp at this worktree's
// HEAD) -- their bodies are copied verbatim below, calling only public-header primitives,
// never a re-derivation of their arithmetic. Identical copies to T-1762's; not shared by
// build because this file lives on an independent branch.

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

// -- verbatim copies of intmath.cpp's internal-linkage helpers `IExpScaleConstants` itself
// uses to form `shift_ln2` (src/intmath.cpp:678-690, `namespace {}` scope -- not externally
// callable, hence copied rather than declared). Reproduced so T-1768's diagnostic can report
// the EXACT `shift_ln2` the production derivation forms, not a re-derivation from a different
// formula. `IExpScaleConstants`'s own Step 2 (src/intmath.cpp:703-708) is
// `SaturatingAdd64ForIExpScale(SaturatingAdd64ForIExpScale(ln2_fmt, 62), e)`; this copy is
// called the identical way below.
inline bool AddOverflows64ForIExpScaleCopy(int64_t a, int64_t b, int64_t* out) {
	const uint64_t ua = static_cast<uint64_t>(a);
	const uint64_t ub = static_cast<uint64_t>(b);
	*out = static_cast<int64_t>(ua + ub);
	return (a >= 0) == (b >= 0) && (*out >= 0) != (a >= 0);
}
inline int64_t SaturatingAdd64ForIExpScaleCopy(int64_t a, int64_t b) {
	int64_t out;
	if (!AddOverflows64ForIExpScaleCopy(a, b, &out)) return out;
	return (a >= 0) ? INT64_MAX : INT64_MIN;
}

// -- verbatim copies of forward_sites.cpp's internal-linkage helpers (see file header) --
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

constexpr int64_t kIExpClipN = 30;

// Ideal (double-precision) softmax probability for one row, rounded to Q15 by
// round-to-nearest -- identical derivation to T-1762's IdealProbsQ15 (itself identical to
// T-1755's ideal_probs), grounded at SoftmaxRowQ15's own composition comment and
// IExpConstruct's own decomposition (intmath.h), unchanged here.
void IdealProbsQ15(const std::vector<int64_t>& scores, int64_t q_ln2, std::vector<int64_t>* out_q15) {
	const size_t width = scores.size();
	int64_t mx = scores[0];
	for (size_t k = 1; k < width; ++k) mx = std::max(mx, scores[k]);
	const int64_t clip = -kIExpClipN * q_ln2;
	std::vector<double> real_x(width);
	for (size_t k = 0; k < width; ++k) {
		const int64_t shifted = scores[k] - mx;
		const int64_t clipped = std::max<int64_t>(shifted, clip);
		real_x[k] = static_cast<double>(clipped) * (std::log(2.0) / static_cast<double>(q_ln2));
	}
	std::vector<double> exps(width);
	double total = 0.0;
	for (size_t k = 0; k < width; ++k) {
		exps[k] = std::exp(real_x[k]);
		total += exps[k];
	}
	out_q15->resize(width);
	for (size_t k = 0; k < width; ++k) {
		const double p = exps[k] / total;
		(*out_q15)[k] = static_cast<int64_t>(std::llround(p * 32768.0));
	}
}

// Per-(layer,head) softmax-row diagnostics -- candidates named in the T-1763 brief:
// dynamic range of the row (max-min score, ShiftByMax's own input range), the derived
// (q_ln2,q_b,q_c) triple, and how close the row's own peak evaluated exponent comes to
// SoftmaxRowQ15's own per-element domain bound M = q_b*q_b + q_c (intmath.h 620-648).
// NOTE on the i-exp domain-proximity candidate (T-1763 brief item 1): ShiftByMax's own
// construction guarantees the row's argmax element has shifted=0, hence clipped=0, z=0,
// q_p=0, base=q_p+q_b=q_b, evaluated=base^2+q_c=q_b^2+q_c=M EXACTLY -- the argmax element
// of every row, at every layer, sits exactly at SoftmaxRowQ15's own per-element domain
// ceiling M by construction (intmath.h 620-648's own bound). This is an algebraic identity
// of ShiftByMax + IExpConstruct, not something that varies per row or per layer, so it
// cannot by itself distinguish layer 0 from any other layer -- recorded here as a ruled-out
// candidate rather than measured as a per-row variable.
struct RowDiag {
	uint32_t layer = 0;
	uint32_t head = 0;
	int64_t width = 0;
	int64_t q_ln2 = 0, q_b = 0, q_c = 0;
	int64_t score_range = 0;   // max(scores) - min(scores), i.e. ShiftByMax's own input range
	double m_bound = 0.0;      // q_b*q_b + q_c, as double -- constant per (layer,head) triple,
	                            // recorded for completeness, not diagnostic across layers (see
	                            // note above)
	// Domain-clip crowding: IExpConstruct clamps `shifted` (score - max) to
	// `clip = -kIExpClipN*q_ln2` (kIExpClipN=30, this file and intmath.h). A SMALL q_ln2
	// makes that clip window narrow in raw score units; how much of the row's own
	// score_range falls inside vs outside that window is a directly measurable candidate
	// for why a small q_ln2 might correlate with construction disagreement, tested (not
	// assumed) below.
	int64_t clip_window = 0;      // kIExpClipN * q_ln2, in raw score units
	int64_t clipped_elements = 0;  // count of this row's `width` elements with shifted < -clip_window
};

// T-1768 item 5c: one row's own peakedness -- the IDEAL arm's largest Q15 probability in the
// row (top1 share of 32768) and how many of the row's unclamped elements individually carry at
// least 1 Q15 unit of ideal probability ("non-negligible" elements) -- tests whether layer 0's
// elevated per-element delta (item 5/5b) coincides with a FLATTER row distribution (more mass
// spread across more non-negligible elements, so normalization is more sensitive to each
// element's own approximation error) rather than a peaked one dominated by 1-2 terms.
struct RowPeakDiag {
	uint32_t layer = 0;
	int64_t ideal_top1_q15 = 0;
	int64_t nonneg_unclamped = 0;  // unclamped elements with ideal_q15 >= 1
	int64_t unclamped_total = 0;
};

// T-1768 item 5 (coordinator addendum, D-SLM1159's open item): per-UNCLAMPED-element capture
// of the octave shift `z` (the same `z = -clipped / q_ln2` IdealProbsQ15/IExpEvaluate compute,
// re-derived here from the identical formula rather than read from a private
// IExpConstruction field) alongside the real/ideal Q15 probability delta at that element --
// tests, by execution, whether the T-1769/Popper-measured unclamped-element disagreement
// (mean |delta| 21.2 at layer 0 vs 0.9-1.2 elsewhere) is a function of z (how many octaves an
// element sits below the row's own max), which is the mechanism candidate D-SLM1159 names.
struct ElemZDiag {
	uint32_t layer = 0;
	int64_t z = 0;
	int64_t real_q15 = 0, ideal_q15 = 0, abs_delta = 0;
};

// T-1768: the derivation chain `IExpScaleConstants` actually consumes, captured at the exact
// production call site (forward_sites.cpp:1362-1364, mirrored by T-1763's own instrument at
// the site this file inherits) -- one entry per (layer, kv_head), captured once per kv_head
// per layer (matching production's own once-per-distinct-kv_head derivation, §4.5 step 2a/b/c
// in forward_sites.cpp). Every field here is a value the derivation reads or an intermediate
// intmath.cpp's own IExpScaleConstants computes (src/intmath.cpp:694-770), not a value
// reconstructed from q_ln2 after the fact.
struct ChainDiag {
	uint32_t layer = 0;
	uint32_t kv_head = 0;
	int64_t normed_scale_m = 0, normed_scale_e = 0;   // RmsNormSite's own output carried scale
	int64_t q_scale_m = 0, q_scale_e = 0;             // ProjectAndFunnel(q_proj)'s own output carried scale
	int64_t khead_m = 0, khead_e = 0;                 // artifact-baked lw.iexp_softmax_khead_m/e[kv_head]
	int64_t sm_m = 0, sm_e = 0;                       // CombineCarriedScale(q_scale, khead) -- IExpScaleConstants' own (m, e) args
	int64_t r_m = 0;                                  // DynamicScaleReciprocal(sm_m) -- intmath.cpp step 3
	int64_t shift_ln2 = 0;                            // SaturatingAdd64ForIExpScale(SaturatingAdd64ForIExpScale(30, 62), sm_e) -- intmath.cpp step 2
	int64_t q_ln2 = 0;                                // the derivation's own output (cross-check against RowDiag's copy)
};

// Per pre-layer-input diagnostic: distribution of the hidden_codes vector the checkpoint
// layer's RmsNorm actually consumes (raw embeddings at layer 0; a residual-stream output
// at every other layer, by construction -- measured here rather than assumed) AND the
// value-side (V) codes landed for this layer's own KV rows at the current position --
// tests the "value-side scale" candidate the T-1763 brief names, directly.
struct InputDiag {
	uint32_t layer = 0;
	int64_t max_abs_code = 0;
	double mean_abs_code = 0.0;
	int64_t saturated_count = 0;  // codes at exactly +-127 (int8 rail)
	size_t hidden_size = 0;
	int64_t v_max_abs_code = 0;
	double v_mean_abs_code = 0.0;
};

// Full-layer, both-arms downstream propagation result: the REAL arm's ctx_codes and the
// IDEAL arm's ctx_codes (T-1762's own metric, unchanged), PLUS both arms carried through
// o_proj -> attention residual -> mlp norm -> gate/up/down -> mlp residual, using the
// IDENTICAL downstream weights and site constants for both arms -- the only difference
// between the two final hidden_codes arrays is which probability vector fed the V-matmul
// at this layer. Answers "does layer 0's attn_ctx disagreement propagate or get absorbed."
struct CapturedFullRow {
	uint32_t layer = 0;
	std::vector<int8_t> real_ctx_codes, ideal_ctx_codes;      // hidden_size each (T-1762 metric)
	std::vector<int8_t> real_out_codes, ideal_out_codes;      // hidden_size each, post-layer
	CarriedScale real_out_scale{}, ideal_out_scale{};
};

void PrintUsage(const char* argv0) {
	std::fprintf(stderr,
	             "usage: %s <model.sslm> <tokenizer.sslm> \"<prompt>\" --dump <path>\n", argv0);
}

// Manual replay of exactly ONE layer's forward, capturing:
//  - every query head's softmax row diagnostics (RowDiag)
//  - the pre-layer input distribution (InputDiag)
//  - both arms' ctx_codes and full downstream layer output (CapturedFullRow)
// `seq`/`workspace` are advanced in place by the REAL arm, exactly as
// RunLayerLoop(layer_budget=1) would (used for the self-check); the IDEAL arm's downstream
// pass is computed into separate buffers and does not touch `seq`/`workspace`.
SslmForwardStatus ManualRunOneLayer(SequenceLayerState& seq, const LayerWeights& lw,
                                     size_t hidden_size, size_t head_dim, size_t num_key_value_heads,
                                     size_t intermediate_size, int64_t context_cap,
                                     const SslmTensorManifest& rope_tables, uint8_t* workspace,
                                     std::vector<RowDiag>* out_rows, InputDiag* out_input_diag,
                                     CapturedFullRow* out_full_row,
                                     std::vector<ChainDiag>* out_chain,
                                     std::vector<ElemZDiag>* out_elem_z,
                                     std::vector<RowPeakDiag>* out_peak) {
	const size_t num_heads = hidden_size / head_dim;
	const size_t group = num_heads / num_key_value_heads;
	const int64_t position = seq.context_length;
	const size_t width = static_cast<size_t>(seq.context_length) + 1;
	const uint32_t l = seq.layer_index;

	// Pre-layer input distribution -- captured BEFORE any computation touches hidden_codes.
	if (out_input_diag != nullptr) {
		out_input_diag->layer = l;
		out_input_diag->hidden_size = hidden_size;
		int64_t max_abs = 0, sat = 0;
		double sum_abs = 0.0;
		for (size_t i = 0; i < hidden_size; ++i) {
			const int64_t v = seq.hidden_codes[i];
			const int64_t av = v < 0 ? -v : v;
			max_abs = std::max(max_abs, av);
			sum_abs += static_cast<double>(av);
			if (v == 127 || v == -127) ++sat;
		}
		out_input_diag->max_abs_code = max_abs;
		out_input_diag->mean_abs_code = sum_abs / static_cast<double>(hidden_size);
		out_input_diag->saturated_count = sat;
	}

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

	// Value-side scale candidate (T-1763 brief item 1): mean/max |code| of kv_head 0's
	// landed V rows over the row's own causal width, measured directly rather than assumed.
	if (out_input_diag != nullptr) {
		const int8_t* const v_rows_base =
		    ValueRow(workspace, l, context_cap, num_key_value_heads, head_dim, /*kv_head=*/0, 0);
		int64_t v_max_abs = 0;
		double v_sum_abs = 0.0;
		for (size_t p = 0; p < width; ++p) {
			for (size_t d = 0; d < head_dim; ++d) {
				const int64_t v = v_rows_base[p * head_dim + d];
				const int64_t av = v < 0 ? -v : v;
				v_max_abs = std::max(v_max_abs, av);
				v_sum_abs += static_cast<double>(av);
			}
		}
		out_input_diag->v_max_abs_code = v_max_abs;
		out_input_diag->v_mean_abs_code = v_sum_abs / static_cast<double>(width * head_dim);
	}

	std::vector<int64_t> ctx_wide(hidden_size), ctx_wide_ideal(hidden_size);
	{
		std::vector<int64_t> khead_q_ln2(num_key_value_heads), khead_q_b(num_key_value_heads),
		    khead_q_c(num_key_value_heads);
		std::vector<bool> khead_derived(num_key_value_heads, false);
		for (size_t h = 0; h < num_heads; ++h) {
			std::vector<int64_t> scores(width), probs(width), ctx_acc(head_dim);
			std::vector<int64_t> ideal_probs_q15, ctx_acc_ideal(head_dim);
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

				if (out_chain != nullptr) {
					ChainDiag cd;
					cd.layer = l;
					cd.kv_head = static_cast<uint32_t>(kv_head);
					cd.normed_scale_m = normed_scale.m;
					cd.normed_scale_e = normed_scale.e;
					cd.q_scale_m = q_scale.m;
					cd.q_scale_e = q_scale.e;
					cd.khead_m = sm_khead_m;
					cd.khead_e = sm_khead_e;
					cd.sm_m = sm.m;
					cd.sm_e = sm.e;
					// DynamicScaleReciprocal is a public intmath.h primitive
					// (declared line 123) -- called directly, not re-derived; the
					// SAME function IExpScaleConstants itself calls at its own
					// Step 3 (src/intmath.cpp:723) on this same `sm.m`.
					cd.r_m = DynamicScaleReciprocal(sm.m);
					// shift_ln2, via the verbatim-copied saturating-add helper above,
					// called the identical way IExpScaleConstants's own Step 2 does
					// (src/intmath.cpp:707-708): ln2_fmt is always 30 at this
					// production call site (forward_sites.cpp:1362-1364,
					// kIExpLn2Q's own format argument).
					cd.shift_ln2 = SaturatingAdd64ForIExpScaleCopy(
					    SaturatingAdd64ForIExpScaleCopy(static_cast<int64_t>(30), 62), sm.e);
					cd.q_ln2 = derived_q_ln2;
					out_chain->push_back(cd);
				}
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
				RowDiag rd;
				rd.layer = l;
				rd.head = static_cast<uint32_t>(h);
				rd.width = static_cast<int64_t>(width);
				rd.q_ln2 = khead_q_ln2[kv_head];
				rd.q_b = khead_q_b[kv_head];
				rd.q_c = khead_q_c[kv_head];
				int64_t smin = scores[0], smax = scores[0];
				for (size_t k = 1; k < width; ++k) {
					smin = std::min(smin, scores[k]);
					smax = std::max(smax, scores[k]);
				}
				rd.score_range = smax - smin;
				const double qb_d = static_cast<double>(khead_q_b[kv_head]);
				const double qc_d = static_cast<double>(khead_q_c[kv_head]);
				rd.m_bound = qb_d * qb_d + qc_d;
				rd.clip_window = kIExpClipN * khead_q_ln2[kv_head];
				int64_t clipped = 0;
				for (size_t k = 0; k < width; ++k) {
					if ((smax - scores[k]) > rd.clip_window) ++clipped;
				}
				rd.clipped_elements = clipped;
				out_rows->push_back(rd);
			}

			const int8_t* const v_rows_base =
			    ValueRow(workspace, l, context_cap, num_key_value_heads, head_dim, kv_head, 0);
			GemmProbQ15Accumulate(probs.data(), v_rows_base, width, head_dim, ctx_acc.data());
			for (size_t d = 0; d < head_dim; ++d) {
				ctx_wide[h * head_dim + d] = ApplyWeightScaleFold(
				    ctx_acc[d], lw.ctx_fold_identity[h], lw.ctx_fold_mult[h], lw.ctx_fold_shift[h]);
			}

			IdealProbsQ15(scores, khead_q_ln2[kv_head], &ideal_probs_q15);

			if (out_elem_z != nullptr) {
				// Same z/clip formula as IdealProbsQ15 (this file, above) and IExpEvaluate's
				// own decomposition (intmath.h:288-290): z = -clipped / q_ln2, clipped =
				// max(shifted, -kIExpClipN*q_ln2), shifted = score - row max. clipped <= 0
				// always, so -clipped >= 0 and C++ integer division truncates == floors here
				// (no negative-operand ambiguity).
				int64_t mx = scores[0];
				for (size_t k = 1; k < width; ++k) mx = std::max(mx, scores[k]);
				const int64_t q_ln2_h = khead_q_ln2[kv_head];
				const int64_t clip = -kIExpClipN * q_ln2_h;
				for (size_t k = 0; k < width; ++k) {
					const int64_t shifted = scores[k] - mx;
					const bool is_clamped = shifted < clip;
					if (is_clamped) continue;  // excluded -- Popper's own population split (T-1769)
					const int64_t clipped = std::max<int64_t>(shifted, clip);
					const int64_t z = (-clipped) / q_ln2_h;
					ElemZDiag ez;
					ez.layer = l;
					ez.z = z;
					ez.real_q15 = probs[k];
					ez.ideal_q15 = ideal_probs_q15[k];
					ez.abs_delta = std::abs(probs[k] - ideal_probs_q15[k]);
					out_elem_z->push_back(ez);
				}
			}

			if (out_peak != nullptr) {
				RowPeakDiag pk;
				pk.layer = l;
				int64_t top1 = 0, nonneg = 0, unclamped_n = 0;
				int64_t mx = scores[0];
				for (size_t k = 1; k < width; ++k) mx = std::max(mx, scores[k]);
				const int64_t q_ln2_h = khead_q_ln2[kv_head];
				const int64_t clip = -kIExpClipN * q_ln2_h;
				for (size_t k = 0; k < width; ++k) {
					top1 = std::max(top1, ideal_probs_q15[k]);
					if ((scores[k] - mx) >= clip) {
						++unclamped_n;
						if (ideal_probs_q15[k] >= 1) ++nonneg;
					}
				}
				pk.ideal_top1_q15 = top1;
				pk.nonneg_unclamped = nonneg;
				pk.unclamped_total = unclamped_n;
				out_peak->push_back(pk);
			}

			GemmProbQ15Accumulate(ideal_probs_q15.data(), v_rows_base, width, head_dim,
			                      ctx_acc_ideal.data());
			for (size_t d = 0; d < head_dim; ++d) {
				ctx_wide_ideal[h * head_dim + d] = ApplyWeightScaleFold(
				    ctx_acc_ideal[d], lw.ctx_fold_identity[h], lw.ctx_fold_mult[h], lw.ctx_fold_shift[h]);
			}
		}
		const ChainResult ctx_result =
		    RequantChainChecked(ctx_wide.data(), hidden_size, std::span<const CarriedScale>{},
		                        lw.ctx_fold_site_constant, ctx_codes.data(), &ctx_scale);
		if (ctx_result.status != SslmForwardStatus::Ok) return ctx_result.status;
	}

	// --- REAL arm: o_proj -> attn residual -> mlp -> mlp residual (identical to
	// RunLayerLoop's own remaining body; this is the arm the self-check validates). ---
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

	// --- IDEAL arm downstream: SAME weights, SAME site constants, SAME every call as the
	// real arm above -- the only substituted input across the whole layer is ctx_wide_ideal
	// in place of ctx_wide (i.e. the ideal-softmax-routed attn_ctx). Computed into separate
	// buffers; does not touch seq/workspace. ---
	std::vector<int8_t> ideal_ctx_codes(hidden_size), ideal_o_codes(hidden_size);
	std::vector<int8_t> ideal_normed(hidden_size), ideal_attn_stream(hidden_size);
	std::vector<int8_t> ideal_gate_codes(intermediate_size), ideal_up_codes(intermediate_size);
	std::vector<int8_t> ideal_act_codes(intermediate_size), ideal_down_codes(hidden_size);
	std::vector<int8_t> ideal_stream_next(hidden_size);
	CarriedScale ideal_ctx_scale{}, ideal_o_scale{}, ideal_attn_stream_scale{};
	CarriedScale ideal_mlp_normed_scale{}, ideal_gate_scale{}, ideal_up_scale{}, ideal_act_scale{},
	    ideal_down_scale{}, ideal_stream_scale{};

	const ChainResult ideal_ctx_result = RequantChainChecked(
	    ctx_wide_ideal.data(), hidden_size, std::span<const CarriedScale>{}, lw.ctx_fold_site_constant,
	    ideal_ctx_codes.data(), &ideal_ctx_scale);
	if (ideal_ctx_result.status != SslmForwardStatus::Ok) return ideal_ctx_result.status;

	st = ProjectAndFunnelCopy(ideal_ctx_codes.data(), ideal_ctx_scale, lw.o_weight, hidden_size,
	                      hidden_size, lw.o_fold_identity, lw.o_fold_mult, lw.o_fold_shift,
	                      lw.o_site_constant, /*bias=*/nullptr, ideal_o_codes.data(), &ideal_o_scale);
	if (st != SslmForwardStatus::Ok) return st;

	st = ResidualReconcileSite(ideal_o_codes.data(), ideal_o_scale, seq.hidden_codes, seq.hidden_scale,
	                           hidden_size, lw.attn_residual_site_constant, ideal_attn_stream.data(),
	                           &ideal_attn_stream_scale);
	if (st != SslmForwardStatus::Ok) return st;

	st = RmsNormSite(ideal_attn_stream.data(), lw.mlp_norm_gain, hidden_size, ideal_attn_stream_scale,
	                 lw.mlp_norm_site_constant, ideal_normed.data(), &ideal_mlp_normed_scale);
	if (st != SslmForwardStatus::Ok) return st;

	st = ProjectAndFunnelCopy(ideal_normed.data(), ideal_mlp_normed_scale, lw.gate_weight, hidden_size,
	                      intermediate_size, lw.gate_fold_identity, lw.gate_fold_mult,
	                      lw.gate_fold_shift, lw.gate_site_constant, /*bias=*/nullptr,
	                      ideal_gate_codes.data(), &ideal_gate_scale);
	if (st != SslmForwardStatus::Ok) return st;
	st = ProjectAndFunnelCopy(ideal_normed.data(), ideal_mlp_normed_scale, lw.up_weight, hidden_size,
	                      intermediate_size, lw.up_fold_identity, lw.up_fold_mult, lw.up_fold_shift,
	                      lw.up_site_constant, /*bias=*/nullptr, ideal_up_codes.data(), &ideal_up_scale);
	if (st != SslmForwardStatus::Ok) return st;

	st = MlpActSite(ideal_gate_codes.data(), ideal_gate_scale, ideal_up_codes.data(), ideal_up_scale,
	                intermediate_size, kSiluLutCanonicalTable, lw.mlp_act_site_constant,
	                ideal_act_codes.data(), &ideal_act_scale);
	if (st != SslmForwardStatus::Ok) return st;

	st = ProjectAndFunnelCopy(ideal_act_codes.data(), ideal_act_scale, lw.down_weight, intermediate_size,
	                      hidden_size, lw.down_fold_identity, lw.down_fold_mult, lw.down_fold_shift,
	                      lw.down_site_constant, /*bias=*/nullptr, ideal_down_codes.data(),
	                      &ideal_down_scale);
	if (st != SslmForwardStatus::Ok) return st;

	st = ResidualReconcileSite(ideal_down_codes.data(), ideal_down_scale, ideal_attn_stream.data(),
	                           ideal_attn_stream_scale, hidden_size, lw.mlp_residual_site_constant,
	                           ideal_stream_next.data(), &ideal_stream_scale);
	if (st != SslmForwardStatus::Ok) return st;

	if (out_full_row != nullptr) {
		out_full_row->layer = l;
		out_full_row->real_ctx_codes.assign(ctx_codes.begin(), ctx_codes.end());
		out_full_row->ideal_ctx_codes = std::move(ideal_ctx_codes);
		out_full_row->real_out_codes.assign(stream_next.begin(), stream_next.end());
		out_full_row->ideal_out_codes = std::move(ideal_stream_next);
		out_full_row->real_out_scale = stream_scale;
		out_full_row->ideal_out_scale = ideal_stream_scale;
	}

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

	// Checkpoint layers extended vs T-1762's {0,9,18,27}: {0,1,2} added to distinguish
	// "layer 0 specifically" from "the first few layers generally" (T-1763 brief item 2).
	const std::vector<uint32_t> checkpoint_layers = {0, 1, 2, 9, 18, 27};
	std::vector<RowDiag> all_rows;
	std::vector<InputDiag> all_inputs;
	std::vector<CapturedFullRow> all_full_rows;
	std::vector<ChainDiag> all_chains;
	std::vector<ElemZDiag> all_elem_z;
	std::vector<RowPeakDiag> all_peak;

	for (uint32_t step = 0; step < num_hidden_layers; ++step) {
		const bool is_checkpoint =
		    std::find(checkpoint_layers.begin(), checkpoint_layers.end(), step) !=
		    checkpoint_layers.end();

		if (is_checkpoint) {
			std::vector<int8_t> manual_hidden_codes(trace_hidden_codes);
			SequenceLayerState manual_seq = trace_seq;
			manual_seq.hidden_codes = manual_hidden_codes.data();

			std::vector<RowDiag> step_rows;
			InputDiag step_input;
			CapturedFullRow step_full;
			std::vector<ChainDiag> step_chain;
			std::vector<ElemZDiag> step_elem_z;
			std::vector<RowPeakDiag> step_peak;
			const SslmForwardStatus mst =
			    ManualRunOneLayer(manual_seq, layers[step], hidden_size, head_dim, num_kv_heads,
			                      intermediate_size, context_cap, model_view.rope_tables,
			                      trace_workspace.data(), &step_rows, &step_input, &step_full,
			                      &step_chain, &step_elem_z, &step_peak);
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
			// REAL arm's own full-layer output is ALSO proven bit-identical to
			// production above (it IS manual_seq.hidden_codes) -- so the real half of
			// step_full is self-checked transitively, same argument T-1762 uses for
			// ctx_codes, extended through the rest of the layer.
			std::printf("self_check: layer=%u manual replay and production agree bit-for-bit "
			            "(%zu rows captured)\n",
			            step, step_rows.size());
			for (auto& row : step_rows) all_rows.push_back(row);
			all_inputs.push_back(step_input);
			all_full_rows.push_back(std::move(step_full));
			for (auto& cd : step_chain) all_chains.push_back(cd);
			for (auto& ez : step_elem_z) all_elem_z.push_back(ez);
			for (auto& pk : step_peak) all_peak.push_back(pk);
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

	// -- Report 1: T-1762's own metric, per checkpoint layer (now including 1,2). --
	std::printf("\n[T-1763 downstream int8 attn_ctx code disagreement, per checkpoint layer "
	            "(T-1762 metric, extended layer set)]\n");
	for (const auto& fr : all_full_rows) {
		int64_t diff = 0, mx = 0;
		for (size_t i = 0; i < fr.real_ctx_codes.size(); ++i) {
			const int64_t d = std::abs(static_cast<int64_t>(fr.real_ctx_codes[i]) -
			                            static_cast<int64_t>(fr.ideal_ctx_codes[i]));
			if (d != 0) ++diff;
			mx = std::max(mx, d);
		}
		std::printf("  layer=%u: ctx_codes %lld/%zu differ (%.4f%%), max |delta| %lld\n", fr.layer,
		            static_cast<long long>(diff), fr.real_ctx_codes.size(),
		            100.0 * static_cast<double>(diff) / static_cast<double>(fr.real_ctx_codes.size()),
		            static_cast<long long>(mx));
	}

	// -- Report T-1768: the q_ln2 derivation chain, per (layer, kv_head) -- every
	// intermediate `IExpScaleConstants` (src/intmath.cpp:694-770) actually forms at its real
	// production call site (forward_sites.cpp:1362-1364), captured by execution rather than
	// inferred from the output q_ln2 alone.
	std::printf("\n[T-1768 q_ln2 derivation chain: normed_scale -> q_scale -> (combine with "
	            "artifact khead scale) -> sm -> r_m=DynamicScaleReciprocal(sm.m), "
	            "shift_ln2=30+62+sm.e (saturating) -> q_ln2]\n");
	for (const auto& cd : all_chains) {
		std::printf("  layer=%u kv_head=%u: normed_scale=(m=%lld,e=%lld) q_scale=(m=%lld,e=%lld) "
		            "khead=(m=%lld,e=%lld) sm=(m=%lld,e=%lld) r_m=%lld shift_ln2=%lld q_ln2=%lld\n",
		            cd.layer, cd.kv_head, static_cast<long long>(cd.normed_scale_m),
		            static_cast<long long>(cd.normed_scale_e), static_cast<long long>(cd.q_scale_m),
		            static_cast<long long>(cd.q_scale_e), static_cast<long long>(cd.khead_m),
		            static_cast<long long>(cd.khead_e), static_cast<long long>(cd.sm_m),
		            static_cast<long long>(cd.sm_e), static_cast<long long>(cd.r_m),
		            static_cast<long long>(cd.shift_ln2), static_cast<long long>(cd.q_ln2));
	}

	// -- Report T-1768 item 5: unclamped-element real/ideal Q15 delta, bucketed by octave
	// shift z -- tests whether D-SLM1159's candidate mechanism (small q_ln2 coarsens
	// resolution as a function of z, not only at the clip floor) holds by execution.
	std::printf("\n[T-1768 item 5: unclamped-element |real-ideal Q15 delta|, aggregated per "
	            "checkpoint layer and per octave-shift z]\n");
	for (uint32_t cl : checkpoint_layers) {
		int64_t n = 0, sum_delta = 0, max_delta = 0, z_min = -1, z_max = -1;
		for (const auto& ez : all_elem_z) {
			if (ez.layer != cl) continue;
			++n;
			sum_delta += ez.abs_delta;
			max_delta = std::max(max_delta, ez.abs_delta);
			if (z_min < 0 || ez.z < z_min) z_min = ez.z;
			if (ez.z > z_max) z_max = ez.z;
		}
		if (n == 0) continue;
		std::printf("  layer=%u: unclamped n=%lld mean|delta|=%.4f max|delta|=%lld z_range=[%lld,%lld]\n",
		            cl, static_cast<long long>(n), static_cast<double>(sum_delta) / static_cast<double>(n),
		            static_cast<long long>(max_delta), static_cast<long long>(z_min),
		            static_cast<long long>(z_max));
	}
	std::printf("\n[T-1768 item 5: same population, bucketed by z only (pooled across all "
	            "checkpoint layers) -- tests whether |delta| is a function of z independent of "
	            "which layer/q_ln2 produced it]\n");
	{
		int64_t max_z = 0;
		for (const auto& ez : all_elem_z) max_z = std::max(max_z, ez.z);
		for (int64_t zb = 0; zb <= max_z; ++zb) {
			int64_t n = 0, sum_delta = 0, max_delta = 0;
			for (const auto& ez : all_elem_z) {
				if (ez.z != zb) continue;
				++n;
				sum_delta += ez.abs_delta;
				max_delta = std::max(max_delta, ez.abs_delta);
			}
			if (n == 0) continue;
			std::printf("  z=%lld: n=%lld mean|delta|=%.4f max|delta|=%lld\n",
			            static_cast<long long>(zb), static_cast<long long>(n),
			            static_cast<double>(sum_delta) / static_cast<double>(n),
			            static_cast<long long>(max_delta));
		}
	}

	// -- Report T-1768 item 5b: joint (layer, z) breakdown for z=0..5 -- distinguishes "z
	// alone explains the magnitude" (same z, same delta regardless of layer) from "layer 0's
	// small q_ln2 elevates delta AT EVERY z" (same z, layer 0 still worse than other layers).
	std::printf("\n[T-1768 item 5b: joint (layer, z) mean|delta| for z=0..5 -- same-z "
	            "cross-layer comparison]\n");
	for (int64_t zb = 0; zb <= 5; ++zb) {
		std::printf("  z=%lld:", static_cast<long long>(zb));
		for (uint32_t cl : checkpoint_layers) {
			int64_t n = 0, sum_delta = 0;
			for (const auto& ez : all_elem_z) {
				if (ez.layer != cl || ez.z != zb) continue;
				++n;
				sum_delta += ez.abs_delta;
			}
			if (n == 0) {
				std::printf(" layer%u=(n/a)", cl);
			} else {
				std::printf(" layer%u=(n=%lld,mean=%.3f)", cl, static_cast<long long>(n),
				            static_cast<double>(sum_delta) / static_cast<double>(n));
			}
		}
		std::printf("\n");
	}

	// -- Report T-1768 item 5c: row peakedness (ideal top1 Q15 share; how many of the row's
	// own unclamped elements carry non-negligible ideal probability) -- tests whether layer
	// 0's elevated per-element delta (item 5/5b) coincides with a FLATTER row distribution.
	std::printf("\n[T-1768 item 5c: row peakedness per checkpoint layer -- ideal top1 Q15 "
	            "share (of 32768) and non-negligible unclamped element count]\n");
	for (uint32_t cl : checkpoint_layers) {
		int64_t n = 0, sum_top1 = 0, sum_nonneg = 0, sum_unclamped = 0;
		for (const auto& pk : all_peak) {
			if (pk.layer != cl) continue;
			++n;
			sum_top1 += pk.ideal_top1_q15;
			sum_nonneg += pk.nonneg_unclamped;
			sum_unclamped += pk.unclamped_total;
		}
		if (n == 0) continue;
		std::printf("  layer=%u: rows=%lld mean_top1_q15=%.1f(%.2f%%) mean_nonneg_unclamped=%.1f "
		            "mean_unclamped_total=%.1f\n",
		            cl, static_cast<long long>(n), static_cast<double>(sum_top1) / static_cast<double>(n),
		            100.0 * static_cast<double>(sum_top1) / static_cast<double>(n) / 32768.0,
		            static_cast<double>(sum_nonneg) / static_cast<double>(n),
		            static_cast<double>(sum_unclamped) / static_cast<double>(n));
	}

	// -- Report 2: pre-layer input distribution + value-side scale. --
	std::printf("\n[T-1763 pre-layer input code distribution + value-side (kv_head 0) code "
	            "distribution]\n");
	for (const auto& id : all_inputs) {
		std::printf("  layer=%u: input max|code|=%lld mean|code|=%.4f saturated(+-127)=%lld/%zu | "
		            "V max|code|=%lld mean|code|=%.4f\n",
		            id.layer, static_cast<long long>(id.max_abs_code), id.mean_abs_code,
		            static_cast<long long>(id.saturated_count), id.hidden_size,
		            static_cast<long long>(id.v_max_abs_code), id.v_mean_abs_code);
	}

	// -- Report 3: softmax row diagnostics, aggregated per layer, including domain-clip
	// crowding (T-1763 brief item 1: whether the i-exp construction's domain is being
	// approached/exceeded at layer 0). --
	std::printf("\n[T-1763 softmax row diagnostics, aggregated per checkpoint layer]\n");
	for (uint32_t cl : checkpoint_layers) {
		int64_t n = 0;
		int64_t range_min = -1, range_max = -1;
		double range_sum = 0.0;
		int64_t qln2_min = -1, qln2_max = -1;
		int64_t clip_window_val = -1;
		int64_t total_elems = 0, total_clipped = 0;
		for (const auto& rd : all_rows) {
			if (rd.layer != cl) continue;
			++n;
			if (range_min < 0 || rd.score_range < range_min) range_min = rd.score_range;
			if (rd.score_range > range_max) range_max = rd.score_range;
			range_sum += static_cast<double>(rd.score_range);
			if (qln2_min < 0 || rd.q_ln2 < qln2_min) qln2_min = rd.q_ln2;
			if (rd.q_ln2 > qln2_max) qln2_max = rd.q_ln2;
			clip_window_val = rd.clip_window;
			total_elems += rd.width;
			total_clipped += rd.clipped_elements;
		}
		if (n == 0) continue;
		std::printf("  layer=%u: heads=%lld score_range[min=%lld mean=%.1f max=%lld] "
		            "q_ln2[min=%lld max=%lld] clip_window=%lld clipped_elements=%lld/%lld (%.2f%%)\n",
		            cl, static_cast<long long>(n), static_cast<long long>(range_min),
		            range_sum / static_cast<double>(n), static_cast<long long>(range_max),
		            static_cast<long long>(qln2_min), static_cast<long long>(qln2_max),
		            static_cast<long long>(clip_window_val), static_cast<long long>(total_clipped),
		            static_cast<long long>(total_elems),
		            100.0 * static_cast<double>(total_clipped) / static_cast<double>(total_elems));
	}

	// -- Report 4: downstream propagation -- final post-layer hidden_codes/scale, real vs
	// ideal arm, same layer's downstream weights for both. --
	std::printf("\n[T-1763 downstream propagation: post-layer hidden_codes, real vs ideal attn_ctx "
	            "arm]\n");
	for (const auto& fr : all_full_rows) {
		int64_t code_diff = 0, code_max = 0;
		double sq_sum = 0.0;
		for (size_t i = 0; i < fr.real_out_codes.size(); ++i) {
			const int64_t d = std::abs(static_cast<int64_t>(fr.real_out_codes[i]) -
			                            static_cast<int64_t>(fr.ideal_out_codes[i]));
			if (d != 0) ++code_diff;
			code_max = std::max(code_max, d);
			sq_sum += static_cast<double>(d) * static_cast<double>(d);
		}
		const double rmse = std::sqrt(sq_sum / static_cast<double>(fr.real_out_codes.size()));
		const bool scale_match = fr.real_out_scale.m == fr.ideal_out_scale.m &&
		                          fr.real_out_scale.e == fr.ideal_out_scale.e;
		std::printf(
		    "  layer=%u: post-layer hidden_codes %lld/%zu differ (%.4f%%), max |delta| %lld, "
		    "code RMSE %.4f, out_scale match=%d (real m=%lld e=%lld, ideal m=%lld e=%lld)\n",
		    fr.layer, static_cast<long long>(code_diff), fr.real_out_codes.size(),
		    100.0 * static_cast<double>(code_diff) / static_cast<double>(fr.real_out_codes.size()),
		    static_cast<long long>(code_max), rmse, scale_match ? 1 : 0,
		    static_cast<long long>(fr.real_out_scale.m), static_cast<long long>(fr.real_out_scale.e),
		    static_cast<long long>(fr.ideal_out_scale.m), static_cast<long long>(fr.ideal_out_scale.e));
	}

	// Dump: raw arrays for independent re-verification.
	std::ofstream f(dump_path);
	if (!f) {
		std::fprintf(stderr, "FAILED at stage=dump_open: could not open \"%s\"\n", dump_path.c_str());
		return 1;
	}
	f << all_rows.size() << "\n";
	for (const auto& rd : all_rows) {
		f << rd.layer << " " << rd.head << " " << rd.width << " " << rd.q_ln2 << " " << rd.q_b << " "
		  << rd.q_c << " " << rd.score_range << " " << rd.m_bound << " " << rd.clip_window << " "
		  << rd.clipped_elements << "\n";
	}
	f << all_inputs.size() << "\n";
	for (const auto& id : all_inputs) {
		f << id.layer << " " << id.max_abs_code << " " << id.mean_abs_code << " " << id.saturated_count
		  << " " << id.hidden_size << " " << id.v_max_abs_code << " " << id.v_mean_abs_code << "\n";
	}
	f << all_full_rows.size() << " " << hidden_size << "\n";
	for (const auto& fr : all_full_rows) {
		f << fr.layer << "\n";
		for (size_t i = 0; i < fr.real_ctx_codes.size(); ++i)
			f << static_cast<int>(fr.real_ctx_codes[i]) << (i + 1 < fr.real_ctx_codes.size() ? ' ' : '\n');
		for (size_t i = 0; i < fr.ideal_ctx_codes.size(); ++i)
			f << static_cast<int>(fr.ideal_ctx_codes[i])
			  << (i + 1 < fr.ideal_ctx_codes.size() ? ' ' : '\n');
		for (size_t i = 0; i < fr.real_out_codes.size(); ++i)
			f << static_cast<int>(fr.real_out_codes[i]) << (i + 1 < fr.real_out_codes.size() ? ' ' : '\n');
		for (size_t i = 0; i < fr.ideal_out_codes.size(); ++i)
			f << static_cast<int>(fr.ideal_out_codes[i])
			  << (i + 1 < fr.ideal_out_codes.size() ? ' ' : '\n');
	}
	std::printf("\ndump written: %s (%zu softmax-diag rows + %zu input-diag rows + %zu full-layer "
	            "rows, width=%lld, layers={0,1,2,9,18,27})\n",
	            dump_path.c_str(), all_rows.size(), all_inputs.size(), all_full_rows.size(),
	            static_cast<long long>(width_at_last_token));
	return 0;
}
