// t1787_score_stage_probe.cpp -- T-1787: decompose the RAW INTEGER attention score
// divergence T-1786 found upstream of its own softmax construction (D-SLM1286: 99.6% of the
// measured probability-space TVD survives an exact softmax over production's own raw scores)
// into the stages that PRODUCE those raw scores -- Query quantization, Key quantization (the
// landing-rescale requantization chain), and RoPE -- by the SAME substitution discipline T-1786
// used one level downstream (`tools/t1786_stage_dump_probe.cpp`, read in full before reuse;
// this probe's self-check structure, checkpoint-layer convention, and manual-replay pattern for
// the LAST token are copied verbatim from it).
//
// WHAT'S NEW, AND WHY.
//
// T-1786's own probe captures the raw post-GEMM `scores` array and the softmax's own
// (q_ln2,q_b,q_c) triple -- everything SoftmaxRowQ15 consumes. It does NOT capture anything
// upstream of that point. `scores[k] = dot(q_rot, k_rot[k])`, an EXACT int8x int8 dot product
// (GemmInt8AccumulateRow, lossless integer accumulation -- confirmed the same way T-1786 Sec3
// confirmed accumulation is lossless at the softmax level). Both operands are already-quantized,
// already-RoPE'd int8 codes. This probe opens up THAT construction:
//
//   normed (int8, RmsNorm output) --[Q-projection: GEMM+fold+bias+RequantTokenCodeWide]--> q_codes (int8)
//   normed (int8, shared)         --[K-projection: GEMM+fold+bias+LandingRescale+ClampRopeCode]--> k_row (int8)
//   q_codes --[RopeApplyPair+ClampRopeCode]--> q_rot (int8)
//   k_row   --[RopeApplyPair+ClampRopeCode]--> k_rot (int8)      (recomputed EVERY cached position
//                                                                  at ITS OWN prefill timestep)
//   scores[k] = dot(q_rot, k_rot[k])
//
// Three isolable ROUNDING mechanisms sit in this chain, matching the three candidates T-1786's
// own handoff named (D-SLM1286): (1) Query's own quantization round, RequantTokenCodeWide's
// round-half-away-from-zero + implicit [-127,127] clamp; (2) Key's own quantization round, the
// LandingRescale+ClampRopeCode requantization chain -- a MECHANICALLY DISTINCT chain from
// Query's, since K routes through a fixed target ("landing") scale rather than Query's dynamic
// per-token RequantChainChecked funnel; (3) RoPE's own round, RopeApplyPair's
// round-half-away-from-zero + ClampRopeCode clamp, applied identically to both Q and K. This
// probe's own reading of the ticket's three named candidates ("the query and key quantization,
// the rotary embedding, and the requantization chain feeding the score matmul") is: query
// quantization and RoPE are the first two clauses; "the requantization chain feeding the score
// matmul" is Key's OWN distinct landing-rescale mechanism, singled out because it is textually
// and mechanically separate from Query's -- stated explicitly here because the source has no
// FOURTH mechanism between RoPE's output and the score GEMM that a literal three-with-Query-
// twice reading would require, checked directly (GemmInt8AccumulateRow is called immediately on
// RopeApplySite's own output, forward_sites.cpp).
//
// METHOD, matching T-1786's own substitution shape exactly: for each stage, PRODUCTION uses the
// real rounding/clamping function (called directly -- RequantTokenCodeWide, LandingRescale,
// ClampRopeCode, RopeApplyPair -- zero formula-porting risk for the baseline); the EXACT variant
// removes ONLY that stage's own round/clamp, carrying a continuous (double) value through, while
// every OTHER stage's production rounding still applies to whatever it is handed (matching
// T-1786's stage B, which re-quantized its own exact exponential to Q15 before comparing --
// "matching how production would consume a corrected exponential"). The three EXACT formulas
// below are read directly from the production source with the file:line each is copied from,
// not re-derived:
//
//   Query exact code (no round, no clamp), from RequantTokenCodeWide (src/intmath.cpp:468-482):
//     exact_q[d] = wide_row[d] * 127.0 * r / 2^(62 - s)          (r, s from the SAME
//     MaxAbsReduceWide/NormalizeScale/CarriedScaleReciprocal call production made on this exact
//     row -- called for real below, not re-derived, so a self-check against the dumped q_codes
//     bit-for-bit validates r/s are the SAME values production used.)
//
//   Key exact code (no round, no clamp), from LandingRescale (src/forward/forward_sites.cpp:301-
//   441, "C27's residual_reconcile" comment, its own documented formula):
//     exact_k[d] = kacc[d] * m_a * r_t / 2^(62 - (e_a - e_t))
//
//   RoPE exact rotation (round removed, clamp removed), from RopeApplyPair (src/intmath.cpp:774-
//   784):
//     xr = x*cos_q30 - y*sin_q30; yr = x*sin_q30 + y*cos_q30
//     exact_rot_x = xr / 2^30; exact_rot_y = yr / 2^30            (ROPE_FRAC_BITS = 30)
//
// SELF-CHECK, extending T-1786's own discipline from "the last token only" to EVERY prefill
// position: for every (layer, position, kv_head) at the 6 checkpoint layers, this probe computes
// K's OWN pre-landing-rescale accumulator (kacc) via a SIDE computation (RmsNormSite + K's own
// GEMM+fold+bias, called BEFORE production's real per-layer forward for that token), then
// reconstructs k_rot by calling the REAL LandingRescale/ClampRopeCode/RopeApplyPair functions on
// it, and compares the result BIT-FOR-BIT against what production's own real forward call
// (immediately following) actually wrote into the KV cache (read back via KeyRow() -- the cache
// slot holds ONLY the post-RoPE value by the time production's call returns, which is exactly
// what this self-check needs to match). A mismatch on ANY (layer,position,kv_head) hard-stops
// the probe before any dump is written -- identical discipline to T-1786's own assert-on-
// mismatch self-check. This validates kacc, normed_scale, and every fixed constant
// (kv_landing_r_t_k/e_t_k) this probe reads, TRANSITIVELY, the same argument T-1786 Sec"instruments
// built" makes for its own captured values: read from the identical compiled call sequence
// production uses, whose later, externally-observable output is proven bit-identical to
// production.
//
// Query's OWN pre-round accumulator (q_wide) is captured only at the LAST prefill token (the one
// graded row), self-checked the same way: RequantTokenCodeWide(q_wide[d], r, s) compared against
// the actual q_codes[d] production computed at that step (captured via a small, read-only
// extension of ManualRunOneLayer, mirroring T-1786's own extension of T-1778's probe).
//
// Splitting the prefill loop into layer_budget=1 steps for EVERY token (not just the last, as
// T-1786's own probe does) relies on RunLayerLoop's own budget-invariance guarantee (a tested
// system invariant this codebase's own test suite gates on -- "budget invariance green at every
// enumerated budget", checked_chain_funnel.h's own §11 S3.5 citation) -- calling with
// layer_budget=1 twenty-eight times produces bit-identical state to one layer_budget=28 call,
// which is exactly what T-1786's own probe already relies on for the LAST token (mixing
// layer_budget=1 manual-replay steps with layer_budget=1 production steps across one token's own
// 28 layers). This probe applies the identical pattern to every token.

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

// Verbatim copy of forward_sites.cpp's own anonymous-namespace ReadRopeTableEntryI64
// (src/forward/forward_sites.cpp:48-53) -- internal linkage there, so this probe carries its
// own copy, a pure mechanical little-endian byte-assembly read with no fixed-point
// interpretation of its own.
int64_t ReadRopeTableEntryI64Copy(const uint8_t* base, uint64_t index) {
	const uint8_t* p = base + index * 8;
	uint64_t v = 0;
	for (int i = 0; i < 8; ++i) v |= static_cast<uint64_t>(p[i]) << (8 * i);
	return static_cast<int64_t>(v);
}

// -- verbatim copies of forward_sites.cpp's internal-linkage helpers, identical to T-1786's own
// probe (which copied them from T-1778's). --
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

// Q's own funnel, EXTENDED to also hand back the pre-round wide accumulator (the exact same
// `acc` RequantChainChecked would consume) -- read-only addition, the real RequantChainChecked
// call and its real inputs/outputs are unchanged from T-1786's own ProjectAndFunnelCopy.
SslmForwardStatus ProjectAndFunnelCopyCapture(const int8_t* in_codes, CarriedScale in_scale,
                                               const int8_t* weight, size_t in_channels,
                                               size_t out_channels, const int32_t* identity,
                                               const int32_t* mult, const int32_t* shift,
                                               CarriedScale site_constant, const int64_t* bias,
                                               int8_t* out_codes, CarriedScale* out_scale,
                                               std::vector<int64_t>* out_wide /* nullable */) {
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
	if (out_wide != nullptr) *out_wide = acc;  // captured AFTER fold+bias, BEFORE the round below
	const CarriedScale incoming[1] = {in_scale};
	const ChainResult result = RequantChainChecked(
	    acc.data(), out_channels, std::span<const CarriedScale>{incoming, 1}, site_constant, out_codes,
	    out_scale);
	return result.status;
}

// K's own pre-landing-rescale accumulator (kacc), computed as a SIDE, read-only replay of
// exactly the GEMM+fold+bias steps production's own RunLayerLoop performs internally for the K
// projection at this (layer, position) -- called BEFORE production's own real per-layer forward
// for this token, from the SAME `normed`/`normed_scale` this function derives via the SAME
// RmsNormSite call production makes (deterministic given the same seq state and layer weights,
// so this call reproduces production's own internal normed/normed_scale bit-for-bit -- validated
// transitively below, not assumed).
struct KCapture {
	CarriedScale normed_scale{};
	std::vector<int64_t> kacc;  // num_key_value_heads * head_dim, post fold+bias, pre-landing-rescale
};

SslmForwardStatus CaptureKAccumulator(const SequenceLayerState& seq, const LayerWeights& lw,
                                       size_t hidden_size, size_t head_dim,
                                       size_t num_key_value_heads, KCapture* out) {
	std::vector<int8_t> normed(hidden_size);
	SslmForwardStatus st = RmsNormSite(seq.hidden_codes, lw.attn_norm_gain, hidden_size,
	                                    seq.hidden_scale, lw.attn_norm_site_constant, normed.data(),
	                                    &out->normed_scale);
	if (st != SslmForwardStatus::Ok) return st;

	const size_t kv_hidden_size = num_key_value_heads * head_dim;
	out->kacc.assign(kv_hidden_size, 0);
	GemmInt8AccumulateRow(normed.data(), lw.k_weight, hidden_size, kv_hidden_size, out->kacc.data());
	for (size_t i = 0; i < kv_hidden_size; ++i) {
		out->kacc[i] =
		    ApplyWeightScaleFold(out->kacc[i], lw.k_fold_identity[i], lw.k_fold_mult[i], lw.k_fold_shift[i]);
	}
	if (lw.k_bias != nullptr) {
		st = ApplyBiasReconcileRowCopy(out->kacc.data(), kv_hidden_size, lw.k_bias, out->normed_scale.m,
		                                out->normed_scale.e);
		if (st != SslmForwardStatus::Ok) return st;
	}
	return SslmForwardStatus::Ok;
}

// One captured (layer, head) softmax row: identical shape to T-1786's own CapturedProbRow.
struct CapturedProbRow {
	uint32_t layer = 0;
	uint32_t head = 0;
	int64_t width = 0;
	int64_t q_ln2 = 0;
	int64_t q_b = 0;
	int64_t q_c = 0;
	std::vector<int64_t> scores;
	std::vector<int64_t> probs;
};

// Per-layer Q capture: the full hidden_size pre-round accumulator, plus the recomputed (r,s) --
// called for real via MaxAbsReduceWide/NormalizeScale/CarriedScaleReciprocal, self-checked
// against the real q_codes below before being trusted.
struct QCapture {
	uint32_t layer = 0;
	int64_t r = 0;
	int32_t s = 0;
	std::vector<int64_t> wide;  // hidden_size
};

// Per-(layer,position,kv_head) K capture, written to the dump.
struct KRow {
	uint32_t layer = 0;
	int64_t position = 0;
	uint32_t kv_head = 0;
	int64_t normed_m = 0;
	int64_t normed_e = 0;
	int64_t r_t = 0;
	int64_t e_t = 0;
	std::vector<int64_t> kacc;  // head_dim
};

void PrintUsage(const char* argv0) {
	std::fprintf(stderr,
	             "usage: %s <model.sslm> <tokenizer.sslm> \"<prompt>\" --dump-prefix <path>\n", argv0);
}

// Manual replay of exactly ONE layer's forward through the softmax for the LAST token, extending
// T-1786's own ManualRunOneLayer with Q's pre-round wide-accumulator capture (ProjectAndFunnelCopyCapture)
// and self-check.
SslmForwardStatus ManualRunOneLayer(SequenceLayerState& seq, const LayerWeights& lw,
                                     size_t hidden_size, size_t head_dim, size_t num_key_value_heads,
                                     size_t intermediate_size, int64_t context_cap,
                                     const SslmTensorManifest& rope_tables, uint8_t* workspace,
                                     std::vector<CapturedProbRow>* out_rows, QCapture* out_qcap) {
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

	std::vector<int64_t> q_wide;
	st = ProjectAndFunnelCopyCapture(normed.data(), normed_scale, lw.q_weight, hidden_size, hidden_size,
	                                  lw.q_fold_identity, lw.q_fold_mult, lw.q_fold_shift,
	                                  lw.q_site_constant, lw.q_bias, q_codes.data(), &q_scale, &q_wide);
	if (st != SslmForwardStatus::Ok) return st;

	if (out_qcap != nullptr) {
		out_qcap->layer = l;
		out_qcap->wide = q_wide;
		const int64_t d_prime = MaxAbsReduceWide(q_wide.data(), q_wide.size());
		const NormalizedScale ns = NormalizeScale(d_prime);
		out_qcap->r = CarriedScaleReciprocal(ns.dn);
		out_qcap->s = ns.s;
		// Self-check: RequantTokenCodeWide(q_wide[i], r, s) must reproduce the REAL q_codes[i]
		// production just computed via RequantChainChecked, for EVERY element -- proves r/s are
		// the SAME values production's own internal fold derived, before anything downstream
		// trusts them.
		for (size_t i = 0; i < hidden_size; ++i) {
			const int8_t replica = RequantTokenCodeWide(q_wide[i], out_qcap->r, out_qcap->s);
			if (replica != q_codes[i]) {
				std::fprintf(stderr,
				             "FAILED at stage=q_wide_self_check: layer=%u i=%zu replica=%d "
				             "production=%d\n",
				             l, i, static_cast<int>(replica), static_cast<int>(q_codes[i]));
				return SslmForwardStatus::SoftmaxKernelRefusedAfterGateAccepted;  // reuse: any
				                                                                  // non-Ok halts main()
			}
		}
	}

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

	// --- REAL arm continues, identical to RunLayerLoop's own remaining body. ---
	st = ProjectAndFunnelCopyCapture(ctx_codes.data(), ctx_scale, lw.o_weight, hidden_size, hidden_size,
	                      lw.o_fold_identity, lw.o_fold_mult, lw.o_fold_shift, lw.o_site_constant,
	                      /*bias=*/nullptr, o_codes.data(), &o_scale, /*out_wide=*/nullptr);
	if (st != SslmForwardStatus::Ok) return st;

	st = ResidualReconcileSite(o_codes.data(), o_scale, seq.hidden_codes, seq.hidden_scale, hidden_size,
	                           lw.attn_residual_site_constant, attn_stream.data(), &attn_stream_scale);
	if (st != SslmForwardStatus::Ok) return st;

	st = RmsNormSite(attn_stream.data(), lw.mlp_norm_gain, hidden_size, attn_stream_scale,
	                 lw.mlp_norm_site_constant, normed.data(), &mlp_normed_scale);
	if (st != SslmForwardStatus::Ok) return st;

	st = ProjectAndFunnelCopyCapture(normed.data(), mlp_normed_scale, lw.gate_weight, hidden_size,
	                      intermediate_size, lw.gate_fold_identity, lw.gate_fold_mult,
	                      lw.gate_fold_shift, lw.gate_site_constant, /*bias=*/nullptr, gate_codes.data(),
	                      &gate_scale, /*out_wide=*/nullptr);
	if (st != SslmForwardStatus::Ok) return st;
	st = ProjectAndFunnelCopyCapture(normed.data(), mlp_normed_scale, lw.up_weight, hidden_size, intermediate_size,
	                      lw.up_fold_identity, lw.up_fold_mult, lw.up_fold_shift, lw.up_site_constant,
	                      /*bias=*/nullptr, up_codes.data(), &up_scale, /*out_wide=*/nullptr);
	if (st != SslmForwardStatus::Ok) return st;

	st = MlpActSite(gate_codes.data(), gate_scale, up_codes.data(), up_scale, intermediate_size,
	                kSiluLutCanonicalTable, lw.mlp_act_site_constant, act_codes.data(), &act_scale);
	if (st != SslmForwardStatus::Ok) return st;

	st = ProjectAndFunnelCopyCapture(act_codes.data(), act_scale, lw.down_weight, intermediate_size, hidden_size,
	                      lw.down_fold_identity, lw.down_fold_mult, lw.down_fold_shift,
	                      lw.down_site_constant, /*bias=*/nullptr, down_codes.data(), &down_scale,
	                      /*out_wide=*/nullptr);
	if (st != SslmForwardStatus::Ok) return st;

	st = ResidualReconcileSite(down_codes.data(), down_scale, attn_stream.data(), attn_stream_scale,
	                           hidden_size, lw.mlp_residual_site_constant, stream_next.data(),
	                           &stream_scale);
	if (st != SslmForwardStatus::Ok) return st;

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
	std::string dump_prefix;
	for (int i = 4; i < argc; ++i) {
		if (std::strcmp(argv[i], "--dump-prefix") == 0 && i + 1 < argc) {
			dump_prefix = argv[++i];
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

	const std::vector<uint32_t> checkpoint_layers = {0, 1, 2, 9, 18, 27};
	auto is_checkpoint = [&](uint32_t l) {
		return std::find(checkpoint_layers.begin(), checkpoint_layers.end(), l) !=
		       checkpoint_layers.end();
	};

	std::vector<KRow> all_k_rows;  // every (layer,position,kv_head) at checkpoint layers, every token

	// Prefill: EVERY token, layer-by-layer (layer_budget=1 x num_hidden_layers), relying on
	// RunLayerLoop's own tested budget-invariance (see file header). At checkpoint layers, K's
	// pre-landing-rescale accumulator is captured and self-checked BEFORE production's real call.
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
				KCapture kcap;
				st = CaptureKAccumulator(trace_seq, layers[l], hidden_size, head_dim, num_kv_heads, &kcap);
				if (st != SslmForwardStatus::Ok) {
					std::fprintf(stderr,
					             "FAILED at stage=k_capture: layer=%u position=%lld status=%s\n", l,
					             static_cast<long long>(position), SslmForwardStatusName(st));
					return 1;
				}
				// Reconstruct k_rot from kacc via the REAL production functions, per kv_head.
				std::vector<std::vector<int8_t>> side_k_rot(num_kv_heads, std::vector<int8_t>(head_dim));
				const SslmTensorView* cos_t = model_view.rope_tables.Tensor("cos");
				const SslmTensorView* sin_t = model_view.rope_tables.Tensor("sin");
				(void)cos_t;
				(void)sin_t;
				for (uint32_t h = 0; h < num_kv_heads; ++h) {
					std::vector<int8_t> pre_rope(head_dim);
					for (size_t d = 0; d < head_dim; ++d) {
						const size_t i = h * head_dim + d;
						pre_rope[d] = static_cast<int8_t>(ClampRopeCode(LandingRescale(
						    kcap.kacc[i], kcap.normed_scale.m, layers[l].kv_landing_r_t_k[h],
						    kcap.normed_scale.e, layers[l].kv_landing_e_t_k[h],
						    &trace_seq.kv_saturation_count)));
					}
					st = RopeApplySite(pre_rope.data(), head_dim, position, context_cap,
					                    model_view.rope_tables, side_k_rot[h].data());
					if (st != SslmForwardStatus::Ok) {
						std::fprintf(stderr,
						             "FAILED at stage=k_side_rope: layer=%u position=%lld kv_head=%u "
						             "status=%s\n",
						             l, static_cast<long long>(position), h, SslmForwardStatusName(st));
						return 1;
					}
					KRow row;
					row.layer = l;
					row.position = position;
					row.kv_head = h;
					row.normed_m = kcap.normed_scale.m;
					row.normed_e = kcap.normed_scale.e;
					row.r_t = layers[l].kv_landing_r_t_k[h];
					row.e_t = layers[l].kv_landing_e_t_k[h];
					row.kacc.assign(kcap.kacc.begin() + h * head_dim, kcap.kacc.begin() + (h + 1) * head_dim);
					all_k_rows.push_back(std::move(row));
				}
				// Production's real forward for this layer, then self-check side_k_rot against
				// what it actually wrote into the KV cache.
				st = RunLayerLoop(trace_seq, layers.data(), num_hidden_layers, /*layer_budget=*/1,
				                   hidden_size, head_dim, num_kv_heads, intermediate_size, context_cap,
				                   model_view.rope_tables, trace_workspace.data(), trace_workspace.size());
				if (st != SslmForwardStatus::Ok) {
					std::fprintf(stderr, "FAILED at stage=prefill_layer: layer=%u position=%lld status=%s\n",
					             l, static_cast<long long>(position), SslmForwardStatusName(st));
					return 1;
				}
				for (uint32_t h = 0; h < num_kv_heads; ++h) {
					const int8_t* const actual =
					    KeyRow(trace_workspace.data(), l, context_cap, num_kv_heads, head_dim, h, position);
					if (std::memcmp(actual, side_k_rot[h].data(), head_dim) != 0) {
						std::fprintf(stderr,
						             "FAILED at stage=k_self_check: layer=%u position=%lld kv_head=%u -- "
						             "side replay diverges from production's own KV cache write\n",
						             l, static_cast<long long>(position), h);
						return 1;
					}
				}
			} else {
				st = RunLayerLoop(trace_seq, layers.data(), num_hidden_layers, /*layer_budget=*/1,
				                   hidden_size, head_dim, num_kv_heads, intermediate_size, context_cap,
				                   model_view.rope_tables, trace_workspace.data(), trace_workspace.size());
				if (st != SslmForwardStatus::Ok) {
					std::fprintf(stderr, "FAILED at stage=prefill_layer: layer=%u position=%lld status=%s\n",
					             l, static_cast<long long>(position), SslmForwardStatusName(st));
					return 1;
				}
			}
		}
		std::printf("prefill position=%zu: %u/%u checkpoint layers self-checked bit-for-bit\n", tok_idx,
		            static_cast<unsigned>(checkpoint_layers.size()),
		            static_cast<unsigned>(checkpoint_layers.size()));
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

	std::vector<CapturedProbRow> all_rows;
	std::vector<QCapture> all_qcaps;

	for (uint32_t step = 0; step < num_hidden_layers; ++step) {
		if (is_checkpoint(step)) {
			// K's own pre-landing-rescale accumulator at the LAST token's own position (the
			// query attends to its own key too, causal self-attention -- this position is
			// otherwise never captured, since the prefill capture loop above only covers
			// positions 0..width-2). Same capture+self-check discipline as the prefill loop,
			// applied here to the one remaining position.
			const int64_t last_position = trace_seq.context_length;
			KCapture last_kcap;
			SslmForwardStatus kst =
			    CaptureKAccumulator(trace_seq, layers[step], hidden_size, head_dim, num_kv_heads,
			                        &last_kcap);
			if (kst != SslmForwardStatus::Ok) {
				std::fprintf(stderr, "FAILED at stage=k_capture_last: layer=%u status=%s\n", step,
				             SslmForwardStatusName(kst));
				return 1;
			}
			std::vector<std::vector<int8_t>> last_side_k_rot(num_kv_heads, std::vector<int8_t>(head_dim));
			for (uint32_t h = 0; h < num_kv_heads; ++h) {
				std::vector<int8_t> pre_rope(head_dim);
				for (size_t d = 0; d < head_dim; ++d) {
					const size_t i = h * head_dim + d;
					pre_rope[d] = static_cast<int8_t>(ClampRopeCode(LandingRescale(
					    last_kcap.kacc[i], last_kcap.normed_scale.m, layers[step].kv_landing_r_t_k[h],
					    last_kcap.normed_scale.e, layers[step].kv_landing_e_t_k[h],
					    &trace_seq.kv_saturation_count)));
				}
				kst = RopeApplySite(pre_rope.data(), head_dim, last_position, context_cap,
				                     model_view.rope_tables, last_side_k_rot[h].data());
				if (kst != SslmForwardStatus::Ok) {
					std::fprintf(stderr, "FAILED at stage=k_side_rope_last: layer=%u kv_head=%u status=%s\n",
					             step, h, SslmForwardStatusName(kst));
					return 1;
				}
				KRow row;
				row.layer = step;
				row.position = last_position;
				row.kv_head = h;
				row.normed_m = last_kcap.normed_scale.m;
				row.normed_e = last_kcap.normed_scale.e;
				row.r_t = layers[step].kv_landing_r_t_k[h];
				row.e_t = layers[step].kv_landing_e_t_k[h];
				row.kacc.assign(last_kcap.kacc.begin() + h * head_dim,
				                 last_kcap.kacc.begin() + (h + 1) * head_dim);
				all_k_rows.push_back(std::move(row));
			}

			std::vector<int8_t> manual_hidden_codes(trace_hidden_codes);
			SequenceLayerState manual_seq = trace_seq;
			manual_seq.hidden_codes = manual_hidden_codes.data();

			std::vector<CapturedProbRow> step_rows;
			QCapture qcap;
			const SslmForwardStatus mst =
			    ManualRunOneLayer(manual_seq, layers[step], hidden_size, head_dim, num_kv_heads,
			                      intermediate_size, context_cap, model_view.rope_tables,
			                      trace_workspace.data(), &step_rows, &qcap);
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
				             "FAILED at stage=self_check: layer=%u codes_match=%d scale_match=%d\n",
				             step, codes_match ? 1 : 0, scale_match ? 1 : 0);
				return 1;
			}
			// Self-check the last-position K capture against what production (via the
			// ManualRunOneLayer call just proven bit-identical to production above) actually
			// wrote for that position's k_rot -- read back via KeyRow() on the manual replay's
			// own workspace (shared `trace_workspace`, the same buffer production writes into).
			for (uint32_t h = 0; h < num_kv_heads; ++h) {
				const int8_t* const actual = KeyRow(trace_workspace.data(), step, context_cap,
				                                     num_kv_heads, head_dim, h, last_position);
				if (std::memcmp(actual, last_side_k_rot[h].data(), head_dim) != 0) {
					std::fprintf(stderr,
					             "FAILED at stage=k_self_check_last: layer=%u kv_head=%u -- side "
					             "replay diverges from production's own KV cache write\n",
					             step, h);
					return 1;
				}
			}
			std::printf("self_check: layer=%u manual replay and production agree bit-for-bit "
			            "(%zu rows captured, q_wide self-check passed on %zu elements, last-position "
			            "K self-check passed on %u kv_heads)\n",
			            step, step_rows.size(), qcap.wide.size(), num_kv_heads);
			for (auto& row : step_rows) all_rows.push_back(std::move(row));
			all_qcaps.push_back(std::move(qcap));
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

	// Dump 1: scores (identical format to t1786_stage_dump_probe.cpp's own dump).
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
	// Dump 2: Q's per-layer pre-round accumulator + recomputed (r,s).
	{
		std::ofstream f(dump_prefix + "_qwide.txt");
		f << all_qcaps.size() << " " << hidden_size << "\n";
		for (const auto& q : all_qcaps) {
			f << q.layer << " " << q.r << " " << q.s;
			for (int64_t w : q.wide) f << " " << w;
			f << "\n";
		}
	}
	// Dump 3: K's per-(layer,position,kv_head) pre-landing-rescale accumulator.
	{
		std::ofstream f(dump_prefix + "_kacc.txt");
		f << all_k_rows.size() << " " << head_dim << "\n";
		for (const auto& r : all_k_rows) {
			f << r.layer << " " << r.position << " " << r.kv_head << " " << r.normed_m << " "
			  << r.normed_e << " " << r.r_t << " " << r.e_t;
			for (int64_t v : r.kacc) f << " " << v;
			f << "\n";
		}
	}
	// Dump 4: RoPE tables (position, pair) -> (cos_q30, sin_q30). Shared across every layer.
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

	std::printf("\ndumps written: %s_{scores,qwide,kacc,rope}.txt (%zu score rows, %zu q captures, "
	            "%zu k rows, width=%lld)\n",
	            dump_prefix.c_str(), all_rows.size(), all_qcaps.size(), all_k_rows.size(),
	            static_cast<long long>(width_at_last_token));
	return 0;
}
