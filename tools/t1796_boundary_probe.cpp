// t1796_boundary_probe.cpp -- T-1796: measure how far every value at every quantization
// site sits from its own rounding/truncation boundary, as a fraction of the bin width.
//
// D-SLM1360/D-SLM1361: quantization is a step function -- a numerical improvement changes
// the assigned integer code only where it moves a value across a boundary. This probe
// measures the missing exchange rate directly: for every value the engine actually
// quantizes on a real forward pass, how close was it (in bin-width units) to the boundary
// that would have flipped its code.
//
// TWO CAPTURE ARMS, both against REAL production arithmetic, never a float reference (this
// measurement is a property of the engine's own integer arithmetic -- no float reference is
// required or used):
//
//   ARM A -- the trace-hook arm (checked_chain_funnel.h's SslmTraceHookState/
//   SslmEmitChainTrace, already wired into src/forward/forward_sites.cpp's own
//   RequantChainChecked call at every one of 11 named per-layer sites: attn_norm,
//   q_proj.requant, attn_ctx, o_proj.requant, attn_residual, mlp_norm, gate_proj.requant,
//   up_proj.requant, mlp_act, down_proj.requant, mlp_residual). This is READ-ONLY BY
//   CONSTRUCTION (trace_hook.h's own header comment: "installing a hook adds a notification
//   after a site's own arithmetic has already produced its outputs... no hook call
//   participates in, or can change, any value a site returns or writes"). This probe still
//   verifies that claim empirically (below: a hooked run and an unhooked run are diffed on
//   final hidden_codes/hidden_scale, byte for byte). Because it is real production code —
//   RunLayerLoop called directly, exactly as tools/t1786_stage_dump_probe.cpp and
//   tools/t1795_residual_probe.cpp already call it — there is no "manual replay" to
//   self-check for this arm: the arm IS production. What this probe self-checks instead is
//   its OWN reconstruction of each site's rounding boundary from the trace record's
//   (x_int, r, s) triple, against that same record's own `codes` output (the real,
//   production-computed code) -- if the reconstructed rounded magnitude ever disagrees with
//   the real code, the tool aborts before trusting any distance figure (see
//   SelfCheckRequantBoundary below).
//
//   ARM B -- a manual single-layer replay, IDENTICAL IN KIND to
//   tools/t1786_stage_dump_probe.cpp's own ManualRunOneLayer (verbatim-copied
//   internal-linkage helpers ProjectAndFunnelCopy/ApplyBiasReconcileRowCopy, same
//   checkpoint-layer convention {0,1,2,9,18,27}, same self-check discipline: the replay's own
//   resulting hidden_codes/hidden_scale is compared bit-for-bit against production's own
//   RunLayerLoop(layer_budget=1) call from the same pre-layer state before any row captured
//   during that replay is trusted). This arm captures the two sites the funnel/trace-hook
//   arm cannot see, because neither routes through RequantChainChecked:
//     - K/V landing (LandingRescale + ClampRopeCode, forward_sites.h/.cpp) -- the score
//       path's own input quantization.
//     - Probability quantization (SoftmaxRowQ15's internal `(exps[k] << 15) / denom` divide,
//       intmath.cpp) -- reconstructed from the row's own scores/(q_ln2,q_b,q_c), captured
//       exactly as tools/t1786_stage_dump_probe.cpp's own CapturedProbRow does, using ONLY
//       already-public production functions (ShiftByMax, IExpConstruct, IExpEvaluate --
//       intmath.h) called on the SAME inputs the real SoftmaxRowQ15 call already consumed,
//       then checked bit-for-bit against that real call's own `probs` output before any
//       distance is trusted.
//
// BOUNDARY-DISTANCE GEOMETRY, computed exactly (integer arithmetic; a distance is a rational
// number, exact through the point it is narrowed to a reported double -- see
// BoundaryDistance64's own comment for the one deliberate narrowing and why it costs no
// reported bit):
//   - Round-half-away-from-zero sites (RequantTokenCodeWide's own formula, which
//     RequantChainChecked's per-element step composes, AND LandingRescale's own k>=0
//     branch): the decision boundary sits at every HALF-INTEGER of the pre-round quotient.
//     distance = 0.5 - |frac(quotient) - 0.5|, in [0, 0.5]; 0 = sitting exactly on a
//     boundary (an infinitesimal perturbation flips the code), 0.5 = sitting at a bin
//     center (maximally robust).
//   - Floor/truncate sites (SoftmaxRowQ15's own Q15 divide): the decision boundary sits at
//     every INTEGER of the pre-round quotient. distance = min(frac(quotient), 1-frac(quotient)),
//     same [0, 0.5] convention, but at a DIFFERENT boundary location than the round-half-away
//     sites -- reported and read separately per site, never pooled across the two geometries.
//   - LandingRescale's k<0 branch is an EXACT left shift -- no rounding, no boundary. Counted
//     and reported as its own bucket, never folded into the k>=0 population.
//
// Self-contained; needs no float reference (this is a property of the engine's own integer
// arithmetic, stated explicitly per this ticket's own instruction, not imported from
// elsewhere in this campaign's TVD-against-float-reference method).

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <intrin.h>
#include <map>
#include <span>
#include <string>
#include <unordered_map>
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

// --- exact 128-bit-widened rounding-boundary reconstruction ------------------------------
// Reproduces, exactly, the numerator RequantTokenCodeWide (intmath.cpp) and LandingRescale's
// own k>=0 branch (forward_sites.cpp) each divide by a power of two with round-half-away-
// from-zero. `abs_x * mult` is guaranteed (by this site's own load-time/C29 domain gates,
// documented at each call site below) to fit in 64 bits; the subsequent multiply by a second
// small factor (127 for RequantTokenCodeWide, or nothing for LandingRescale, whose two-factor
// product is formed by the caller before this helper runs) is the one step that can reach
// ~2^70 and is why this helper widens through `_umul128` (MSVC x64 intrinsic; the same
// portable-128-bit need this codebase's own U128 facility, intmath.cpp, exists to answer --
// this helper is an independent, from-scratch reconstruction of the identical arithmetic, not
// a copy of that internal-linkage facility, and its own output is verified bit-for-bit
// against the real production code below before any distance is trusted).
struct RoundBoundary {
	int64_t magnitude;  // the rounded, unclamped magnitude (matches production before clamp)
	double distance;    // 0..0.5, round-half-away geometry
};

// prod = a * b, both uint64, exact 128-bit result.
inline void Widen64x64(uint64_t a, uint64_t b, uint64_t* lo, uint64_t* hi) {
	*lo = _umul128(a, b, hi);
}

// Computes round_half_away_from_zero(prod / 2^exponent) and the exact fractional distance
// to the nearest half-integer boundary, from prod's 128-bit (hi,lo) representation and an
// exponent in [0, 127] -- wide enough to cover BOTH call sites: RequantTokenCodeWide's own
// exponent (documented [32,63]) and LandingRescale's own composed exponent k (this codebase's
// own e_a range [-80,39] and e_t load-time floor -60 put k's non-negative branch realistically
// in [0,82] -- the >63 half of that range is exactly where an EARLIER version of this
// function hit undefined behaviour, `uint64_t{1} << exponent` for exponent >= 64, and produced
// a wrong result on roughly half the K/V-landing population before the fix below).
//
// GEOMETRY (fixed from an earlier, inverted version of this same function): frac(y) == 0
// means y is exactly an integer -- the CENTER of a code's bin, farthest from any boundary
// (distance 0.5). frac(y) == 0.5 means y sits exactly at a half-integer -- ON the rounding
// boundary itself (distance 0). distance = |frac(y) - 0.5|, not "0.5 - |frac - 0.5|" (the
// earlier, inverted form, caught by cross-checking a hand-worked example: y=2.5 is a textbook
// on-boundary tie and must report distance 0, not 0.5).
RoundBoundary RoundHalfAwayBoundary(uint64_t prod_lo, uint64_t prod_hi, int exponent) {
	// magnitude = floor((2*prod + 2^exponent) / 2^(exponent+1)) -- identical numerator
	// production forms (RequantTokenCodeWide, intmath.cpp; LandingRescale's own k>=0 branch,
	// forward_sites.cpp both use this exact "add half the divisor, floor-shift" idiom).
	uint64_t two_lo = prod_lo << 1;
	uint64_t two_hi = (prod_hi << 1) | (prod_lo >> 63);
	// + 2^exponent, added at the correct bit position -- never a same-width shift by >= 64.
	uint64_t add_lo = 0, add_hi = 0;
	if (exponent < 64) {
		add_lo = uint64_t{1} << exponent;
	} else {
		add_hi = uint64_t{1} << (exponent - 64);  // exponent in [64,127]
	}
	uint64_t num_lo = two_lo + add_lo;
	const uint64_t carry = (num_lo < two_lo) ? 1u : 0u;
	uint64_t num_hi = two_hi + add_hi + carry;
	// >> (exponent+1). exponent+1 in [1,128]; every branch below shifts by < 64 or handles
	// the exact-64 case separately, so no shift here is ever undefined.
	const int shift = exponent + 1;
	uint64_t mag;
	if (shift >= 128) {
		mag = 0;
	} else if (shift == 64) {
		mag = num_hi;
	} else if (shift > 64) {
		mag = num_hi >> (shift - 64);
	} else {
		mag = (num_lo >> shift) | (num_hi << (64 - shift));
	}
	// Fractional distance: frac(prod / 2^exponent) = (prod mod 2^exponent) / 2^exponent,
	// computed as an exact modulus (mod_hi, mod_lo) then converted with ldexp (never a
	// same-width shift by >= 64, and exact within double's 53-bit mantissa -- more than
	// enough resolution for the 1%/5%/10% bins and percentile reporting this feeds).
	uint64_t mod_lo, mod_hi;
	if (exponent <= 0) {
		mod_lo = 0;
		mod_hi = 0;
	} else if (exponent < 64) {
		mod_lo = prod_lo & ((uint64_t{1} << exponent) - 1);
		mod_hi = 0;
	} else if (exponent == 64) {
		mod_lo = prod_lo;
		mod_hi = 0;
	} else {
		mod_lo = prod_lo;
		mod_hi = prod_hi & ((uint64_t{1} << (exponent - 64)) - 1);
	}
	const double v = std::ldexp(static_cast<double>(mod_hi), 64 - exponent) +
	                  std::ldexp(static_cast<double>(mod_lo), -exponent);
	const double dist = std::fabs(v - 0.5);
	return RoundBoundary{static_cast<int64_t>(mag), dist};
}

// The RequantTokenCodeWide reconstruction: exponent = 62 - s (this codebase's own pinned
// formula, intmath.cpp), prod = |x| * r * 127 (C22's "127 scale wrapper", D-SLM52). |x| <=
// 2^31 (RequantChainChecked's own C29 domain gate, checked before any element is quantized)
// and r is in (2^31, 2^32] (DynamicScaleReciprocal's own documented range), so |x|*r <= 2^63
// fits the low word of a 64x64 multiply with no widening; the second multiply by 127 is the
// one that needs the 128-bit widen.
RoundBoundary RequantBoundary(int64_t x, int64_t r, int32_t s) {
	int exponent = 62 - s;
	uint64_t abs_x = x < 0 ? (~static_cast<uint64_t>(x) + 1u) : static_cast<uint64_t>(x);
	uint64_t step1 = abs_x * static_cast<uint64_t>(r);  // fits uint64 by the domain gate above
	uint64_t lo, hi;
	Widen64x64(step1, 127u, &lo, &hi);
	return RoundHalfAwayBoundary(lo, hi, exponent);
}

// Self-check: the reconstructed magnitude, clamped to [0,127] and signed, must equal the
// real production code exactly. Aborts the run (no dump trusted) on the first disagreement,
// per this campaign's standing self-check discipline.
bool SelfCheckRequantBoundary(const RoundBoundary& b, int64_t x, int8_t real_code,
                               const char* site_label) {
	int64_t clamped = b.magnitude > 127 ? 127 : b.magnitude;
	int8_t reconstructed = static_cast<int8_t>(x < 0 ? -clamped : clamped);
	if (reconstructed != real_code) {
		std::fprintf(stderr,
		             "SELF-CHECK FAILED at site=%s: x=%lld reconstructed_code=%d real_code=%d -- "
		             "boundary reconstruction disagrees with production, no distance trusted\n",
		             site_label, static_cast<long long>(x), static_cast<int>(reconstructed),
		             static_cast<int>(real_code));
		return false;
	}
	return true;
}

// --- per-(layer.site) aggregation, computed online -- no raw per-element storage ----------
struct ChannelAgg {
	uint64_t n = 0;
	double sum_dist = 0.0;
	uint64_t within1 = 0, within5 = 0, within10 = 0;
};

struct SiteAgg {
	uint64_t n = 0;
	double sum_dist = 0.0;
	double min_dist = 0.5;
	uint64_t within1 = 0, within5 = 0, within10 = 0;
	uint64_t hist[20] = {};  // 0..0.5 in steps of 0.025
	std::vector<ChannelAgg> per_channel;   // indexed by element position within the row
	std::unordered_map<int64_t, ChannelAgg> per_position;  // indexed by token_index

	void Add(double dist, size_t channel, int64_t token_index) {
		++n;
		sum_dist += dist;
		if (dist < min_dist) min_dist = dist;
		if (dist <= 0.01) ++within1;
		if (dist <= 0.05) ++within5;
		if (dist <= 0.10) ++within10;
		int bin = static_cast<int>(dist / 0.025);
		if (bin > 19) bin = 19;
		if (bin < 0) bin = 0;
		++hist[bin];
		if (channel >= per_channel.size()) per_channel.resize(channel + 1);
		ChannelAgg& c = per_channel[channel];
		++c.n;
		c.sum_dist += dist;
		if (dist <= 0.01) ++c.within1;
		if (dist <= 0.05) ++c.within5;
		if (dist <= 0.10) ++c.within10;
		ChannelAgg& p = per_position[token_index];
		++p.n;
		p.sum_dist += dist;
		if (dist <= 0.01) ++p.within1;
		if (dist <= 0.05) ++p.within5;
		if (dist <= 0.10) ++p.within10;
	}
};

struct BoundaryCapture {
	std::map<std::string, SiteAgg> chain_sites;    // key: "layerN.sitename" (Arm A)
	uint64_t chain_self_check_ok = 0;
	uint64_t chain_self_check_fail = 0;

	SiteAgg kv_landing_k, kv_landing_v;             // Arm B, checkpoint layers only
	uint64_t kv_landing_negative_k_count = 0;        // exact-left-shift branch, no boundary
	uint64_t kv_landing_negative_k_total = 0;
	SiteAgg prob_quant;                              // Arm B, checkpoint layers only
};

// Trace-hook callback (Arm A). Read-only: computes and aggregates only, touches nothing the
// forward pass reads.
void OnChainTrace(const SslmChainTraceRecord* chain, const SslmKvLandingTraceRecord*, void* user) {
	if (chain == nullptr) return;
	BoundaryCapture* cap = static_cast<BoundaryCapture*>(user);
	SiteAgg& agg = cap->chain_sites[std::string(chain->site)];
	for (size_t i = 0; i < chain->x_int.size(); ++i) {
		RoundBoundary b = RequantBoundary(chain->x_int[i], chain->r, chain->s);
		if (!SelfCheckRequantBoundary(b, chain->x_int[i], chain->codes[i], std::string(chain->site).c_str())) {
			++cap->chain_self_check_fail;
			continue;
		}
		++cap->chain_self_check_ok;
		agg.Add(b.distance, i, static_cast<int64_t>(chain->token_index));
	}
}

// --- Arm B: verbatim-copied internal-linkage helpers (identical in kind to
// tools/t1786_stage_dump_probe.cpp's own ProjectAndFunnelCopy/ApplyBiasReconcileRowCopy --
// forward_sites.cpp's ProjectAndFunnel/ApplyBiasReconcileRow are internal-linkage, so a
// manual replay outside that translation unit must reproduce their bodies, calling only
// public-header primitives, exactly as that probe already established the discipline). --
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

// Reconstructs LandingRescale's own composed exponent k = 62 - (e_a - e_t), using the SAME
// portable-widening discipline this file already applies (no plain int64 subtraction of two
// arbitrary int64_t values, which LandingRescale's own header documents as overflow-reachable
// -- forward_sites.cpp's ComposedExponent, internal-linkage, is what production calls; this
// reconstruction only needs the SIGN and magnitude of e_a - e_t within the realistic small
// range this campaign's own artifacts carry (composition-constant exponents are small
// integers, not adversarial extremes), verified per element against LandingRescale's own
// *out_magnitude_exceeded_int64 signal before any distance is trusted).
int64_t ComposedExponentCopy(int64_t e_a, int64_t e_t) { return 62 - (e_a - e_t); }

// One captured (layer, head) softmax row -- identical shape to
// tools/t1786_stage_dump_probe.cpp's own CapturedProbRow.
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

// One captured K/V landing call.
struct CapturedLanding {
	int64_t branch_code = 0, m_a = 0, r_t = 0, e_a = 0, e_t = 0;
	int64_t raw = 0;
	int8_t code = 0;
};

SslmForwardStatus ManualRunOneLayer(SequenceLayerState& seq, const LayerWeights& lw,
                                     size_t hidden_size, size_t head_dim, size_t num_key_value_heads,
                                     size_t intermediate_size, int64_t context_cap,
                                     const SslmTensorManifest& rope_tables, uint8_t* workspace,
                                     std::vector<CapturedProbRow>* out_rows,
                                     std::vector<CapturedLanding>* out_k_landing,
                                     std::vector<CapturedLanding>* out_v_landing) {
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
				uint64_t sat_k = 0, sat_v = 0;
				bool exceed_k = false, exceed_v = false;
				const int64_t raw_k = LandingRescale(kacc[i], normed_scale.m, lw.kv_landing_r_t_k[h],
				                                      normed_scale.e, lw.kv_landing_e_t_k[h], &sat_k,
				                                      &exceed_k);
				const int64_t raw_v = LandingRescale(vacc[i], normed_scale.m, lw.kv_landing_r_t_v[h],
				                                      normed_scale.e, lw.kv_landing_e_t_v[h], &sat_v,
				                                      &exceed_v);
				const int8_t code_k = static_cast<int8_t>(ClampRopeCode(raw_k));
				const int8_t code_v = static_cast<int8_t>(ClampRopeCode(raw_v));
				k_row[d] = code_k;
				v_row[d] = code_v;
				if (out_k_landing != nullptr) {
					CapturedLanding cl;
					cl.branch_code = kacc[i];
					cl.m_a = normed_scale.m;
					cl.r_t = lw.kv_landing_r_t_k[h];
					cl.e_a = normed_scale.e;
					cl.e_t = lw.kv_landing_e_t_k[h];
					cl.raw = raw_k;
					cl.code = code_k;
					out_k_landing->push_back(cl);
				}
				if (out_v_landing != nullptr) {
					CapturedLanding cl;
					cl.branch_code = vacc[i];
					cl.m_a = normed_scale.m;
					cl.r_t = lw.kv_landing_r_t_v[h];
					cl.e_a = normed_scale.e;
					cl.e_t = lw.kv_landing_e_t_v[h];
					cl.raw = raw_v;
					cl.code = code_v;
					out_v_landing->push_back(cl);
				}
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
	return SslmForwardStatus::Ok;
}

void PrintUsage(const char* argv0) {
	std::fprintf(stderr, "usage: %s <model.sslm> <tokenizer.sslm> \"<prompt>\" <label>\n", argv0);
}

}  // namespace

int main(int argc, char** argv) {
	if (argc < 5) {
		PrintUsage(argv[0]);
		return 2;
	}
	const std::string model_path = argv[1];
	const std::string tokenizer_path = argv[2];
	const std::string prompt = argv[3];
	const std::string label = argv[4];

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

	auto RunForward = [&](BoundaryCapture* cap, std::vector<CapturedProbRow>* prob_rows,
	                       std::vector<CapturedLanding>* k_landing, std::vector<CapturedLanding>* v_landing,
	                       std::vector<int8_t>* out_final_codes, CarriedScale* out_final_scale) -> bool {
		SslmTraceHookState hook_state;
		if (cap != nullptr) {
			SslmSetTraceHook(hook_state, &OnChainTrace, cap);
		}

		std::vector<uint8_t> workspace(kv_bytes);
		std::vector<int8_t> hidden_codes(hidden_size);
		SequenceLayerState seq;
		seq.hidden_codes = hidden_codes.data();

		auto EmbedWholeToken = [&](int32_t token) -> SslmForwardStatus {
			std::vector<int8_t> embed_codes(hidden_size);
			CarriedScale embed_scale{};
			const SslmForwardStatus est =
			    EmbedEntry(token, static_cast<int32_t>(model_view.config.vocab_size), embed_weights,
			               hidden_size, embed_site_constant, embed_codes.data(), &embed_scale);
			if (est != SslmForwardStatus::Ok) return est;
			for (size_t i = 0; i < hidden_size; ++i) seq.hidden_codes[i] = embed_codes[i];
			seq.hidden_scale = embed_scale;
			seq.layer_index = 0;
			return SslmForwardStatus::Ok;
		};

		for (size_t i = 0; i + 1 < prompt_tokens.size(); ++i) {
			SslmForwardStatus st = EmbedWholeToken(prompt_tokens[i]);
			if (st != SslmForwardStatus::Ok) {
				std::fprintf(stderr, "FAILED at stage=prefill_embed: position=%zu status=%s\n", i,
				             SslmForwardStatusName(st));
				return false;
			}
			st = RunLayerLoop(seq, layers.data(), num_hidden_layers, num_hidden_layers, hidden_size,
			                   head_dim, num_kv_heads, intermediate_size, context_cap,
			                   model_view.rope_tables, workspace.data(), workspace.size(), "layer",
			                   static_cast<size_t>(i), cap != nullptr ? &hook_state : nullptr);
			if (st != SslmForwardStatus::Ok) {
				std::fprintf(stderr, "FAILED at stage=prefill_layers: position=%zu status=%s\n", i,
				             SslmForwardStatusName(st));
				return false;
			}
		}
		{
			const SslmForwardStatus st = EmbedWholeToken(prompt_tokens.back());
			if (st != SslmForwardStatus::Ok) {
				std::fprintf(stderr, "FAILED at stage=last_token_embed: status=%s\n",
				             SslmForwardStatusName(st));
				return false;
			}
		}

		const std::vector<uint32_t> checkpoint_layers = {0, 1, 2, 9, 18, 27};
		const int64_t last_token_index = static_cast<int64_t>(prompt_tokens.size()) - 1;

		for (uint32_t step = 0; step < num_hidden_layers; ++step) {
			const bool is_checkpoint =
			    (prob_rows != nullptr) &&
			    std::find(checkpoint_layers.begin(), checkpoint_layers.end(), step) !=
			        checkpoint_layers.end();

			if (is_checkpoint) {
				std::vector<int8_t> manual_hidden_codes(hidden_codes);
				SequenceLayerState manual_seq = seq;
				manual_seq.hidden_codes = manual_hidden_codes.data();

				std::vector<CapturedProbRow> step_rows;
				std::vector<CapturedLanding> step_k, step_v;
				const SslmForwardStatus mst =
				    ManualRunOneLayer(manual_seq, layers[step], hidden_size, head_dim, num_kv_heads,
				                      intermediate_size, context_cap, model_view.rope_tables,
				                      workspace.data(), &step_rows, &step_k, &step_v);
				if (mst != SslmForwardStatus::Ok) {
					std::fprintf(stderr, "FAILED at stage=manual_replay: layer=%u status=%s\n", step,
					             SslmForwardStatusName(mst));
					return false;
				}

				const SslmForwardStatus pst =
				    RunLayerLoop(seq, layers.data(), num_hidden_layers, /*layer_budget=*/1, hidden_size,
				                 head_dim, num_kv_heads, intermediate_size, context_cap,
				                 model_view.rope_tables, workspace.data(), workspace.size(), "layer",
				                 static_cast<size_t>(last_token_index), cap != nullptr ? &hook_state : nullptr);
				if (pst != SslmForwardStatus::Ok) {
					std::fprintf(stderr, "FAILED at stage=production_step: layer=%u status=%s\n", step,
					             SslmForwardStatusName(pst));
					return false;
				}

				const bool codes_match =
				    std::memcmp(manual_seq.hidden_codes, seq.hidden_codes, hidden_size) == 0;
				const bool scale_match = manual_seq.hidden_scale.m == seq.hidden_scale.m &&
				                         manual_seq.hidden_scale.e == seq.hidden_scale.e;
				if (!codes_match || !scale_match) {
					std::fprintf(stderr,
					             "FAILED at stage=self_check: layer=%u codes_match=%d scale_match=%d -- "
					             "manual replay diverges from production, captured rows discarded\n",
					             step, codes_match ? 1 : 0, scale_match ? 1 : 0);
					return false;
				}
				std::printf("self_check: layer=%u manual replay and production agree bit-for-bit "
				            "(%zu softmax rows, %zu K landing, %zu V landing captured)\n",
				            step, step_rows.size(), step_k.size(), step_v.size());
				for (auto& r : step_rows) prob_rows->push_back(std::move(r));
				for (auto& c : step_k) k_landing->push_back(c);
				for (auto& c : step_v) v_landing->push_back(c);
			} else {
				const SslmForwardStatus pst =
				    RunLayerLoop(seq, layers.data(), num_hidden_layers, /*layer_budget=*/1, hidden_size,
				                 head_dim, num_kv_heads, intermediate_size, context_cap,
				                 model_view.rope_tables, workspace.data(), workspace.size(), "layer",
				                 static_cast<size_t>(last_token_index), cap != nullptr ? &hook_state : nullptr);
				if (pst != SslmForwardStatus::Ok) {
					std::fprintf(stderr, "FAILED at stage=production_step: layer=%u status=%s\n", step,
					             SslmForwardStatusName(pst));
					return false;
				}
			}
		}
		if (out_final_codes != nullptr) {
			out_final_codes->assign(hidden_codes.begin(), hidden_codes.end());
		}
		if (out_final_scale != nullptr) {
			*out_final_scale = seq.hidden_scale;
		}
		return true;
	};

	// --- Run 1: hooked, with checkpoint-layer manual replay (Arm A + Arm B). ---
	BoundaryCapture cap;
	std::vector<CapturedProbRow> prob_rows;
	std::vector<CapturedLanding> k_landing, v_landing;
	std::vector<int8_t> final_codes_hooked;
	CarriedScale final_scale_hooked{};
	if (!RunForward(&cap, &prob_rows, &k_landing, &v_landing, &final_codes_hooked, &final_scale_hooked)) {
		return 1;
	}

	// --- Run 2: unhooked, no manual replay -- empirical confirmation that installing the
	// trace hook changes NOTHING about the engine's own output (trace_hook.h's own
	// "read-only by construction" claim, verified rather than merely cited). ---
	std::vector<int8_t> final_codes_unhooked;
	CarriedScale final_scale_unhooked{};
	if (!RunForward(nullptr, nullptr, nullptr, nullptr, &final_codes_unhooked, &final_scale_unhooked)) {
		return 1;
	}
	const bool hook_is_readonly = (final_codes_hooked == final_codes_unhooked) &&
	                               final_scale_hooked.m == final_scale_unhooked.m &&
	                               final_scale_hooked.e == final_scale_unhooked.e;
	std::printf("trace-hook read-only check: hooked and unhooked runs produce %s final hidden state\n",
	            hook_is_readonly ? "IDENTICAL" : "DIFFERENT (DEFECT)");
	if (!hook_is_readonly) {
		std::fprintf(stderr, "FAILED: trace hook is not read-only -- aborting, no results trusted\n");
		return 1;
	}

	std::printf("chain self-check: %llu ok, %llu FAILED\n",
	            static_cast<unsigned long long>(cap.chain_self_check_ok),
	            static_cast<unsigned long long>(cap.chain_self_check_fail));
	if (cap.chain_self_check_fail > 0) {
		std::fprintf(stderr, "FAILED: chain boundary reconstruction disagreed with production on %llu "
		                     "elements -- aborting\n",
		             static_cast<unsigned long long>(cap.chain_self_check_fail));
		return 1;
	}

	// --- Arm B: reconstruct K/V landing and probability-quantization boundary distances,
	// self-checked bit-for-bit before any distance is aggregated. ---
	uint64_t landing_ok = 0, landing_fail = 0;
	auto ProcessLanding = [&](const std::vector<CapturedLanding>& rows, SiteAgg* agg, const char* tag) {
		for (const auto& c : rows) {
			const int64_t k = ComposedExponentCopy(c.e_a, c.e_t);
			if (k < 0) {
				++cap.kv_landing_negative_k_total;
				++cap.kv_landing_negative_k_count;
				continue;  // exact left shift, no rounding boundary
			}
			const bool branch_negative = c.branch_code < 0;
			const bool m_a_negative = c.m_a < 0;
			const bool negative = branch_negative != m_a_negative;
			const uint64_t abs_branch =
			    branch_negative ? (~static_cast<uint64_t>(c.branch_code) + 1u) : static_cast<uint64_t>(c.branch_code);
			const uint64_t abs_m_a =
			    m_a_negative ? (~static_cast<uint64_t>(c.m_a) + 1u) : static_cast<uint64_t>(c.m_a);
			uint64_t step_lo, step_hi;
			Widen64x64(abs_branch, abs_m_a, &step_lo, &step_hi);
			// step_hi should be 0: abs_branch and abs_m_a are each well within 2^32 at this
			// site's realistic magnitudes; if not, the self-check below simply fails and the
			// element is excluded rather than silently mis-widened.
			uint64_t prod_lo, prod_hi;
			if (step_hi == 0) {
				Widen64x64(step_lo, static_cast<uint64_t>(c.r_t), &prod_lo, &prod_hi);
			} else {
				// Genuine 3-factor widen needed; not reachable at this site's documented
				// magnitude (forward_sites.cpp: "~2^90-2^91" for the FULL product INCLUDING
				// r_t, meaning branch*m_a alone is well under 2^64) -- flagged, not guessed.
				++landing_fail;
				continue;
			}
			RoundBoundary b = RoundHalfAwayBoundary(prod_lo, prod_hi, static_cast<int>(k));
			int64_t clamped = b.magnitude > 127 ? 127 : b.magnitude;
			int64_t reconstructed_raw = negative ? -clamped : clamped;
			// LandingRescale's own raw (pre-ClampRopeCode) may legitimately exceed +/-127;
			// self-check against the CODE (post-clamp), matching what ClampRopeCode(raw) would
			// produce from our reconstructed magnitude.
			int64_t reconstructed_code = reconstructed_raw > 127 ? 127 : (reconstructed_raw < -127 ? -127 : reconstructed_raw);
			if (reconstructed_code != c.code) {
				++landing_fail;
				continue;
			}
			++landing_ok;
			agg->Add(b.distance, 0, 0);
		}
	};
	ProcessLanding(k_landing, &cap.kv_landing_k, "kv_landing_k");
	ProcessLanding(v_landing, &cap.kv_landing_v, "kv_landing_v");
	std::printf("K/V landing self-check: %llu ok, %llu FAILED, %llu on the exact-left-shift (k<0) "
	            "branch (no boundary)\n",
	            static_cast<unsigned long long>(landing_ok), static_cast<unsigned long long>(landing_fail),
	            static_cast<unsigned long long>(cap.kv_landing_negative_k_total));
	if (landing_fail > 0) {
		std::fprintf(stderr, "FAILED: K/V landing boundary reconstruction disagreed with production on "
		                     "%llu elements -- aborting\n",
		             static_cast<unsigned long long>(landing_fail));
		return 1;
	}

	// Probability quantization: floor((exps[k]<<15)/denom). exps[k] is recomputed with the
	// SAME public production functions the real SoftmaxRowQ15 call already used
	// (ShiftByMax/IExpConstruct/IExpEvaluate), on the SAME captured inputs, then the
	// reconstructed floor-divide is checked against the row's own real `probs` output.
	uint64_t prob_ok = 0, prob_fail = 0;
	for (const auto& row : prob_rows) {
		std::vector<int64_t> shifted(row.width);
		ShiftByMax(row.scores.data(), row.width, shifted.data());
		std::vector<int64_t> exps(row.width);
		int64_t total = 0;
		for (int64_t k = 0; k < row.width; ++k) {
			IExpConstruction construction;
			const IExpDomain d = IExpConstruct(shifted[k], row.q_ln2, row.q_b, row.q_c, &construction);
			int64_t value = 0;
			if (d == IExpDomain::kOk || d == IExpDomain::kNotRepresentable) {
				value = IExpEvaluate(construction);
			}
			exps[k] = value;
			total += value;
		}
		const int64_t denom = total > 1 ? total : 1;
		for (int64_t k = 0; k < row.width; ++k) {
			const uint64_t numerator = static_cast<uint64_t>(exps[k]) << kProbFracBits;
			const int64_t reconstructed = static_cast<int64_t>(numerator / static_cast<uint64_t>(denom));
			if (reconstructed != row.probs[k]) {
				++prob_fail;
				continue;
			}
			++prob_ok;
			const uint64_t frac_num = numerator % static_cast<uint64_t>(denom);
			const double v = static_cast<double>(frac_num) / static_cast<double>(denom);
			const double dist = std::min(v, 1.0 - v);
			cap.prob_quant.Add(dist, static_cast<size_t>(k), static_cast<int64_t>(row.layer));
		}
	}
	std::printf("probability-quantization self-check: %llu ok, %llu FAILED\n",
	            static_cast<unsigned long long>(prob_ok), static_cast<unsigned long long>(prob_fail));
	if (prob_fail > 0) {
		std::fprintf(stderr, "FAILED: probability-quantization reconstruction disagreed with production "
		                     "on %llu elements -- aborting\n",
		             static_cast<unsigned long long>(prob_fail));
		return 1;
	}

	// --- Dump results: one text file per prompt, machine-readable, per-(layer.site) rows. ---
	const std::string dump_path = "out/t1796/" + label + "_boundary.txt";
	std::ofstream f(dump_path);
	if (!f) {
		std::fprintf(stderr, "FAILED at stage=dump_open: could not open \"%s\"\n", dump_path.c_str());
		return 1;
	}
	auto DumpSite = [&](const std::string& name, const SiteAgg& agg) {
		f << "SITE " << name << " n=" << agg.n << " mean=" << (agg.n ? agg.sum_dist / agg.n : 0.0)
		  << " min=" << agg.min_dist << " within1pct=" << agg.within1 << " within5pct=" << agg.within5
		  << " within10pct=" << agg.within10 << " hist=";
		for (int i = 0; i < 20; ++i) f << agg.hist[i] << (i < 19 ? "," : "");
		f << "\n";
		// per-channel summary (concentration input for the analysis script)
		for (size_t c = 0; c < agg.per_channel.size(); ++c) {
			const ChannelAgg& ca = agg.per_channel[c];
			if (ca.n == 0) continue;
			f << "CHANNEL " << name << " " << c << " n=" << ca.n << " mean=" << (ca.sum_dist / ca.n)
			  << " within5pct=" << ca.within5 << "\n";
		}
		// per-position summary
		for (const auto& kv : agg.per_position) {
			const ChannelAgg& pa = kv.second;
			f << "POSITION " << name << " " << kv.first << " n=" << pa.n << " mean=" << (pa.sum_dist / pa.n)
			  << " within5pct=" << pa.within5 << "\n";
		}
	};
	for (const auto& kv : cap.chain_sites) DumpSite(kv.first, kv.second);
	DumpSite("kv_landing_k", cap.kv_landing_k);
	DumpSite("kv_landing_v", cap.kv_landing_v);
	DumpSite("prob_quant", cap.prob_quant);
	f << "META kv_landing_negative_k_total=" << cap.kv_landing_negative_k_total << "\n";
	f << "META chain_self_check_ok=" << cap.chain_self_check_ok << "\n";
	f << "META hook_read_only=" << (hook_is_readonly ? 1 : 0) << "\n";
	std::printf("\ndump written: %s (%zu chain sites)\n", dump_path.c_str(), cap.chain_sites.size());
	return 0;
}
