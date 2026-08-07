// t1797_residual_arms.cpp -- T-1797 E1/E2: full manual replay of the engine's forward with
// the RESIDUAL STREAM's representation as the one experimental variable.
//
// Arms:
//   k0  -- the residual stream carried exactly as production carries it (int8 codes), but
//          through this tool's own widened machinery. EVERY site call is DUAL-RUN against
//          the production site function (RmsNormSite / ResidualReconcileSite /
//          RequantChainChecked) and must agree bit-for-bit on every element of every row of
//          every layer of every position -- this arm IS the machinery's self-check, and its
//          dumped states must additionally equal the production stepping dump
//          (t1797_multipos_probe) exactly.
//   k4/k8 -- the residual stream carries k extra bits (codes in [-127*2^k, 127*2^k], scale
//          exponent lowered by k). Same shared-max-abs scale scheme, same sites, same
//          arithmetic shape -- ONLY the residual grid resolution changes. This is the
//          middle-link dose-response: if the drift is driven by residual-site
//          representation error, it closes as k rises.
//   rot -- the residual stream carried as int8 codes OF THE ROTATED row: at each residual
//          site the wide row is first taken to a fine (k=8) grid, then sign-flipped,
//          permuted, and block-Hadamard-transformed (3 blocks of 512, add/subtract only,
//          exact in int64), and THAT row is quantized to int8 by the production funnel
//          (RequantChainChecked). Consumers reconstruct the unrotated row exactly
//          (H*H = 512*I) and carry the 2^-9 in the scale exponent. The information
//          bottleneck is the stored int8 rotated codes -- the construction under test.
//
// Everything outside the residual representation -- all projections, K/V landing, RoPE,
// attention, softmax, MLP, and every quantizer -- is the production site function itself,
// exactly as tools/t1795_residual_probe.cpp's ManualRunOneLayer (copied here) calls them.
//
// Dump: raw float32 [position][state 0..29-1][hidden] identical in layout to
// t1797_multipos_probe.cpp's, so one analysis loader serves both; plus per-arm
// last-position boundary rows and site codes for conversion-curve analysis, and timing.

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <numeric>
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

namespace {

// ---------------------------------------------------------------- basic helpers

double Dequant(int64_t code, CarriedScale scale) {
	return static_cast<double>(code) * static_cast<double>(scale.m) *
	       std::pow(2.0, static_cast<double>(scale.e));
}

// Portable 64x64 -> 128 unsigned multiply via 32-bit limbs (this tool's own copy; MSVC has
// no __int128). Self-checked at k=0 by the elementwise dual-run against production's own
// RequantTokenCodeWide, which uses intmath.cpp's own 128-bit facility.
struct U128 {
	uint64_t lo, hi;
};
U128 UMul64(uint64_t a, uint64_t b) {
	const uint64_t a_lo = a & 0xFFFFFFFFull, a_hi = a >> 32;
	const uint64_t b_lo = b & 0xFFFFFFFFull, b_hi = b >> 32;
	const uint64_t p0 = a_lo * b_lo;
	const uint64_t p1 = a_lo * b_hi;
	const uint64_t p2 = a_hi * b_lo;
	const uint64_t p3 = a_hi * b_hi;
	const uint64_t mid = (p0 >> 32) + (p1 & 0xFFFFFFFFull) + (p2 & 0xFFFFFFFFull);
	U128 r;
	r.lo = (mid << 32) | (p0 & 0xFFFFFFFFull);
	r.hi = p3 + (p1 >> 32) + (p2 >> 32) + (mid >> 32);
	return r;
}
uint64_t UShr128To64(U128 v, int k) {  // k in [1,63]
	return (v.lo >> k) | (v.hi << (64 - k));
}

// round_half_away_from_zero((x * 127 * r) / 2^e), clamped to +/- clamp_mag.
// x is int64 with |x| <= 2^31 (C29-checked upstream), r in (2^31, 2^32], e in [1,63].
int64_t RequantCodeWideK(int64_t x, int64_t r, int e, int64_t clamp_mag) {
	const bool neg = x < 0;
	const uint64_t ax = neg ? static_cast<uint64_t>(-(x + 1)) + 1u : static_cast<uint64_t>(x);
	U128 prod = UMul64(ax * 127u, static_cast<uint64_t>(r));  // ax<=2^31 so ax*127 < 2^38 exact
	// rounded = floor(prod / 2^e) + (remainder >= half ? 1 : 0); half-away == half-up on
	// magnitude then sign restored (ties away from zero).
	const uint64_t shifted = UShr128To64(prod, e);
	const uint64_t half_bit = (e >= 1)
	    ? ((e <= 64 ? ((e == 64) ? (prod.hi & 1u) : ((prod.lo >> (e - 1)) & 1u))
	                : ((prod.hi >> (e - 65)) & 1u)))
	    : 0u;
	uint64_t mag = shifted + half_bit;
	if (mag > static_cast<uint64_t>(clamp_mag)) mag = static_cast<uint64_t>(clamp_mag);
	int64_t out = static_cast<int64_t>(mag);
	return neg ? -out : out;
}

// ------------------------------------------------------- the wide funnel (this tool's)

struct WideRequantOut {
	std::vector<int64_t> codes;
	CarriedScale scale;
	// captured for boundary analysis
	int64_t d_prime = 0;
	int64_t r = 0;
	int s = 0;
};

// The production funnel's own steps (RequantChainChecked, checked_chain_funnel.cpp, read in
// full and mirrored here) with the ONE change: the emitted code grid carries k extra bits
// (multiply the quotient by 2^k == lower the rounding divide's exponent by k), clamp at
// +/-127*2^k, and lower the folded scale exponent by k. At k == 0 this is the production
// funnel exactly, asserted elementwise by the caller's dual-run.
bool RequantChainWideK(const int64_t* wide_row, size_t n, std::span<const CarriedScale> incoming,
                       CarriedScale site_constant, int k, WideRequantOut* out,
                       bool include_site = true) {
	const int64_t d_raw = MaxAbsReduceWide(wide_row, n);
	if (d_raw > (int64_t{1} << 31)) return false;  // C29
	const NormalizedScale ns = NormalizeScale(d_raw);
	const int64_t r = DynamicScaleReciprocal(ns.dn);
	// left-associated fold, exactly production's order: incoming..., site, D'-factor
	const CarriedScale d_prime_factor{ns.dn, -static_cast<int64_t>(ns.s)};
	bool have = false;
	CarriedScale running{};
	auto fold_in = [&](const CarriedScale& next) {
		running = have ? CombineCarriedScale(running, next) : next;
		have = true;
	};
	for (const CarriedScale& f : incoming) fold_in(f);
	if (include_site) fold_in(site_constant);
	fold_in(d_prime_factor);
	out->codes.resize(n);
	const int e = 62 - ns.s - k;
	const int64_t clamp_mag = int64_t{127} << k;
	for (size_t i = 0; i < n; ++i) {
		out->codes[i] = RequantCodeWideK(wide_row[i], r, e, clamp_mag);
	}
	out->scale = CarriedScale{running.m, running.e - k};
	out->d_prime = d_raw;
	out->r = r;
	out->s = ns.s;
	return true;
}

// ------------------------------------------------------------- rotation machinery

constexpr size_t kHadamardBlock = 512;
constexpr int kHadamardLog2 = 9;

struct RotationPlan {
	std::vector<uint32_t> perm;     // rotated slot j reads original channel perm[j]
	std::vector<int32_t> sign;      // +/-1 per rotated slot j (applied before FWHT)
	std::vector<uint32_t> inv_perm; // original channel i is read by rotated slot inv_perm[i]
};

uint64_t SplitMix64(uint64_t* state) {
	uint64_t z = (*state += 0x9E3779B97F4A7C15ull);
	z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
	z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
	return z ^ (z >> 31);
}

RotationPlan MakeRotationPlan(size_t n, uint64_t seed) {
	RotationPlan p;
	p.perm.resize(n);
	std::iota(p.perm.begin(), p.perm.end(), 0u);
	uint64_t st = seed;
	for (size_t i = n - 1; i > 0; --i) {
		const size_t j = static_cast<size_t>(SplitMix64(&st) % (i + 1));
		std::swap(p.perm[i], p.perm[j]);
	}
	p.sign.resize(n);
	for (size_t i = 0; i < n; ++i) p.sign[i] = (SplitMix64(&st) & 1u) ? 1 : -1;
	p.inv_perm.resize(n);
	for (size_t j = 0; j < n; ++j) p.inv_perm[p.perm[j]] = static_cast<uint32_t>(j);
	return p;
}

// In-place fast Walsh-Hadamard transform per 512-block: add/subtract only, exact in int64.
void FwhtBlocks(int64_t* x, size_t n) {
	for (size_t base = 0; base < n; base += kHadamardBlock) {
		for (size_t len = 1; len < kHadamardBlock; len <<= 1) {
			for (size_t i = 0; i < kHadamardBlock; i += len << 1) {
				for (size_t j = 0; j < len; ++j) {
					const int64_t a = x[base + i + j];
					const int64_t b = x[base + i + j + len];
					x[base + i + j] = a + b;
					x[base + i + j + len] = a - b;
				}
			}
		}
	}
}

// forward: y[j] = FWHT( sign[j] * x[perm[j]] )
void RotateForward(const RotationPlan& p, const int64_t* x, int64_t* y, size_t n) {
	for (size_t j = 0; j < n; ++j) y[j] = static_cast<int64_t>(p.sign[j]) * x[p.perm[j]];
	FwhtBlocks(y, n);
}
// reconstruct: x_rec[perm[j]] = sign[j] * FWHT(y)[j]  (== 512 * x when y = forward(x))
void RotateReconstruct(const RotationPlan& p, const int64_t* y, int64_t* x_rec, size_t n,
                       std::vector<int64_t>* scratch) {
	scratch->assign(y, y + n);
	FwhtBlocks(scratch->data(), n);
	for (size_t j = 0; j < n; ++j) x_rec[p.perm[j]] = static_cast<int64_t>(p.sign[j]) * (*scratch)[j];
}

// ------------------------------------------------------------- production-copied bits

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

// ------------------------------------------------------------- the wide residual sites

enum class Arm { kWide, kRot };

struct ArmConfig {
	Arm arm = Arm::kWide;
	int k = 0;         // extra residual bits (kWide); the fine-grid k for kRot is fixed 8
	int norm_j = 0;    // norm input down-shift compensation (== excess input bits over int8)
	uint64_t seed = 0x1797C0FFEEull;
};

struct WideState {
	std::vector<int64_t> codes;  // unrotated residual codes (reconstruction, for kRot)
	CarriedScale scale;          // dequant = codes * m * 2^e directly (all adjustments folded)
};

struct SelfCheckCounters {
	uint64_t norm_calls = 0, norm_mismatch = 0;
	uint64_t reconcile_calls = 0, reconcile_mismatch = 0;
};

struct BoundaryDump {
	std::string site;
	std::vector<int64_t> row;
	int64_t r = 0;
	int s = 0;
	int k = 0;
};

// The production RmsNormSite formula (forward_sites.cpp, read in full), generalized to wide
// input codes by the j-shift compensation: with j == 0 and int8-valued codes this IS the
// production formula, asserted by dual-run. RMSNorm is invariant to the input's scale, so
// only the codes enter.
SslmForwardStatus WideRmsNorm(const std::vector<int64_t>& h, const int32_t* g, size_t hidden_size,
                              CarriedScale site_constant, int norm_j, int8_t* out_codes,
                              CarriedScale* out_scale) {
	constexpr int kNormFracBits = 16;  // forward_sites.cpp:38
	int64_t sumsq = 0;
	for (size_t i = 0; i < hidden_size; ++i) sumsq += h[i] * h[i];
	// production: root = max(ISqrt(FloorDivI64(sumsq << 32, n)), 1)
	// wide: shift by (32 - 2j) instead, i.e. root' ~= root / 2^j -- exact split division so
	// the j == 0 case is bit-identical to the production expression.
	const int shift_a = 2 * kNormFracBits - 2 * norm_j;
	const int64_t n64 = static_cast<int64_t>(hidden_size);
	const int64_t d1 = FloorDivI64(sumsq, n64);
	const int64_t r1 = sumsq - d1 * n64;
	const int64_t q = (d1 << shift_a) + FloorDivI64(r1 << shift_a, n64);
	const int64_t root = std::max<int64_t>(ISqrt(q), 1);
	std::vector<int64_t> wide(hidden_size);
	const int shift_b = 2 * kNormFracBits - norm_j;
	for (size_t i = 0; i < hidden_size; ++i) {
		wide[i] = FloorDivI64(h[i] << shift_b, root) * static_cast<int64_t>(g[i]);
	}
	const ChainResult res = RequantChainChecked(wide.data(), hidden_size,
	                                            std::span<const CarriedScale>{}, site_constant,
	                                            out_codes, out_scale);
	return res.status;
}

// One residual reconciliation at wide stream width, then the arm's own quantization.
// Returns false on any domain rejection (aborts the run; logged by caller).
bool WideResidualSite(const ArmConfig& cfg, const RotationPlan& rot, const int8_t* branch_code,
                      CarriedScale branch_scale, WideState* stream, CarriedScale site_constant,
                      const CarriedScale& ident_site, size_t hidden_size,
                      SelfCheckCounters* sc, bool capture_boundary, const char* site_name,
                      std::vector<BoundaryDump>* boundary_out,
                      std::vector<int64_t>* scratch) {
	// production steps 0-3 (ResidualReconcileSite, forward_sites.h contract):
	if (!(stream->scale.m >= INT32_MIN && stream->scale.m <= INT32_MAX)) return false;
	const int64_t r_h = CarriedScaleReciprocal(stream->scale.m);
	std::vector<int64_t> wide(hidden_size);
	for (size_t i = 0; i < hidden_size; ++i) {
		bool exceeded = false;
		const int64_t rec = LandingRescale(branch_code[i], branch_scale.m, r_h, branch_scale.e,
		                                   stream->scale.e, nullptr, &exceeded);
		if (exceeded) return false;
		wide[i] = rec + stream->codes[i];
	}
	const CarriedScale incoming[1] = {stream->scale};

	if (cfg.arm == Arm::kWide) {
		WideRequantOut out;
		if (!RequantChainWideK(wide.data(), hidden_size, std::span<const CarriedScale>{incoming, 1},
		                       site_constant, cfg.k, &out)) {
			return false;
		}
		if (cfg.k == 0) {
			// dual-run: the full production site on the SAME int8-valued inputs.
			std::vector<int8_t> stream8(hidden_size), prod_codes(hidden_size);
			for (size_t i = 0; i < hidden_size; ++i) stream8[i] = static_cast<int8_t>(stream->codes[i]);
			CarriedScale prod_scale{};
			const SslmForwardStatus st =
			    ResidualReconcileSite(branch_code, branch_scale, stream8.data(), stream->scale,
			                          hidden_size, site_constant, prod_codes.data(), &prod_scale);
			sc->reconcile_calls++;
			bool ok = (st == SslmForwardStatus::Ok) && prod_scale.m == out.scale.m &&
			          prod_scale.e == out.scale.e;
			if (ok) {
				for (size_t i = 0; i < hidden_size; ++i) {
					if (static_cast<int64_t>(prod_codes[i]) != out.codes[i]) { ok = false; break; }
				}
			}
			if (!ok) sc->reconcile_mismatch++;
		}
		if (capture_boundary) {
			boundary_out->push_back({site_name, wide, out.r, out.s, cfg.k});
		}
		stream->codes = std::move(out.codes);
		stream->scale = out.scale;
		return true;
	}

	// Arm::kRot -- fine grid, rotate, production int8 quantize, reconstruct.
	WideRequantOut fine;
	if (!RequantChainWideK(wide.data(), hidden_size, std::span<const CarriedScale>{incoming, 1},
	                       site_constant, /*k=*/8, &fine)) {
		return false;
	}
	std::vector<int64_t> rotated(hidden_size);
	RotateForward(rot, fine.codes.data(), rotated.data(), hidden_size);
	// Final int8 quantization of the ROTATED row: this tool's own funnel copy (elementwise
	// dual-run-verified against production at k=0 by the kWide arm). The requant maps
	// c ~= x*127/D', so a site whose only job is re-representation must fold exactly 1/127
	// (the "127 scale wrapper", D-SLM52's convention, which production sites carry inside
	// their offline-derived site constants). 1/127 has no exact CarriedScale form; the
	// 31-bit canonical approximation {round(2^37/127), -37} = {1082196484, -37} is used --
	// the same approximation class every production site constant already carries
	// (relative error ~3e-10, deterministic).
	WideRequantOut rot8;
	const CarriedScale recip127{1082196484, -37};
	const CarriedScale inc2[1] = {fine.scale};
	if (!RequantChainWideK(rotated.data(), hidden_size, std::span<const CarriedScale>{inc2, 1},
	                       recip127, /*k=*/0, &rot8, /*include_site=*/true)) {
		return false;
	}
	(void)ident_site;
	if (capture_boundary) {
		// capture BOTH stages: the fine-grid row (what a non-rotated k=8 site would round)
		// and the rotated row (what this construction actually rounds to int8).
		boundary_out->push_back({std::string(site_name) + ".fine16", wide, fine.r, fine.s, 8});
		boundary_out->push_back({std::string(site_name) + ".rot8", rotated, rot8.r, rot8.s, 0});
	}
	const CarriedScale in_grid = stream->scale;  // the wide row's own grid, for the shadow check
	stream->codes.resize(hidden_size);
	RotateReconstruct(rot, rot8.codes.data(), stream->codes.data(), hidden_size, scratch);
	stream->scale = CarriedScale{rot8.scale.m, rot8.scale.e - kHadamardLog2};
	// One-shot float64 shadow check (measurement-only, not on any arithmetic path): the
	// reconstructed representation must agree with the wide row's own values to within the
	// storage's quantization noise; a scale-bookkeeping error shows up here as a constant
	// factor, which is exactly the defect class this check exists to catch.
	static bool shadow_printed = false;
	if (!shadow_printed) {
		shadow_printed = true;
		double num = 0.0, den = 0.0;
		for (size_t i = 0; i < hidden_size; ++i) {
			const double truth = Dequant(wide[i], in_grid);
			const double rec2 = Dequant(stream->codes[i], stream->scale);
			num += (rec2 - truth) * (rec2 - truth);
			den += truth * truth;
		}
		std::printf("rot shadow check (first site): rel_l2(reconstruction vs wide row) = %.6f\n",
		            std::sqrt(num / std::max(den, 1e-30)));
	}
	return true;
}

}  // namespace

// ---------------------------------------------------------------------------- main

int main(int argc, char** argv) {
	if (argc < 6) {
		std::fprintf(stderr,
		             "usage: %s <model.sslm> <tok.sslm> \"<prompt>\" <pid> --arm k0|k4|k8|rot "
		             "[--dump-dir d] [--seed hex]\n",
		             argv[0]);
		return 2;
	}
	const std::string model_path = argv[1];
	const std::string tokenizer_path = argv[2];
	const std::string prompt = argv[3];
	const std::string prompt_id = argv[4];
	std::string arm_name = "k0";
	std::string dump_dir = "out/t1797arms";
	uint64_t seed = 0x1797C0FFEEull;
	for (int i = 5; i < argc; ++i) {
		if (std::strcmp(argv[i], "--arm") == 0 && i + 1 < argc) arm_name = argv[++i];
		else if (std::strcmp(argv[i], "--dump-dir") == 0 && i + 1 < argc) dump_dir = argv[++i];
		else if (std::strcmp(argv[i], "--seed") == 0 && i + 1 < argc)
			seed = std::strtoull(argv[++i], nullptr, 16);
	}
	ArmConfig cfg;
	cfg.seed = seed;
	if (arm_name == "k0") { cfg.arm = Arm::kWide; cfg.k = 0; cfg.norm_j = 0; }
	else if (arm_name == "k4") { cfg.arm = Arm::kWide; cfg.k = 4; cfg.norm_j = 4; }
	else if (arm_name == "k8") { cfg.arm = Arm::kWide; cfg.k = 8; cfg.norm_j = 8; }
	else if (arm_name == "rot") { cfg.arm = Arm::kRot; cfg.k = 8; cfg.norm_j = 10; }
	else { std::fprintf(stderr, "unknown arm %s\n", arm_name.c_str()); return 2; }

	// --- load model/tokenizer (identical to t1797_multipos_probe.cpp) ---
	std::vector<uint8_t> tok_bytes;
	if (!ReadFile(tokenizer_path.c_str(), tok_bytes)) return 1;
	SslmArtifact tok_artifact;
	SslmError tok_open_err;
	if (SslmArtifact::OpenFromMemory(tok_bytes.data(), tok_bytes.size(), tok_artifact,
	                                 &tok_open_err) != SslmStatus::Ok) return 1;
	TokenizerView tokenizer;
	std::string tok_err;
	if (!TokenizerView::Open(tok_artifact, tokenizer, &tok_err)) return 1;
	const std::vector<int32_t> prompt_tokens = tokenizer.Encode(prompt);
	if (prompt_tokens.empty()) return 1;

	std::vector<uint8_t> model_bytes;
	if (!ReadFile(model_path.c_str(), model_bytes)) return 1;
	SslmModelView model_view;
	std::string model_err;
	if (SslmModel::Load(model_bytes.data(), model_bytes.size(), model_view, &model_err) !=
	    SslmModelStatus::Ok) {
		std::fprintf(stderr, "FAILED model_load: %s\n", model_err.c_str());
		return 1;
	}
	const uint32_t num_heads = model_view.config.num_attention_heads;
	const uint32_t num_kv_heads = model_view.config.num_key_value_heads;
	const uint32_t num_hidden_layers = model_view.config.num_hidden_layers;
	const size_t hidden_size = model_view.config.hidden_size;
	const size_t head_dim = model_view.config.head_dim;
	const size_t intermediate_size = model_view.config.intermediate_size;
	const size_t group = num_heads / num_kv_heads;
	std::printf("model loaded: hidden=%zu layers=%u tokens=%zu arm=%s (%s)\n", hidden_size,
	            num_hidden_layers, prompt_tokens.size(), arm_name.c_str(), prompt_id.c_str());

	PreflightScanWscFolds(model_view);
	std::vector<LayerBacking> backings(num_hidden_layers);
	std::vector<LayerWeights> layers(num_hidden_layers);
	for (uint32_t l = 0; l < num_hidden_layers; ++l) {
		std::string marshal_err;
		if (!MarshalLayer(model_view, l, num_heads, num_kv_heads, backings[l], layers[l],
		                  &marshal_err)) {
			std::fprintf(stderr, "FAILED marshal layer %u: %s\n", l, marshal_err.c_str());
			return 1;
		}
	}
	const SslmTensorView* embed_w = model_view.weights.Tensor("embed");
	if (!embed_w) return 1;
	bool okc = true;
	CarriedScale embed_site_constant = ReadCarriedScale(model_view.composition_constants, "embed", &okc);
	if (!okc) return 1;
	const int8_t* embed_weights = reinterpret_cast<const int8_t*>(embed_w->data);
	const int64_t context_cap = static_cast<int64_t>(model_view.config.context_cap);
	const size_t kv_bytes = static_cast<size_t>(num_hidden_layers) *
	                        static_cast<size_t>(context_cap) * num_kv_heads * head_dim * 2;

	// --- rotation plan + structural self-tests ---
	const RotationPlan rot = MakeRotationPlan(hidden_size, cfg.seed);
	{
		// exact reconstruction: RotateReconstruct(RotateForward(x)) == 512 * x
		uint64_t st = 12345;
		std::vector<int64_t> x(hidden_size), y(hidden_size), xr(hidden_size), scratch;
		for (auto& v : x) v = static_cast<int64_t>(SplitMix64(&st) % 65001) - 32500;
		RotateForward(rot, x.data(), y.data(), hidden_size);
		RotateReconstruct(rot, y.data(), xr.data(), hidden_size, &scratch);
		for (size_t i = 0; i < hidden_size; ++i) {
			if (xr[i] != 512 * x[i]) {
				std::fprintf(stderr, "FAILED rotation self-test at %zu\n", i);
				return 1;
			}
		}
		std::printf("rotation self-test: H(sign,perm) reconstruction exact (512x) on %zu dims\n",
		            hidden_size);
	}
	// The rotated row's second quantization folds NO site constant (the residual site's own
	// constant is folded once, in the fine-grid step) -- CombineCarriedScale's high-mul
	// admits no exact identity (verified by execution: {2^30,-30} loses odd mantissas' low
	// bit), so the fold is skipped rather than fed a pseudo-identity. This placeholder is
	// passed to the skip-site call and never folded.
	const CarriedScale ident_site{int64_t{1} << 30, -30};

	// --- state ---
	WideState stream;
	stream.codes.assign(hidden_size, 0);
	std::vector<uint8_t> workspace(kv_bytes);
	int64_t context_length = 0;
	SelfCheckCounters sc;
	std::vector<float> dump;
	dump.reserve(prompt_tokens.size() * (num_hidden_layers + 1) * hidden_size);
	std::vector<BoundaryDump> boundary;
	std::vector<int64_t> rot_scratch;
	std::vector<int8_t> sitecodes_norm_attn, sitecodes_o, sitecodes_down;
	std::vector<int64_t> sitecodes_probs;  // Q15 probs, layer-major then head then pos
	double transform_seconds = 0.0;
	const auto t_start = std::chrono::steady_clock::now();

	const int arm_extra_bits = (cfg.arm == Arm::kRot) ? 8 : cfg.k;

	for (size_t t = 0; t < prompt_tokens.size(); ++t) {
		const bool last_pos = (t + 1 == prompt_tokens.size());
		// embed -> fine-grid wide codes (exact shift; scale exponent lowered by the arm's k)
		{
			std::vector<int8_t> ecodes(hidden_size);
			CarriedScale escale{};
			const SslmForwardStatus est =
			    EmbedEntry(prompt_tokens[t], static_cast<int32_t>(model_view.config.vocab_size),
			               embed_weights, hidden_size, embed_site_constant, ecodes.data(), &escale);
			if (est != SslmForwardStatus::Ok) return 1;
			for (size_t i = 0; i < hidden_size; ++i)
				stream.codes[i] = static_cast<int64_t>(ecodes[i]) << arm_extra_bits;
			stream.scale = CarriedScale{escale.m, escale.e - arm_extra_bits};
		}
		for (size_t i = 0; i < hidden_size; ++i)
			dump.push_back(static_cast<float>(Dequant(stream.codes[i], stream.scale)));

		for (uint32_t l = 0; l < num_hidden_layers; ++l) {
			const LayerWeights& lw = layers[l];
			const int64_t position = context_length;
			const size_t width = static_cast<size_t>(position) + 1;

			// -- attn norm (wide input) --
			std::vector<int8_t> normed(hidden_size);
			CarriedScale normed_scale{};
			SslmForwardStatus st = WideRmsNorm(stream.codes, lw.attn_norm_gain, hidden_size,
			                                   lw.attn_norm_site_constant, cfg.norm_j,
			                                   normed.data(), &normed_scale);
			if (st != SslmForwardStatus::Ok) { std::fprintf(stderr, "norm fail l=%u\n", l); return 1; }
			if (cfg.k == 0 && cfg.arm == Arm::kWide) {
				std::vector<int8_t> h8(hidden_size), pcodes(hidden_size);
				for (size_t i = 0; i < hidden_size; ++i) h8[i] = static_cast<int8_t>(stream.codes[i]);
				CarriedScale pscale{};
				const SslmForwardStatus pst =
				    RmsNormSite(h8.data(), lw.attn_norm_gain, hidden_size, stream.scale,
				                lw.attn_norm_site_constant, pcodes.data(), &pscale);
				sc.norm_calls++;
				bool ok = pst == SslmForwardStatus::Ok && pscale.m == normed_scale.m &&
				          pscale.e == normed_scale.e &&
				          std::memcmp(pcodes.data(), normed.data(), hidden_size) == 0;
				if (!ok) sc.norm_mismatch++;
			}

			// -- q/k/v projections, K/V landing, RoPE, attention, o_proj: production, copied
			//    verbatim from t1795_residual_probe.cpp's ManualRunOneLayer --
			std::vector<int8_t> q_codes(hidden_size), o_codes(hidden_size);
			std::vector<int8_t> q_rot(hidden_size), k_rot(hidden_size), ctx_codes(hidden_size);
			CarriedScale q_scale{}, ctx_scale{}, o_scale{};
			st = ProjectAndFunnelCopy(normed.data(), normed_scale, lw.q_weight, hidden_size,
			                          hidden_size, lw.q_fold_identity, lw.q_fold_mult,
			                          lw.q_fold_shift, lw.q_site_constant, lw.q_bias,
			                          q_codes.data(), &q_scale);
			if (st != SslmForwardStatus::Ok) return 1;
			{
				const size_t kv_hidden_size = num_kv_heads * head_dim;
				std::vector<int64_t> kacc(kv_hidden_size), vacc(kv_hidden_size);
				GemmInt8AccumulateRow(normed.data(), lw.k_weight, hidden_size, kv_hidden_size, kacc.data());
				GemmInt8AccumulateRow(normed.data(), lw.v_weight, hidden_size, kv_hidden_size, vacc.data());
				for (size_t i = 0; i < kv_hidden_size; ++i) {
					kacc[i] = ApplyWeightScaleFold(kacc[i], lw.k_fold_identity[i], lw.k_fold_mult[i],
					                               lw.k_fold_shift[i]);
					vacc[i] = ApplyWeightScaleFold(vacc[i], lw.v_fold_identity[i], lw.v_fold_mult[i],
					                               lw.v_fold_shift[i]);
				}
				uint64_t sat = 0;
				if (lw.k_bias != nullptr) {
					st = ApplyBiasReconcileRowCopy(kacc.data(), kv_hidden_size, lw.k_bias,
					                               normed_scale.m, normed_scale.e);
					if (st != SslmForwardStatus::Ok) return 1;
				}
				if (lw.v_bias != nullptr) {
					st = ApplyBiasReconcileRowCopy(vacc.data(), kv_hidden_size, lw.v_bias,
					                               normed_scale.m, normed_scale.e);
					if (st != SslmForwardStatus::Ok) return 1;
				}
				for (size_t h = 0; h < num_kv_heads; ++h) {
					int8_t* const k_row = MutableKeyRow(workspace.data(), l, context_cap, num_kv_heads,
					                                    head_dim, h, position);
					int8_t* const v_row = MutableValueRow(workspace.data(), l, context_cap,
					                                      num_kv_heads, head_dim, h, position);
					for (size_t d = 0; d < head_dim; ++d) {
						const size_t i = h * head_dim + d;
						k_row[d] = static_cast<int8_t>(ClampRopeCode(
						    LandingRescale(kacc[i], normed_scale.m, lw.kv_landing_r_t_k[h],
						                   normed_scale.e, lw.kv_landing_e_t_k[h], &sat)));
						v_row[d] = static_cast<int8_t>(ClampRopeCode(
						    LandingRescale(vacc[i], normed_scale.m, lw.kv_landing_r_t_v[h],
						                   normed_scale.e, lw.kv_landing_e_t_v[h], &sat)));
					}
				}
			}
			for (size_t h = 0; h < num_heads; ++h) {
				st = RopeApplySite(q_codes.data() + h * head_dim, head_dim, position, context_cap,
				                   model_view.rope_tables, q_rot.data() + h * head_dim);
				if (st != SslmForwardStatus::Ok) return 1;
				const size_t kv_head = h / group;
				const int8_t* const k_row_before = KeyRow(workspace.data(), l, context_cap,
				                                          num_kv_heads, head_dim, kv_head, position);
				st = RopeApplySite(k_row_before, head_dim, position, context_cap,
				                   model_view.rope_tables, k_rot.data() + h * head_dim);
				if (st != SslmForwardStatus::Ok) return 1;
			}
			for (size_t h = 0; h < num_heads; ++h) {
				const size_t kv_head = h / group;
				int8_t* const k_row = MutableKeyRow(workspace.data(), l, context_cap, num_kv_heads,
				                                    head_dim, kv_head, position);
				for (size_t d = 0; d < head_dim; ++d) k_row[d] = k_rot[h * head_dim + d];
			}
			{
				std::vector<int64_t> ctx_wide(hidden_size);
				std::vector<int64_t> khead_q_ln2(num_kv_heads), khead_q_b(num_kv_heads),
				    khead_q_c(num_kv_heads);
				std::vector<bool> khead_derived(num_kv_heads, false);
				for (size_t h = 0; h < num_heads; ++h) {
					std::vector<int64_t> scores(width), probs(width), ctx_acc(head_dim);
					const size_t kv_head = h / group;
					if (!khead_derived[kv_head]) {
						const CarriedScale sm = CombineCarriedScale(
						    q_scale, CarriedScale{lw.iexp_softmax_khead_m[kv_head],
						                          lw.iexp_softmax_khead_e[kv_head]});
						int64_t dq_ln2 = 0, dq_b = 0, dq_c = 0;
						if (IExpScaleConstants(sm.m, sm.e, kIExpLn2Q, 30, kIExpBQ, 30, kIExpCaQ, 30,
						                       &dq_ln2, &dq_b, &dq_c) != IExpScaleDomain::kOk)
							return 1;
						khead_q_ln2[kv_head] = dq_ln2;
						khead_q_b[kv_head] = dq_b;
						khead_q_c[kv_head] = dq_c;
						khead_derived[kv_head] = true;
					}
					st = CheckSoftmaxRowWidthDomain(khead_q_b[kv_head], khead_q_c[kv_head], width);
					if (st != SslmForwardStatus::Ok) return 1;
					const int8_t* const k_rows_base = KeyRow(workspace.data(), l, context_cap,
					                                         num_kv_heads, head_dim, kv_head, 0);
					GemmInt8AccumulateRow(q_rot.data() + h * head_dim, k_rows_base, head_dim, width,
					                      scores.data());
					if (!SoftmaxRowQ15(scores.data(), width, khead_q_ln2[kv_head],
					                   khead_q_b[kv_head], khead_q_c[kv_head], probs.data()))
						return 1;
					if (last_pos) {
						for (size_t w = 0; w < width; ++w) sitecodes_probs.push_back(probs[w]);
					}
					const int8_t* const v_rows_base = ValueRow(workspace.data(), l, context_cap,
					                                           num_kv_heads, head_dim, kv_head, 0);
					GemmProbQ15Accumulate(probs.data(), v_rows_base, width, head_dim, ctx_acc.data());
					for (size_t d = 0; d < head_dim; ++d) {
						ctx_wide[h * head_dim + d] = ApplyWeightScaleFold(
						    ctx_acc[d], lw.ctx_fold_identity[h], lw.ctx_fold_mult[h],
						    lw.ctx_fold_shift[h]);
					}
				}
				const ChainResult cres =
				    RequantChainChecked(ctx_wide.data(), hidden_size, std::span<const CarriedScale>{},
				                        lw.ctx_fold_site_constant, ctx_codes.data(), &ctx_scale);
				if (cres.status != SslmForwardStatus::Ok) return 1;
			}
			st = ProjectAndFunnelCopy(ctx_codes.data(), ctx_scale, lw.o_weight, hidden_size,
			                          hidden_size, lw.o_fold_identity, lw.o_fold_mult,
			                          lw.o_fold_shift, lw.o_site_constant, nullptr, o_codes.data(),
			                          &o_scale);
			if (st != SslmForwardStatus::Ok) return 1;

			// -- attn residual (THE experimental site) --
			const auto tr0 = std::chrono::steady_clock::now();
			if (!WideResidualSite(cfg, rot, o_codes.data(), o_scale, &stream,
			                      lw.attn_residual_site_constant, ident_site, hidden_size, &sc,
			                      last_pos, (std::string("l") + std::to_string(l) + ".attn_residual").c_str(),
			                      &boundary, &rot_scratch)) {
				std::fprintf(stderr, "attn_residual domain fail l=%u t=%zu\n", l, t);
				return 1;
			}
			transform_seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - tr0).count();

			// -- mlp norm (wide input) --
			std::vector<int8_t> normed2(hidden_size);
			CarriedScale normed2_scale{};
			st = WideRmsNorm(stream.codes, lw.mlp_norm_gain, hidden_size, lw.mlp_norm_site_constant,
			                 cfg.norm_j, normed2.data(), &normed2_scale);
			if (st != SslmForwardStatus::Ok) return 1;
			if (cfg.k == 0 && cfg.arm == Arm::kWide) {
				std::vector<int8_t> h8(hidden_size), pcodes(hidden_size);
				for (size_t i = 0; i < hidden_size; ++i) h8[i] = static_cast<int8_t>(stream.codes[i]);
				CarriedScale pscale{};
				const SslmForwardStatus pst =
				    RmsNormSite(h8.data(), lw.mlp_norm_gain, hidden_size, stream.scale,
				                lw.mlp_norm_site_constant, pcodes.data(), &pscale);
				sc.norm_calls++;
				bool ok = pst == SslmForwardStatus::Ok && pscale.m == normed2_scale.m &&
				          pscale.e == normed2_scale.e &&
				          std::memcmp(pcodes.data(), normed2.data(), hidden_size) == 0;
				if (!ok) sc.norm_mismatch++;
			}

			// -- MLP branch: production, verbatim --
			std::vector<int8_t> gate_codes(intermediate_size), up_codes(intermediate_size);
			std::vector<int8_t> act_codes(intermediate_size), down_codes(hidden_size);
			CarriedScale gate_scale{}, up_scale{}, act_scale{}, down_scale{};
			st = ProjectAndFunnelCopy(normed2.data(), normed2_scale, lw.gate_weight, hidden_size,
			                          intermediate_size, lw.gate_fold_identity, lw.gate_fold_mult,
			                          lw.gate_fold_shift, lw.gate_site_constant, nullptr,
			                          gate_codes.data(), &gate_scale);
			if (st != SslmForwardStatus::Ok) return 1;
			st = ProjectAndFunnelCopy(normed2.data(), normed2_scale, lw.up_weight, hidden_size,
			                          intermediate_size, lw.up_fold_identity, lw.up_fold_mult,
			                          lw.up_fold_shift, lw.up_site_constant, nullptr,
			                          up_codes.data(), &up_scale);
			if (st != SslmForwardStatus::Ok) return 1;
			st = MlpActSite(gate_codes.data(), gate_scale, up_codes.data(), up_scale,
			                intermediate_size, kSiluLutCanonicalTable, lw.mlp_act_site_constant,
			                act_codes.data(), &act_scale);
			if (st != SslmForwardStatus::Ok) return 1;
			st = ProjectAndFunnelCopy(act_codes.data(), act_scale, lw.down_weight, intermediate_size,
			                          hidden_size, lw.down_fold_identity, lw.down_fold_mult,
			                          lw.down_fold_shift, lw.down_site_constant, nullptr,
			                          down_codes.data(), &down_scale);
			if (st != SslmForwardStatus::Ok) return 1;

			// -- mlp residual (THE experimental site) --
			const auto tr1 = std::chrono::steady_clock::now();
			if (!WideResidualSite(cfg, rot, down_codes.data(), down_scale, &stream,
			                      lw.mlp_residual_site_constant, ident_site, hidden_size, &sc,
			                      last_pos, (std::string("l") + std::to_string(l) + ".mlp_residual").c_str(),
			                      &boundary, &rot_scratch)) {
				std::fprintf(stderr, "mlp_residual domain fail l=%u t=%zu\n", l, t);
				return 1;
			}
			transform_seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - tr1).count();

			if (last_pos) {
				sitecodes_norm_attn.insert(sitecodes_norm_attn.end(), normed.begin(), normed.end());
				sitecodes_o.insert(sitecodes_o.end(), o_codes.begin(), o_codes.end());
				sitecodes_down.insert(sitecodes_down.end(), down_codes.begin(), down_codes.end());
			}
			for (size_t i = 0; i < hidden_size; ++i)
				dump.push_back(static_cast<float>(Dequant(stream.codes[i], stream.scale)));
		}
		context_length++;
	}

	const double total_seconds =
	    std::chrono::duration<double>(std::chrono::steady_clock::now() - t_start).count();

	std::printf("self_check: norm dual-runs %llu mismatches %llu; reconcile dual-runs %llu "
	            "mismatches %llu\n",
	            static_cast<unsigned long long>(sc.norm_calls),
	            static_cast<unsigned long long>(sc.norm_mismatch),
	            static_cast<unsigned long long>(sc.reconcile_calls),
	            static_cast<unsigned long long>(sc.reconcile_mismatch));
	if (sc.norm_mismatch != 0 || sc.reconcile_mismatch != 0) {
		std::fprintf(stderr, "FAILED: k0 dual-run mismatch\n");
		return 1;
	}
	std::printf("timing: total %.2fs, residual-site (incl. transform) %.3fs (%.2f%%)\n",
	            total_seconds, transform_seconds, 100.0 * transform_seconds / total_seconds);

	// --- dumps ---
	const size_t n_states = static_cast<size_t>(num_hidden_layers) + 1;
	{
		const std::string bin = dump_dir + "/" + prompt_id + "_" + arm_name + ".bin";
		std::ofstream f(bin, std::ios::binary);
		if (!f) { std::fprintf(stderr, "dump open fail\n"); return 1; }
		f.write(reinterpret_cast<const char*>(dump.data()),
		        static_cast<std::streamsize>(dump.size() * sizeof(float)));
		std::ofstream m(dump_dir + "/" + prompt_id + "_" + arm_name + ".meta");
		m << "positions " << prompt_tokens.size() << "\nstates " << n_states << "\nhidden "
		  << hidden_size << "\ndtype float32-le\nlayout pos,state,hidden\n";
	}
	{
		std::ofstream f(dump_dir + "/" + prompt_id + "_" + arm_name + "_boundary.txt");
		for (const auto& b : boundary) {
			f << b.site << " r " << b.r << " s " << b.s << " k " << b.k << " n " << b.row.size();
			for (int64_t v : b.row) f << " " << v;
			f << "\n";
		}
	}
	{
		std::ofstream f(dump_dir + "/" + prompt_id + "_" + arm_name + "_sitecodes.txt");
		auto put8 = [&](const char* name, const std::vector<int8_t>& v) {
			f << name << " " << v.size();
			for (int8_t c : v) f << " " << static_cast<int>(c);
			f << "\n";
		};
		put8("norm_attn", sitecodes_norm_attn);
		put8("o_proj", sitecodes_o);
		put8("down_proj", sitecodes_down);
		f << "probs_q15 " << sitecodes_probs.size();
		for (int64_t p : sitecodes_probs) f << " " << p;
		f << "\n";
	}
	std::printf("dumps written: %s/%s_%s.*\n", dump_dir.c_str(), prompt_id.c_str(),
	            arm_name.c_str());
	return 0;
}
