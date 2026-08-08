// T-1855 Loki strike probe -- M3 (grouped-static landing, design sections 6.9 / 8.2a).
//
// Disposable adversarial probe. Not product source, not a test-suite cell.
//
// What it constructs: one (layer, kv_head) K row at head_dim = 128, landed twice
//   arm BASE -- today's construction: ONE offline (r_t, e_t) pair for the whole head
//   arm M3   -- the design's construction: one offline (r_t, e_t) pair per (head, channel-group),
//               G_L static groups over the 128 channels, calibrated by the design's own
//               stated method ("narrower population, same method", section 8.2a).
//
// Both arms land through the engine's own LandingRescale + ClampRopeCode, and both are
// consumed through the engine's own GemmInt8AccumulateRow over head_dim -- the score GEMM
// at forward_sites.cpp:1397, q_rot . k_rows_base, contracting over head_dim.
//
// The consumer applies ONE scale per kv_head (iexp_softmax_khead_m/e[kv_head], combined with
// q_scale at forward_sites.cpp:1354), because M3's claim is that the K/V store's format is
// untouched, no sidecar is written, and no consumer changes ("0%", section 8.2a's options table).
//
// Position 0 is used, where RoPE is the identity (cos = 1, sin = 0), so site 7 is transparent
// and the strike lands on the arithmetic under study rather than on the rotation.
//
// PREDICTIONS, stated before the run:
//   promise  -- M3 is a strictly finer landing grid at zero consumer cost; every property
//               section 3 requires still holds; M3's dequantized score error <= BASE's.
//   fracture -- the score sum spans all G_L groups on the contraction axis, and one per-head
//               constant cannot un-scale G_L grids; M3's score error is O(1) and GROWS with
//               the refinement M3 buys.

#include <cstdint>
#include <cstdio>
#include <cmath>
#include <vector>
#include <random>
#include <algorithm>

#include "superslm/forward_sites.h"
#include "superslm/matmul.h"

using namespace superslm;

namespace {

constexpr size_t kHeadDim = 128;

// Offline landing pair -> the value-per-code step it represents.
// LandingRescale computes round(branch_code * m_a * r_t / 2^(62 - (e_a - e_t))).
// With v = branch_code * m_a * 2^e_a (the value the wide accumulator stands for),
// the emitted code is v * r_t * 2^(-62 - e_t), so step = 2^(62 + e_t) / r_t.
struct LandingPair {
	int64_t r_t;
	int64_t e_t;
};

double StepOf(LandingPair p) { return std::ldexp(1.0, static_cast<int>(62 + p.e_t)) / static_cast<double>(p.r_t); }

// Calibrate an offline pair for a target step, honouring the artifact's own domain for
// KvLandingReciprocals: r_t in [2^31 + 1, 2^32] (forward_sites.cpp's own U128 comment),
// e_t >= kKvLandingExponentMin = -60 (model.cpp).
LandingPair Calibrate(double target_step) {
	// choose e_t so that r_t = 2^(62+e_t)/step lands near 2^31.5
	const double want_log_r = 31.5;
	int e_t = static_cast<int>(std::llround(std::log2(target_step) + want_log_r - 62.0));
	LandingPair p{0, e_t};
	double r = std::ldexp(1.0, 62 + e_t) / target_step;
	while (r <= 2147483648.0 && e_t < 60) { ++e_t; r = std::ldexp(1.0, 62 + e_t) / target_step; }
	while (r > 4294967296.0 && e_t > -60) { --e_t; r = std::ldexp(1.0, 62 + e_t) / target_step; }
	p.e_t = e_t;
	p.r_t = static_cast<int64_t>(std::llround(r));
	if (p.r_t < 2147483649LL) p.r_t = 2147483649LL;
	if (p.r_t > 4294967296LL) p.r_t = 4294967296LL;
	return p;
}

// Land one channel through the ENGINE's own composite.
int8_t Land(int64_t kacc, int64_t m_a, int64_t e_a, LandingPair p, uint64_t* sat) {
	return static_cast<int8_t>(ClampRopeCode(LandingRescale(kacc, m_a, p.r_t, e_a, p.e_t, sat)));
}

struct ArmResult {
	double score_value;   // dequantized score as the CONSUMER computes it
	double rel_err;       // against the exact real-valued reference
	int64_t score_raw;
	uint64_t saturations;
};

}  // namespace

int main() {
	std::mt19937_64 rng(20260808u);

	// --- the norm output's carried scale (canonical mantissa, section 4 / checked_chain_funnel.h)
	const int64_t m_a = 1181116006LL;   // in [2^30, 2^31)
	const int64_t e_a = -34;

	printf("T-1855 M3 strike probe -- engine composite, head_dim=%zu, position 0 (RoPE identity)\n", kHeadDim);
	printf("m_a=%lld e_a=%lld\n\n", (long long)m_a, (long long)e_a);

	// --- the K accumulator row: section 2.3's channel-magnitude spread geometry, which is the
	//     ONLY geometry M3 exists to attack (section 6.9: "M3 attacks the SAME channel-magnitude-
	//     spread geometry (2.3)"). A minority of channels carry far larger magnitude.
	//     Bound: the projection accumulator's own ~2^27 (forward_sites.cpp's U128 comment).
	std::vector<int64_t> kacc(kHeadDim);
	std::vector<int8_t> q(kHeadDim);
	{
		std::normal_distribution<double> nd(0.0, 1.0);
		std::uniform_int_distribution<int> qd(-127, 127);
		for (size_t d = 0; d < kHeadDim; ++d) {
			// channel scale varies by group of 16: a "hot" band and quiet bands, the wandering-
			// channel population section 2.9 measures.
			const double band = (d / 16 == 3) ? 1.0 : ((d / 16 == 6) ? 0.35 : 0.02);
			kacc[d] = static_cast<int64_t>(std::llround(nd(rng) * band * std::ldexp(1.0, 26)));
			q[d] = static_cast<int8_t>(qd(rng));
		}
	}

	// exact real-valued reference: v_d = kacc_d * m_a * 2^e_a ; score_ref = sum_d q_d * v_d
	std::vector<double> v(kHeadDim);
	double score_ref = 0.0;
	for (size_t d = 0; d < kHeadDim; ++d) {
		v[d] = static_cast<double>(kacc[d]) * static_cast<double>(m_a) * std::ldexp(1.0, static_cast<int>(e_a));
		score_ref += static_cast<double>(q[d]) * v[d];
	}

	// --- BASE arm: one pair for the head, calibrated so the head max lands near 127.
	double head_max = 0.0;
	for (double x : v) head_max = std::max(head_max, std::fabs(x));
	const LandingPair head_pair = Calibrate(head_max / 127.0);
	const double s_head = StepOf(head_pair);

	auto run_arm = [&](const std::vector<LandingPair>& per_channel, const char* name) -> ArmResult {
		std::vector<int8_t> k_row(kHeadDim);
		uint64_t sat = 0;
		for (size_t d = 0; d < kHeadDim; ++d) k_row[d] = Land(kacc[d], m_a, e_a, per_channel[d], &sat);
		int64_t score_raw = 0;
		// the engine's own score GEMM, contracting over head_dim
		GemmInt8AccumulateRow(q.data(), k_row.data(), kHeadDim, 1, &score_raw);
		// the consumer's single per-kv_head un-scaling: M3 changes NOTHING here by its own claim
		const double score_value = static_cast<double>(score_raw) * s_head;
		const double rel = std::fabs(score_value - score_ref) / std::fabs(score_ref);
		printf("  %-28s raw=%-14lld value=%- 18.6g rel_err=%.6g  sat=%llu\n",
		       name, (long long)score_raw, score_value, rel, (unsigned long long)sat);
		return ArmResult{score_value, rel, score_raw, sat};
	};

	printf("exact reference score = %.10g\n", score_ref);
	printf("head max |v| = %.6g   BASE step = %.6g  (r_t=%lld e_t=%lld)\n\n",
	       head_max, s_head, (long long)head_pair.r_t, (long long)head_pair.e_t);

	std::vector<LandingPair> base_pairs(kHeadDim, head_pair);
	printf("arm BASE  (today: one offline pair per head)\n");
	const ArmResult base = run_arm(base_pairs, "per-head landing");

	printf("\narm M3    (design 6.9/8.2a: one offline pair per (head, channel-group))\n");
	printf("  %-6s %-12s %-14s %-12s %s\n", "G_L", "score_raw", "score_value", "rel_err", "max group refinement");
	for (size_t G_L : {2u, 4u, 8u, 16u, 32u, 64u, 128u}) {
		const size_t gw = kHeadDim / G_L;   // channels per group
		std::vector<LandingPair> pairs(kHeadDim);
		double worst_ratio = 1.0;
		for (size_t g = 0; g < G_L; ++g) {
			double gmax = 0.0;
			for (size_t d = g * gw; d < (g + 1) * gw; ++d) gmax = std::max(gmax, std::fabs(v[d]));
			if (gmax <= 0.0) gmax = head_max;   // degenerate group keeps the head grid
			// "narrower population, same method": calibrate this group's own pair so its own
			// max lands near 127. This is the refinement M3 is FOR.
			const LandingPair gp = Calibrate(gmax / 127.0);
			for (size_t d = g * gw; d < (g + 1) * gw; ++d) pairs[d] = gp;
			worst_ratio = std::max(worst_ratio, s_head / StepOf(gp));
		}
		std::vector<int8_t> k_row(kHeadDim);
		uint64_t sat = 0;
		for (size_t d = 0; d < kHeadDim; ++d) k_row[d] = Land(kacc[d], m_a, e_a, pairs[d], &sat);
		int64_t score_raw = 0;
		GemmInt8AccumulateRow(q.data(), k_row.data(), kHeadDim, 1, &score_raw);
		const double score_value = static_cast<double>(score_raw) * s_head;
		const double rel = std::fabs(score_value - score_ref) / std::fabs(score_ref);
		// CONTROL: what the score would be if the consumer DID know each group's own step --
		// i.e. if a sidecar (or a per-group un-scaling in the score GEMM) existed. M3 writes
		// no sidecar and changes no consumer, so this arm is NOT what the design specifies;
		// it exists only to prove the fracture is the missing consumer substitution and not
		// the calibration.
		double ctrl = 0.0;
		for (size_t d = 0; d < kHeadDim; ++d) ctrl += static_cast<double>(q[d]) * static_cast<double>(k_row[d]) * StepOf(pairs[d]);
		const double ctrl_rel = std::fabs(ctrl - score_ref) / std::fabs(score_ref);
		printf("  %-6zu %-12lld %- 14.6g %-12.6g %-10.1fx  [control, per-group unscale: rel_err=%.6g]\n",
		       G_L, (long long)score_raw, score_value, rel, worst_ratio, ctrl_rel);
	}

	// --- the attention-level consequence: a whole score row over `width` positions.
	printf("\nattention row (width=16 positions, same construction per position), G_L=8\n");
	{
		const size_t width = 16;
		const size_t G_L = 8, gw = kHeadDim / G_L;
		std::vector<double> ref_scores(width), base_scores(width), m3_scores(width);
		std::normal_distribution<double> nd(0.0, 1.0);
		for (size_t t = 0; t < width; ++t) {
			std::vector<int64_t> ka(kHeadDim);
			std::vector<double> vv(kHeadDim);
			for (size_t d = 0; d < kHeadDim; ++d) {
				const double band = (d / 16 == 3) ? 1.0 : ((d / 16 == 6) ? 0.35 : 0.02);
				ka[d] = static_cast<int64_t>(std::llround(nd(rng) * band * std::ldexp(1.0, 26)));
				vv[d] = static_cast<double>(ka[d]) * static_cast<double>(m_a) * std::ldexp(1.0, static_cast<int>(e_a));
			}
			double r = 0.0;
			for (size_t d = 0; d < kHeadDim; ++d) r += static_cast<double>(q[d]) * vv[d];
			ref_scores[t] = r;

			std::vector<int8_t> kb(kHeadDim), km(kHeadDim);
			uint64_t s1 = 0, s2 = 0;
			for (size_t d = 0; d < kHeadDim; ++d) kb[d] = Land(ka[d], m_a, e_a, head_pair, &s1);
			std::vector<LandingPair> pairs(kHeadDim);
			for (size_t g = 0; g < G_L; ++g) {
				double gmax = 0.0;
				for (size_t d = g * gw; d < (g + 1) * gw; ++d) gmax = std::max(gmax, std::fabs(vv[d]));
				if (gmax <= 0.0) gmax = head_max;
				const LandingPair gp = Calibrate(gmax / 127.0);
				for (size_t d = g * gw; d < (g + 1) * gw; ++d) pairs[d] = gp;
			}
			for (size_t d = 0; d < kHeadDim; ++d) km[d] = Land(ka[d], m_a, e_a, pairs[d], &s2);
			int64_t rb = 0, rm = 0;
			GemmInt8AccumulateRow(q.data(), kb.data(), kHeadDim, 1, &rb);
			GemmInt8AccumulateRow(q.data(), km.data(), kHeadDim, 1, &rm);
			base_scores[t] = static_cast<double>(rb) * s_head;
			m3_scores[t] = static_cast<double>(rm) * s_head;
		}
		auto softmax = [&](const std::vector<double>& s) {
			std::vector<double> p(s.size());
			double mx = *std::max_element(s.begin(), s.end()), z = 0.0;
			for (size_t i = 0; i < s.size(); ++i) { p[i] = std::exp((s[i] - mx) * 1e-6); z += p[i]; }
			for (double& x : p) x /= z;
			return p;
		};
		const auto pr = softmax(ref_scores), pb = softmax(base_scores), pm = softmax(m3_scores);
		double l1b = 0.0, l1m = 0.0;
		size_t amax_r = std::max_element(pr.begin(), pr.end()) - pr.begin();
		size_t amax_b = std::max_element(pb.begin(), pb.end()) - pb.begin();
		size_t amax_m = std::max_element(pm.begin(), pm.end()) - pm.begin();
		for (size_t i = 0; i < width; ++i) { l1b += std::fabs(pb[i] - pr[i]); l1m += std::fabs(pm[i] - pr[i]); }
		printf("  attention L1(prob, reference): BASE=%.6f   M3=%.6f\n", l1b, l1m);
		printf("  argmax position: reference=%zu  BASE=%zu  M3=%zu\n", amax_r, amax_b, amax_m);
	}

	printf("\nBASE rel_err = %.6g\n", base.rel_err);
	return 0;
}
