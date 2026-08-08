// T-1857 -- adversary strike on M3' (design sections 6.9, 8.2a, 28).
//
// PROMISE UNDER ATTACK (verbatim, section 28.6 / D-SLM2071):
//   "M3-PRIME's mean error never exceeded BASE's at any G_L under this adversarial population"
//   "The adversarial population was built specifically to break M3' and did not -- this is the
//    fold's own executed strike, and it is the strike that must fail for the fold to exit."
//   Section 8.2a's options table, M3' row: "0% per-cached-row storage ... partial, delivered:
//   population-mean relative error 0.24x-0.71x of today's per-head construction at
//   G_L in {2,...,64}, executed (D-SLM2071)."
//
// THE PLANE. Section 6.9 defines M3' by the property that distinguishes it from option E:
//   "a small build-time-constant array k_g[h][g], shipped with the model artifact ... NOT a
//    per-position or per-cached-row quantity, so it does not enter the K/V store's per-row
//    format".
// The fold's own probe (Claude/Vitruvius/probes/t1856-fold17-m3prime-probe.cpp, RunM3Prime,
// lines 184-196) computes k_g[g] from `gmax`, the max over the group OF THE ROW BEING LANDED.
// That is a per-position quantity. Every number in D-SLM2071 was produced by a construction
// whose refinement exponent is chosen with knowledge of the row -- i.e. by option E, the
// construction section 8.2a rules out because a per-position k has nowhere to live.
//
// Neither of the fold's two populations can see the difference: in DrawRow the loud bands are
// always d/16 == 3 and 6, and in DrawRowStress band magnitude is 10^-(d/16). Both hold channel
// magnitude STATIONARY IN CHANNEL INDEX. Under a stationary population a frozen k_g and a
// per-row k_g coincide, so the oracle is invisible. Section 2.9 of the same design states the
// ground truth: "the largest-magnitude channel ... changes identity prompt-to-prompt (6-8
// distinct top channels across 9 prompts at 23/28 layers, rank0/median 22.7x); any
// fixed-channel-index construction is disqualified."
//
// THE INSTRUMENT. One variable is changed from the fold's own probe: which band is loud. The
// marginal distribution, the dynamic range (1.0 / 0.35 / 0.02, a 50x spread -- MILDER than the
// fold's own six-order-of-magnitude stress draw), head_dim, N, the engine composite, m_a, e_a
// and the legal G_L set are all identical. Only channel identity wanders, per section 2.9.
//
// Arms (all land through the engine's own LandingRescale + ClampRopeCode):
//   BASE_oracle  -- the fold's BASE: (r_t,e_t) calibrated from the row's own head max.
//   BASE_static  -- BASE as the engine actually is (section 8.2a: "no runtime reciprocal exists
//                   at this site"): one offline pair calibrated once from a calibration
//                   population, frozen across evaluation.
//   M3P_oracle   -- the fold's M3': k_g recomputed per row. Reproduced as a same-run control to
//                   confirm this probe reproduces D-SLM2071's numbers on the fold's own
//                   population.
//   M3P_static   -- M3' AS SPECIFIED: (r_t,e_t)[g] and k_g[g] calibrated once from the
//                   calibration population and FROZEN, K_h = max_g k_g constant; consumed by
//                   the group-aware exact-shift accumulate of section 6.9.
//
// Two calibration policies for the static arms, because a real offline procedure must choose one:
//   MAX  -- group worst case over the calibration population (safe: no clamping by construction).
//   P999 -- the 99.9th percentile of |v| in the group over the calibration population (the
//           benefit-seeking choice; anything tighter than MAX is some percentile).
//
// FALSIFIABLE PREDICTION, stated before the run.
//   Promise predicts: M3P_static mean relative error <= BASE_static's at every legal G_L on the
//   wandering population, in the 0.24x-1.00x band section 28.6 reports.
//   Fracture predicts: on the wandering population M3P_static is NOT below BASE_static at any
//   G_L -- under MAX calibration it collapses to ~1.00x (k_g driven to 0: the refinement M3'
//   sells is not delivered at all), and under P999 calibration it is materially WORSE than
//   BASE_static (frozen fine group steps clamp a channel that wandered loud). One of these dies.
//   The stationary controls must show M3P_static ~= M3P_oracle, isolating channel identity as
//   the whole of the effect.

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
constexpr size_t kPopulation = 300;
constexpr size_t kCalibPopulation = 300;
constexpr size_t kNumBands = 8;
constexpr size_t kBandWidth = kHeadDim / kNumBands;  // 16, identical to the fold's probe

// ---- calibration pair machinery, byte-for-byte the fold's own (t1856 probe lines 63-86) ----

struct LandingPair {
	int64_t r_t;
	int64_t e_t;
};

double StepOf(LandingPair p) { return std::ldexp(1.0, static_cast<int>(62 + p.e_t)) / static_cast<double>(p.r_t); }

LandingPair Calibrate(double target_step) {
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

int8_t Land(int64_t kacc, int64_t m_a, int64_t e_a, LandingPair p, uint64_t* sat) {
	return static_cast<int8_t>(ClampRopeCode(LandingRescale(kacc, m_a, p.r_t, e_a, p.e_t, sat)));
}

struct Row {
	std::vector<int64_t> kacc;
	std::vector<int8_t> q;
	std::vector<double> v;
	double score_ref;
};

// ---- populations ----
//
// Stationary: the fold's own DrawRow, verbatim -- loud bands are always 3 and 6.
// Wandering:  identical in every respect except that the two loud band indices are drawn per
//             row, which is section 2.9's measured ground truth.

Row DrawRowBanded(std::mt19937_64& rng, int64_t m_a, int64_t e_a, bool wander) {
	Row row;
	row.kacc.resize(kHeadDim);
	row.q.resize(kHeadDim);
	row.v.resize(kHeadDim);
	std::normal_distribution<double> nd(0.0, 1.0);
	std::uniform_int_distribution<int> qd(-127, 127);

	size_t loud = 3, mid = 6;  // the fold's own fixed choice
	if (wander) {
		std::uniform_int_distribution<size_t> bd(0, kNumBands - 1);
		loud = bd(rng);
		do { mid = bd(rng); } while (mid == loud);
	}
	for (size_t d = 0; d < kHeadDim; ++d) {
		const size_t b = d / kBandWidth;
		const double band = (b == loud) ? 1.0 : ((b == mid) ? 0.35 : 0.02);
		row.kacc[d] = static_cast<int64_t>(std::llround(nd(rng) * band * std::ldexp(1.0, 26)));
		row.q[d] = static_cast<int8_t>(qd(rng));
	}
	row.score_ref = 0.0;
	for (size_t d = 0; d < kHeadDim; ++d) {
		row.v[d] = static_cast<double>(row.kacc[d]) * static_cast<double>(m_a) * std::ldexp(1.0, static_cast<int>(e_a));
		row.score_ref += static_cast<double>(row.q[d]) * row.v[d];
	}
	return row;
}

double RelErr(double got, double ref) { return std::fabs(got - ref) / std::fabs(ref); }

// ---- the offline calibration M3' actually specifies: computed once, frozen ----

struct StaticTable {
	LandingPair head_pair;          // BASE_static, and M3''s reference step s_head
	double s_head = 0.0;
	std::vector<int> k_g;           // build-time constant array k_g[g]
	std::vector<LandingPair> pairs; // per channel, from the frozen per-group step
	int K_h = 0;
};

enum class Policy { Max, P999 };

// Percentile of the absolute values seen in a group over the whole calibration population.
double GroupStat(const std::vector<double>& absvals, Policy pol) {
	if (absvals.empty()) return 0.0;
	std::vector<double> s = absvals;
	std::sort(s.begin(), s.end());
	if (pol == Policy::Max) return s.back();
	const size_t idx = static_cast<size_t>(std::llround(0.999 * static_cast<double>(s.size() - 1)));
	return s[idx];
}

StaticTable BuildStaticTable(const std::vector<Row>& calib, size_t G_L, Policy pol) {
	const size_t gw = kHeadDim / G_L;
	StaticTable t;

	// Head reference step: the same offline per-head calibration BASE gets, over the same
	// calibration population, under the same policy.
	std::vector<double> head_abs;
	head_abs.reserve(calib.size() * kHeadDim);
	for (const Row& r : calib) for (double x : r.v) head_abs.push_back(std::fabs(x));
	const double head_stat = GroupStat(head_abs, pol);
	t.head_pair = Calibrate(head_stat / 127.0);
	t.s_head = StepOf(t.head_pair);

	// Section 6.9: "the largest non-negative integer k_g such that the group's own worst-case
	// code does not exceed 127 at step s_head * 2^-k_g" -- against CALIBRATION statistics.
	t.k_g.assign(G_L, 0);
	t.pairs.assign(kHeadDim, t.head_pair);
	for (size_t g = 0; g < G_L; ++g) {
		std::vector<double> ga;
		ga.reserve(calib.size() * gw);
		for (const Row& r : calib)
			for (size_t d = g * gw; d < (g + 1) * gw; ++d) ga.push_back(std::fabs(r.v[d]));
		double gstat = GroupStat(ga, pol);
		if (gstat <= 0.0) gstat = t.s_head * 127.0;
		int k = static_cast<int>(std::floor(std::log2((t.s_head * 127.0) / gstat) + 1e-9));
		if (k < 0) k = 0;
		t.k_g[g] = k;
		t.K_h = std::max(t.K_h, k);
		const LandingPair snapped = Calibrate(t.s_head / std::ldexp(1.0, k));
		for (size_t d = g * gw; d < (g + 1) * gw; ++d) t.pairs[d] = snapped;
	}
	return t;
}

// ---- arms ----

double RunBaseWithPair(const Row& row, int64_t m_a, int64_t e_a, LandingPair p, double step,
                        uint64_t* sat_out = nullptr, size_t* clamp_out = nullptr) {
	std::vector<int8_t> k_row(kHeadDim);
	uint64_t sat = 0;
	size_t clamps = 0;
	for (size_t d = 0; d < kHeadDim; ++d) {
		k_row[d] = Land(row.kacc[d], m_a, e_a, p, &sat);
		if (k_row[d] == 127 || k_row[d] == -127 || k_row[d] == -128) ++clamps;
	}
	if (sat_out) *sat_out += sat;
	if (clamp_out) *clamp_out += clamps;
	int64_t score_raw = 0;
	GemmInt8AccumulateRow(row.q.data(), k_row.data(), kHeadDim, 1, &score_raw);
	return RelErr(static_cast<double>(score_raw) * step, row.score_ref);
}

LandingPair OracleHeadPair(const Row& row) {
	double head_max = 0.0;
	for (double x : row.v) head_max = std::max(head_max, std::fabs(x));
	return Calibrate(head_max / 127.0);
}

// M3' consumed by the group-aware exact-shift accumulate (section 6.9), with a table supplied
// from outside -- frozen (M3' as specified) or per-row (the fold's probe).
double RunM3PrimeWithTable(const Row& row, int64_t m_a, int64_t e_a, const StaticTable& t,
                            size_t G_L, uint64_t* sat_out = nullptr, size_t* clamp_out = nullptr) {
	const size_t gw = kHeadDim / G_L;
	std::vector<int8_t> k_row(kHeadDim);
	uint64_t sat = 0;
	size_t clamps = 0;
	for (size_t d = 0; d < kHeadDim; ++d) {
		k_row[d] = Land(row.kacc[d], m_a, e_a, t.pairs[d], &sat);
		if (k_row[d] == 127 || k_row[d] == -127 || k_row[d] == -128) ++clamps;
	}
	if (sat_out) *sat_out += sat;
	if (clamp_out) *clamp_out += clamps;
	int64_t score_raw = 0;
	for (size_t g = 0; g < G_L; ++g) {
		const int shift = t.K_h - t.k_g[g];
		for (size_t d = g * gw; d < (g + 1) * gw; ++d)
			score_raw += static_cast<int64_t>(row.q[d]) * (static_cast<int64_t>(k_row[d]) << shift);
	}
	return RelErr(static_cast<double>(score_raw) * t.s_head * std::ldexp(1.0, -t.K_h), row.score_ref);
}

// The fold's own M3': k_g derived from the row being landed (t1856 probe lines 184-196).
StaticTable OracleTable(const Row& row, size_t G_L) {
	const size_t gw = kHeadDim / G_L;
	StaticTable t;
	t.head_pair = OracleHeadPair(row);
	t.s_head = StepOf(t.head_pair);
	t.k_g.assign(G_L, 0);
	t.pairs.assign(kHeadDim, t.head_pair);
	for (size_t g = 0; g < G_L; ++g) {
		double gmax = 0.0;
		for (size_t d = g * gw; d < (g + 1) * gw; ++d) gmax = std::max(gmax, std::fabs(row.v[d]));
		if (gmax <= 0.0) gmax = t.s_head * 127.0;
		const LandingPair desired = Calibrate(gmax / 127.0);
		int k = static_cast<int>(std::floor(std::log2(t.s_head / StepOf(desired)) + 1e-9));
		if (k < 0) k = 0;
		t.k_g[g] = k;
		t.K_h = std::max(t.K_h, k);
		const LandingPair snapped = Calibrate(t.s_head / std::ldexp(1.0, k));
		for (size_t d = g * gw; d < (g + 1) * gw; ++d) t.pairs[d] = snapped;
	}
	return t;
}

struct Stats { double mean = 0.0, median = 0.0, max = 0.0; };

Stats Summarize(std::vector<double> v) {
	Stats s;
	double sum = 0.0;
	for (double x : v) { sum += x; s.max = std::max(s.max, x); }
	s.mean = sum / v.size();
	std::sort(v.begin(), v.end());
	s.median = v[v.size() / 2];
	return s;
}

const char* PolName(Policy p) { return p == Policy::Max ? "MAX " : "P999"; }

void RunBlock(const char* label, bool wander, Policy pol, int64_t m_a, int64_t e_a) {
	printf("\n=== %s | channel identity: %s | offline calibration policy: %s ===\n",
	       label, wander ? "WANDERING (section 2.9)" : "stationary (the fold's own)", PolName(pol));
	printf("%-5s %-12s %-12s %-9s %-12s %-9s %-6s %-8s %-10s\n",
	       "G_L", "BASE_static", "M3P_static", "(vs BASE)", "M3P_oracle", "(vs BASE)",
	       "K_h", "k_g[0..]", "M3P clamps");
	for (size_t G_L : {2u, 4u, 8u, 16u, 32u, 64u}) {
		// Calibration population -- drawn from the same distribution, disjoint stream.
		std::mt19937_64 crng(555000u + static_cast<uint64_t>(G_L) * 131u + (wander ? 1u : 0u) + (pol == Policy::P999 ? 77u : 0u));
		std::vector<Row> calib;
		calib.reserve(kCalibPopulation);
		for (size_t i = 0; i < kCalibPopulation; ++i) calib.push_back(DrawRowBanded(crng, m_a, e_a, wander));
		const StaticTable table = BuildStaticTable(calib, G_L, pol);

		// Evaluation population -- the fold's own seed scheme.
		std::mt19937_64 rng(20260808u + static_cast<uint64_t>(G_L) * 7919u);
		std::vector<double> base_static_errs, mp_static_errs, mp_oracle_errs;
		size_t mp_clamps = 0, base_clamps = 0;
		uint64_t mp_sat = 0, base_sat = 0;
		for (size_t i = 0; i < kPopulation; ++i) {
			Row row = DrawRowBanded(rng, m_a, e_a, wander);
			base_static_errs.push_back(RunBaseWithPair(row, m_a, e_a, table.head_pair, table.s_head, &base_sat, &base_clamps));
			mp_static_errs.push_back(RunM3PrimeWithTable(row, m_a, e_a, table, G_L, &mp_sat, &mp_clamps));
			const StaticTable ot = OracleTable(row, G_L);
			mp_oracle_errs.push_back(RunM3PrimeWithTable(row, m_a, e_a, ot, G_L));
		}
		Stats bs = Summarize(base_static_errs), ms = Summarize(mp_static_errs), mo = Summarize(mp_oracle_errs);
		char kbuf[48];
		snprintf(kbuf, sizeof kbuf, "%d,%d%s", table.k_g[0], table.k_g.size() > 1 ? table.k_g[1] : 0,
		         table.k_g.size() > 2 ? ",..." : "");
		printf("%-5zu %-12.5g %-12.5g %-9.2fx %-12.5g %-9.2fx %-6d %-8s %zu/%zu\n",
		       G_L, bs.mean, ms.mean, ms.mean / bs.mean, mo.mean, mo.mean / bs.mean,
		       table.K_h, kbuf, mp_clamps, kPopulation * kHeadDim);
	}
}

}  // namespace

int main() {
	const int64_t m_a = 1181116006LL;
	const int64_t e_a = -34;

	printf("T-1857 strike on M3' -- head_dim=%zu, position 0 (RoPE identity), engine composite\n", kHeadDim);
	printf("m_a=%lld e_a=%lld  N=%zu eval rows, %zu calibration rows, per G_L\n",
	       (long long)m_a, (long long)e_a, kPopulation, kCalibPopulation);
	printf("BASE_static: one offline per-head pair, frozen (section 8.2a: no runtime reciprocal at site 4)\n");
	printf("M3P_static : M3' AS SPECIFIED -- k_g[g] a build-time constant array, frozen (section 6.9)\n");
	printf("M3P_oracle : the fold's own probe -- k_g recomputed from the row being landed\n");

	// Control: the fold's own stationary geometry. M3P_static and M3P_oracle must agree here,
	// which is why the fold's populations could not see the substitution.
	RunBlock("CONTROL A -- the fold's own population geometry", false, Policy::Max, m_a, e_a);
	RunBlock("CONTROL B -- the fold's own geometry, benefit-seeking calibration", false, Policy::P999, m_a, e_a);

	// The strike: one variable changed -- which channels are loud.
	RunBlock("STRIKE A -- wandering channels, safe calibration", true, Policy::Max, m_a, e_a);
	RunBlock("STRIKE B -- wandering channels, benefit-seeking calibration", true, Policy::P999, m_a, e_a);

	return 0;
}
