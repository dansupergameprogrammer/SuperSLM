// Attention and per-row sites plan (rev 3.1): the cells of the plan's coverage model (§8) that its
// slices own, in their own translation unit. test_main.cpp calls RunAttnRowsiteCells and adds this
// unit's check and failure counts to its own totals, the pattern test_tiled_gemm.cpp set.
//
// Slice S1 (§4.1): 256-entry per-row tables in RmsNormSite (the divide), MlpActSite (the SiLU
// sigmoid) and ResidualReconcileSite (the landing rescale). Every table holds the same pure
// function's value at each code, computed with exactly the arguments the per-element loop passes
// (§5.1), so every S1 cell is an equality cell against the v1.9.0 per-element construction plus a
// path assertion: the row-table counter delta of the one call (§8 path rule, §3.6).
//
// "Reference" below is a test-side copy of the v1.9.0 per-element loop of each site, built from
// the public pure functions (FloorDivI64, SiluSigmoidQ15, LandingRescale) and the unchanged funnel
// (RequantChainChecked, PreflightRequantChain), never the build under test's site body; the golden
// cell (6.3) additionally pins the digest input set to a hash generated from the v1.9.0 tag's own
// library (tools/gen_attn_rowsite_golden.cpp). The expected path is a test-side copy of the guard,
// written from §4.1: tables are taken exactly when n >= 512, in every build that has a test binary.
//
// A build without the instrument seam (build.bat's MSVC recipe) runs the value checks alone and
// says so. Cell numbers are the plan's §8 numbering.

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <span>
#include <string>
#include <thread>
#include <vector>

#include "superslm/checked_chain_funnel.h"
#include "superslm/forward_sites.h"
#include "superslm/intmath.h"
#include "superslm/layer_marshal.h"
#include "superslm/matmul.h"
#include "superslm/model.h"
#include "superslm/sha256.h"
#include "superslm/silu_lut.h"
#include "superslm/silu_lut_canonical.h"
#include "superslm/trace_hook.h"
#include "attn_rowsite_golden_pin.h"
#include "support/attention_cases.h"
#include "support/matmul_dispatch_instrument.h"
#include "support/rowsite_cases.h"

static int GChecks = 0;
static int GFailures = 0;

#define CHECK_MSG(cond, ...) \
	do { \
		++GChecks; \
		if (!(cond)) { \
			++GFailures; \
			std::printf("FAIL %s:%d: %s -- ", __FILE__, __LINE__, #cond); \
			std::printf(__VA_ARGS__); \
			std::printf("\n"); \
		} \
	} while (0)

namespace {

using superslm::CarriedScale;
using superslm::SslmForwardStatus;
using superslm::SslmForwardStatusName;
using superslm_rowsite_cases::FillRow;
using superslm_rowsite_cases::kPoison;
using superslm_rowsite_cases::Rng;
using superslm_rowsite_cases::RowShape;

// The test-side copy of the S1 guard (§4.1): the threshold, and "tables off" only under forced
// scalar (which builds no test binary, §3.3).
constexpr size_t kTestRowTableMinWidth = 512;
bool ExpectTableTaken(size_t n) {
#if defined(SUPERSLM_FORCE_SCALAR_MATMUL)
	(void)n;
	return false;
#else
	return n >= kTestRowTableMinWidth;
#endif
}

// ---- the row-table counters (§3.6) --------------------------------------------------------------

enum RowSite : int { kNorm = 0, kSilu = 1, kLanding = 2 };
const char* const kRowSiteNames[] = {"norm", "silu", "landing"};

struct RowCounters {
	long long taken[3] = {0, 0, 0};
	long long skipped[3] = {0, 0, 0};
};

#if defined(SUPERSLM_ENABLE_MATMUL_DISPATCH_INSTRUMENT)
constexpr bool kHaveRowCounters = true;
RowCounters ReadRowCounters() {
	RowCounters c;
	c.taken[kNorm] = superslm_test::g_rowtable_norm_taken.load();
	c.skipped[kNorm] = superslm_test::g_rowtable_norm_skipped.load();
	c.taken[kSilu] = superslm_test::g_rowtable_silu_taken.load();
	c.skipped[kSilu] = superslm_test::g_rowtable_silu_skipped.load();
	c.taken[kLanding] = superslm_test::g_rowtable_landing_taken.load();
	c.skipped[kLanding] = superslm_test::g_rowtable_landing_skipped.load();
	return c;
}
#else
constexpr bool kHaveRowCounters = false;
RowCounters ReadRowCounters() { return RowCounters{}; }
#endif

RowCounters Delta(const RowCounters& a, const RowCounters& b) {
	RowCounters d;
	for (int s = 0; s < 3; ++s) {
		d.taken[s] = b.taken[s] - a.taken[s];
		d.skipped[s] = b.skipped[s] - a.skipped[s];
	}
	return d;
}

// The path rule for one site call: the site's own pair moves by exactly one on the side the guard
// copy names (or not at all when the call is refused before its table decision), and every other
// row-table counter stays put.
void CheckPath(const char* label, const RowCounters& d, RowSite site, bool counted, size_t n) {
	if (!kHaveRowCounters) return;
	const bool taken = ExpectTableTaken(n);
	for (int s = 0; s < 3; ++s) {
		const long long want_taken = (s == site && counted && taken) ? 1 : 0;
		const long long want_skipped = (s == site && counted && !taken) ? 1 : 0;
		CHECK_MSG(d.taken[s] == want_taken && d.skipped[s] == want_skipped,
		          "%s n=%zu: rowtable_%s taken +%lld skipped +%lld, want +%lld/+%lld", label, n,
		          kRowSiteNames[s], d.taken[s], d.skipped[s], want_taken, want_skipped);
	}
}

// ---- the v1.9.0 per-element references ----------------------------------------------------------

struct SiteResult {
	SslmForwardStatus status = SslmForwardStatus::Ok;
	CarriedScale scale{-1, -1};
	std::vector<int8_t> out;
};

// RmsNormSite as v1.9.0 writes it: sumsq, root, then FloorDivI64(h << 32, root) * g per element.
SiteResult RefNorm(const int8_t* h, const int32_t* g, size_t n, CarriedScale site_constant) {
	SiteResult r;
	r.out.assign(n, kPoison);
	int64_t sumsq = 0;
	for (size_t i = 0; i < n; ++i) sumsq += static_cast<int64_t>(h[i]) * static_cast<int64_t>(h[i]);
	int64_t root = superslm::ISqrt(superslm::FloorDivI64(sumsq << 32, static_cast<int64_t>(n)));
	root = root > 1 ? root : 1;
	std::vector<int64_t> wide(n);
	for (size_t i = 0; i < n; ++i)
		wide[i] = superslm::FloorDivI64(static_cast<int64_t>(h[i]) << 32, root) * static_cast<int64_t>(g[i]);
	r.status = superslm::RequantChainChecked(wide.data(), n, std::span<const CarriedScale>{}, site_constant,
	                                         r.out.data(), &r.scale)
	               .status;
	return r;
}

// MlpActSite as v1.9.0 writes it: domain check, then g * SiluSigmoidQ15(g) * u per element.
SiteResult RefSilu(const int8_t* gate, CarriedScale gate_scale, const int8_t* up, CarriedScale up_scale, size_t n,
                   CarriedScale site_constant) {
	SiteResult r;
	r.out.assign(n, kPoison);
	r.status = superslm::CheckSiluCompositionScaleDomain(gate_scale.m, gate_scale.e);
	if (r.status != SslmForwardStatus::Ok) return r;
	std::vector<int64_t> wide(n);
	for (size_t i = 0; i < n; ++i) {
		const int32_t sig = superslm::SiluSigmoidQ15(superslm::kSiluLutCanonicalTable, gate[i], gate_scale.m,
		                                             static_cast<int>(gate_scale.e));
		wide[i] = static_cast<int64_t>(gate[i]) * static_cast<int64_t>(sig) * static_cast<int64_t>(up[i]);
	}
	const CarriedScale incoming[2] = {gate_scale, up_scale};
	r.status = superslm::RequantChainChecked(wide.data(), n, std::span<const CarriedScale>{incoming, 2},
	                                         site_constant, r.out.data(), &r.scale)
	               .status;
	return r;
}

// ResidualReconcileSite as v1.9.0 writes it (its selection rule, its per-element landing loop with
// the first-flag return and the overflow checks, its second candidate, its funnel).
SiteResult RefResidual(const int8_t* branch_code, CarriedScale branch_scale, const int8_t* stream_code,
                       CarriedScale stream_scale, size_t n, CarriedScale site_constant) {
	SiteResult r;
	r.out.assign(n, kPoison);
	if (branch_scale.m == 0 || stream_scale.m == 0) {
		r.status = SslmForwardStatus::ResidualReconciliationScaleOutOfDomain;
		return r;
	}
	const int64_t i32min = INT64_C(-2147483648), i32max = INT64_C(2147483647);
	if (branch_scale.m < i32min || branch_scale.m > i32max || stream_scale.m < i32min || stream_scale.m > i32max) {
		r.status = SslmForwardStatus::CarriedScaleMantissaOutOfDomain;
		return r;
	}
	const auto magnitude = [](int64_t m) -> uint64_t {
		return m < 0 ? ~static_cast<uint64_t>(m) + 1u : static_cast<uint64_t>(m);
	};
	const auto exceeds_by_31 = [](int64_t a, int64_t b) {
		return a > b && static_cast<uint64_t>(a) - static_cast<uint64_t>(b) > 31u;
	};
	bool branch_selected;
	if (exceeds_by_31(branch_scale.e, stream_scale.e)) {
		branch_selected = false;
	} else if (exceeds_by_31(stream_scale.e, branch_scale.e)) {
		branch_selected = true;
	} else {
		const int d = static_cast<int>(branch_scale.e - stream_scale.e);
		branch_selected = d >= 0 ? (magnitude(branch_scale.m) << d) < magnitude(stream_scale.m)
		                         : magnitude(branch_scale.m) < (magnitude(stream_scale.m) << -d);
	}
	struct Candidate {
		SslmForwardStatus status = SslmForwardStatus::Ok;
		CarriedScale scale{};
		std::vector<int64_t> wide;
	};
	const auto build = [&](bool select_branch) {
		Candidate c;
		c.scale = select_branch ? branch_scale : stream_scale;
		const CarriedScale other_scale = select_branch ? stream_scale : branch_scale;
		const int8_t* direct_code = select_branch ? branch_code : stream_code;
		const int8_t* other_code = select_branch ? stream_code : branch_code;
		const auto reciprocal = superslm::CarriedScaleNormalizedReciprocal(magnitude(c.scale.m));
		c.wide.resize(n);
		for (size_t i = 0; i < n; ++i) {
			bool exceeded = false;
			int64_t landed = superslm::LandingRescale(static_cast<int64_t>(other_code[i]), other_scale.m, reciprocal.r,
			                                          other_scale.e, c.scale.e, nullptr, &exceeded, nullptr,
			                                          reciprocal.s);
			if (exceeded || (c.scale.m < 0 && landed == INT64_MIN)) {
				c.status = SslmForwardStatus::ResidualReconciliationMagnitudeOutOfDomain;
				return c;
			}
			if (c.scale.m < 0) landed = -landed;
			const int64_t direct = static_cast<int64_t>(direct_code[i]);
			if ((landed > 0 && direct > INT64_MAX - landed) || (landed < 0 && direct < INT64_MIN - landed)) {
				c.status = SslmForwardStatus::ResidualReconciliationMagnitudeOutOfDomain;
				return c;
			}
			c.wide[i] = direct + landed;
		}
		const CarriedScale incoming[1] = {c.scale};
		c.status = superslm::PreflightRequantChain(c.wide.data(), n, std::span<const CarriedScale>{incoming, 1},
		                                           site_constant)
		               .status;
		return c;
	};
	Candidate c = build(branch_selected);
	if (c.status != SslmForwardStatus::Ok) c = build(!branch_selected);
	if (c.status != SslmForwardStatus::Ok) {
		r.status = c.status;
		return r;
	}
	const CarriedScale incoming[1] = {c.scale};
	r.status = superslm::RequantChainChecked(c.wide.data(), n, std::span<const CarriedScale>{incoming, 1},
	                                         site_constant, r.out.data(), &r.scale)
	               .status;
	return r;
}

// Whether a residual call reaches its table decision (the scale checks accept).
bool ResidualCounted(CarriedScale b, CarriedScale s) {
	const int64_t i32min = INT64_C(-2147483648), i32max = INT64_C(2147483647);
	return b.m != 0 && s.m != 0 && b.m >= i32min && b.m <= i32max && s.m >= i32min && s.m <= i32max;
}

void CheckEqual(const char* label, size_t n, const SiteResult& want, SslmForwardStatus st, const CarriedScale& scale,
                const std::vector<int8_t>& out) {
	CHECK_MSG(st == want.status, "%s n=%zu: status %s, reference %s", label, n, SslmForwardStatusName(st),
	          SslmForwardStatusName(want.status));
	if (st == SslmForwardStatus::Ok && want.status == SslmForwardStatus::Ok)
		CHECK_MSG(scale.m == want.scale.m && scale.e == want.scale.e,
		          "%s n=%zu: scale (%lld, %lld), reference (%lld, %lld)", label, n, static_cast<long long>(scale.m),
		          static_cast<long long>(scale.e), static_cast<long long>(want.scale.m),
		          static_cast<long long>(want.scale.e));
	size_t first = n, count = 0;
	for (size_t i = 0; i < n; ++i)
		if (out[i] != want.out[i]) {
			if (first == n) first = i;
			++count;
		}
	CHECK_MSG(count == 0, "%s n=%zu: %zu output bytes differ from the reference, first at %zu (%d vs %d)", label, n,
	          count, first, first < n ? out[first] : 0, first < n ? want.out[first] : 0);
}

// One call through each site against its reference, with the path assertion.
void RunNorm(const char* label, const std::vector<int8_t>& h, const std::vector<int32_t>& g,
             CarriedScale site_constant) {
	const size_t n = h.size();
	const SiteResult want = RefNorm(h.data(), g.data(), n, site_constant);
	std::vector<int8_t> out(n, kPoison);
	CarriedScale scale{-1, -1};
	const RowCounters c0 = ReadRowCounters();
	const SslmForwardStatus st =
	    superslm::RmsNormSite(h.data(), g.data(), n, CarriedScale{}, site_constant, out.data(), &scale);
	const RowCounters c1 = ReadRowCounters();
	CheckEqual(label, n, want, st, scale, out);
	CheckPath(label, Delta(c0, c1), kNorm, /*counted=*/true, n);
}

void RunSilu(const char* label, const std::vector<int8_t>& gate, CarriedScale gate_scale,
             const std::vector<int8_t>& up, CarriedScale up_scale, CarriedScale site_constant) {
	const size_t n = gate.size();
	const SiteResult want = RefSilu(gate.data(), gate_scale, up.data(), up_scale, n, site_constant);
	std::vector<int8_t> out(n, kPoison);
	CarriedScale scale{-1, -1};
	const RowCounters c0 = ReadRowCounters();
	const SslmForwardStatus st = superslm::MlpActSite(gate.data(), gate_scale, up.data(), up_scale, n,
	                                                  superslm::kSiluLutCanonicalTable, site_constant,
	                                                  out.data(), &scale);
	const RowCounters c1 = ReadRowCounters();
	CheckEqual(label, n, want, st, scale, out);
	const bool counted =
	    superslm::CheckSiluCompositionScaleDomain(gate_scale.m, gate_scale.e) == SslmForwardStatus::Ok;
	CheckPath(label, Delta(c0, c1), kSilu, counted, n);
}

SslmForwardStatus RunResidual(const char* label, const std::vector<int8_t>& branch, CarriedScale branch_scale,
                              const std::vector<int8_t>& stream, CarriedScale stream_scale,
                              CarriedScale site_constant) {
	const size_t n = branch.size();
	const SiteResult want = RefResidual(branch.data(), branch_scale, stream.data(), stream_scale, n, site_constant);
	std::vector<int8_t> out(n, kPoison);
	CarriedScale scale{-1, -1};
	const RowCounters c0 = ReadRowCounters();
	const SslmForwardStatus st = superslm::ResidualReconcileSite(branch.data(), branch_scale, stream.data(),
	                                                             stream_scale, n, site_constant, out.data(), &scale);
	const RowCounters c1 = ReadRowCounters();
	CheckEqual(label, n, want, st, scale, out);
	CheckPath(label, Delta(c0, c1), kLanding, ResidualCounted(branch_scale, stream_scale), n);
	return st;
}

// The widths of 4.S1 (both sides of 512 and the 0.5B's widths).
constexpr size_t kGridWidths[] = {1, 64, 255, 511, 512, 513, 896, 4864};

// ---- 4.S1: the shape grid, every site ----------------------------------------------------------

// The −128 rows at a small gate scale (4.S1, §9 "−128 read from the table"): at m = 2^30, e = −35
// (128·scale ≤ 4) sig(−128) = 589 is neither 0 nor sig(−127) = 608, so a table entry left unset or
// filled from −127 changes the output. The premises are asserted, not assumed.
void TestS1SmallGateScalePremise() {
	const CarriedScale small{INT64_C(1073741824), -35};
	CHECK_MSG(superslm::CheckSiluCompositionScaleDomain(small.m, small.e) == SslmForwardStatus::Ok,
	          "4.S1 premise: MlpActSite's domain check accepts the small gate scale (2^30, -35)");
	const int32_t s128 = superslm::SiluSigmoidQ15(superslm::kSiluLutCanonicalTable, -128, small.m, -35);
	const int32_t s127 = superslm::SiluSigmoidQ15(superslm::kSiluLutCanonicalTable, -127, small.m, -35);
	CHECK_MSG(s128 == 589 && s127 == 608, "4.S1 premise: sig(-128) = %d (want 589), sig(-127) = %d (want 608)",
	          s128, s127);
}

void TestS1Grid() {
	Rng rng(0x3453314752494431ULL);  // "4S1GRID1"
	const RowShape shapes[] = {RowShape::kRandomWithMinus128, RowShape::kRandomNo128, RowShape::kAllMinus127,
	                           RowShape::kAll127};
	const CarriedScale unit{INT64_C(1073741824), 0};
	const CarriedScale gate_scales[] = {{INT64_C(1073741824), -35}, {INT64_C(1073741824), -34},
	                                    {INT64_C(1631069115), -30}};
	const CarriedScale up_scale{INT64_C(1340958474), -18};
	for (size_t n : kGridWidths)
		for (RowShape shape : shapes) {
			char label[96];
			std::snprintf(label, sizeof label, "4.S1 norm shape %d", static_cast<int>(shape));
			std::vector<int8_t> h(n);
			FillRow(h, shape, rng);
			std::vector<int32_t> g(n);
			for (auto& v : g) v = static_cast<int32_t>(rng.InRange(-300, 300));
			RunNorm(label, h, g, unit);

			for (const CarriedScale& gs : gate_scales) {
				std::snprintf(label, sizeof label, "4.S1 silu shape %d gate e=%lld", static_cast<int>(shape),
				              static_cast<long long>(gs.e));
				std::vector<int8_t> gate(n), up(n);
				FillRow(gate, shape, rng);
				FillRow(up, RowShape::kRandomNo128, rng);
				RunSilu(label, gate, gs, up, up_scale, unit);
			}

			std::snprintf(label, sizeof label, "4.S1 residual shape %d", static_cast<int>(shape));
			std::vector<int8_t> branch(n), stream(n);
			FillRow(branch, shape, rng);
			FillRow(stream, RowShape::kRandomWithMinus128, rng);
			RunResidual(label, branch, {INT64_C(1234567890), -37}, stream, {INT64_C(1987654321), -44}, unit);
			RunResidual(label, stream, {INT64_C(-1400000000), -39}, branch, {INT64_C(1100000000), -41}, unit);
		}

	// −128 at exactly one position (first, middle, last), everything else a code whose value differs
	// from −128's, at the small gate scale, both sides of the threshold.
	for (size_t n : {size_t{64}, size_t{511}, size_t{512}, size_t{896}})
		for (size_t pos : {size_t{0}, n / 2, n - 1}) {
			char label[96];
			std::snprintf(label, sizeof label, "4.S1 silu -128 at %zu, small gate scale", pos);
			std::vector<int8_t> gate(n, -127), up(n, 127);
			gate[pos] = -128;
			RunSilu(label, gate, {INT64_C(1073741824), -35}, up, up_scale, unit);
			std::snprintf(label, sizeof label, "4.S1 norm -128 at %zu", pos);
			std::vector<int8_t> h(n, 1);
			h[pos] = -128;
			std::vector<int32_t> g(n, 7);
			RunNorm(label, h, g, unit);
			std::snprintf(label, sizeof label, "4.S1 residual -128 at %zu", pos);
			std::vector<int8_t> other(n, 3);
			other[pos] = -128;
			std::vector<int8_t> direct(n, -5);
			RunResidual(label, direct, {INT64_C(1234567890), -37}, other, {INT64_C(1987654321), -44}, unit);
		}

	// A refused call counts nothing: a gate scale outside the SiLU domain, a zero residual mantissa.
	{
		std::vector<int8_t> gate(896, 5), up(896, 5);
		RunSilu("4.S1 silu refused by the domain check", gate, {INT64_C(1073741824), 9}, up, up_scale, unit);
		RunResidual("4.S1 residual refused by a zero mantissa", gate, {0, -37}, up, {INT64_C(1987654321), -44},
		            unit);
	}
}

// ---- 1.S1: no table kept across calls ----------------------------------------------------------

void TestS1ConsecutiveRowsDifferentConstants() {
	Rng rng(0x3153315245555345ULL);
	const CarriedScale unit{INT64_C(1073741824), 0};
	const size_t n = 896;
	for (int round = 0; round < 2; ++round) {
		// Norm: the row constant is the root, so two rows with different magnitudes.
		std::vector<int8_t> h(n);
		for (auto& v : h) v = static_cast<int8_t>(rng.InRange(round == 0 ? -20 : -127, round == 0 ? 20 : 127));
		std::vector<int32_t> g(n);
		for (auto& v : g) v = static_cast<int32_t>(rng.InRange(-300, 300));
		RunNorm(round == 0 ? "1.S1 norm row 1" : "1.S1 norm row 2", h, g, unit);
		// SiLU: the row constants are the gate scale.
		std::vector<int8_t> gate(n), up(n);
		FillRow(gate, RowShape::kRandomNo128, rng);
		FillRow(up, RowShape::kRandomNo128, rng);
		RunSilu(round == 0 ? "1.S1 silu row 1" : "1.S1 silu row 2", gate,
		        round == 0 ? CarriedScale{INT64_C(1073741824), -34} : CarriedScale{INT64_C(1631069115), -30}, up,
		        {INT64_C(1340958474), -18}, unit);
		// Residual: the row constants are the scale pair.
		std::vector<int8_t> branch(n), stream(n);
		FillRow(branch, RowShape::kRandomNo128, rng);
		FillRow(stream, RowShape::kRandomNo128, rng);
		RunResidual(round == 0 ? "1.S1 residual row 1" : "1.S1 residual row 2", branch,
		            round == 0 ? CarriedScale{INT64_C(1234567890), -37} : CarriedScale{INT64_C(1100000000), -52},
		            stream,
		            round == 0 ? CarriedScale{INT64_C(1987654321), -44} : CarriedScale{INT64_C(-1900000000), -32},
		            unit);
	}
}

// ---- 3.S1: concurrent calls share no table ------------------------------------------------------

// The plan's 3.S1 names "the existing concurrent-read stress cell ... on the 0.5B-width artifact"; no
// such artifact-driven cell exists in the suite, so this is its own: 8 threads call all three sites at
// table-taking widths on their own rows, each result compared with the single-threaded reference.
// The tables are stack arrays, so nothing is shared; the hosted TSan leg runs this cell.
void TestS1ConcurrentCalls() {
	constexpr int kThreads = 8, kCalls = 24;
	const CarriedScale unit{INT64_C(1073741824), 0};
	struct Job {
		std::vector<int8_t> h, gate, up, branch, stream;
		std::vector<int32_t> g;
		SiteResult want_norm, want_silu, want_res;
		int mismatches = 0;
	};
	std::vector<Job> jobs(kThreads);
	for (int t = 0; t < kThreads; ++t) {
		Rng rng(0x3353314354485200ULL + static_cast<uint64_t>(t));
		Job& j = jobs[t];
		j.h.resize(896); j.g.resize(896); j.branch.resize(896); j.stream.resize(896); j.gate.resize(4864); j.up.resize(4864);
		FillRow(j.h, RowShape::kRandomWithMinus128, rng);
		for (auto& v : j.g) v = static_cast<int32_t>(rng.InRange(-300, 300));
		FillRow(j.gate, RowShape::kRandomWithMinus128, rng);
		FillRow(j.up, RowShape::kRandomNo128, rng);
		FillRow(j.branch, RowShape::kRandomWithMinus128, rng);
		FillRow(j.stream, RowShape::kRandomWithMinus128, rng);
		j.want_norm = RefNorm(j.h.data(), j.g.data(), 896, unit);
		j.want_silu = RefSilu(j.gate.data(), {INT64_C(1073741824), -34 + t % 3}, j.up.data(), {INT64_C(1340958474), -18},
		                      4864, unit);
		j.want_res = RefResidual(j.branch.data(), {INT64_C(1234567890), -37 - t % 4}, j.stream.data(),
		                         {INT64_C(1987654321), -44}, 896, unit);
	}
	std::vector<std::thread> threads;
	for (int t = 0; t < kThreads; ++t)
		threads.emplace_back([&jobs, t, unit] {
			Job& j = jobs[t];
			for (int c = 0; c < kCalls; ++c) {
				std::vector<int8_t> out(4864, kPoison);
				CarriedScale sc{-1, -1};
				SslmForwardStatus st = superslm::RmsNormSite(j.h.data(), j.g.data(), 896, CarriedScale{}, unit, out.data(), &sc);
				if (st != j.want_norm.status || sc.m != j.want_norm.scale.m || sc.e != j.want_norm.scale.e ||
				    !std::equal(j.want_norm.out.begin(), j.want_norm.out.end(), out.begin()))
					++j.mismatches;
				st = superslm::MlpActSite(j.gate.data(), {INT64_C(1073741824), -34 + t % 3}, j.up.data(),
				                          {INT64_C(1340958474), -18}, 4864, superslm::kSiluLutCanonicalTable, unit,
				                          out.data(), &sc);
				if (st != j.want_silu.status || sc.m != j.want_silu.scale.m || sc.e != j.want_silu.scale.e ||
				    !std::equal(j.want_silu.out.begin(), j.want_silu.out.end(), out.begin()))
					++j.mismatches;
				std::fill(out.begin(), out.end(), kPoison);
				st = superslm::ResidualReconcileSite(j.branch.data(), {INT64_C(1234567890), -37 - t % 4}, j.stream.data(),
				                                     {INT64_C(1987654321), -44}, 896, unit, out.data(), &sc);
				if (st != j.want_res.status || sc.m != j.want_res.scale.m || sc.e != j.want_res.scale.e ||
				    !std::equal(j.want_res.out.begin(), j.want_res.out.end(), out.begin()))
					++j.mismatches;
			}
		});
	for (auto& th : threads) th.join();
	int total = 0;
	for (const Job& j : jobs) total += j.mismatches;
	CHECK_MSG(total == 0, "3.S1: %d of %d concurrent site calls (8 threads) differ from the single-threaded reference",
	          total, kThreads * kCalls * 3);
}

// ---- 5.S1 and 7.S1c: the landing flag, read per element present ---------------------------------

// Branch e = 30, stream e = −30: the gap exceeds 31, so the stream candidate is built first and lands
// the branch codes. Code 127 overflows int64 there; code 0 does not.
void TestS1LandingFlag() {
	const CarriedScale coarse{INT64_C(1300000000), 30};
	const CarriedScale fine{INT64_C(1300000000), -30};
	const CarriedScale unit{INT64_C(1073741824), 0};
	const CarriedScale bad_site_constant{INT64_C(4294967296), 0};

	// The premises: the table's entry for 127 carries the flag and the entry for 0 does not.
	const auto rec = superslm::CarriedScaleNormalizedReciprocal(static_cast<uint64_t>(fine.m));
	bool flag127 = false, flag0 = true;
	(void)superslm::LandingRescale(127, coarse.m, rec.r, coarse.e, fine.e, nullptr, &flag127, nullptr, rec.s);
	(void)superslm::LandingRescale(0, coarse.m, rec.r, coarse.e, fine.e, nullptr, &flag0, nullptr, rec.s);
	CHECK_MSG(flag127 && !flag0, "5.S1 premise: landing flag at code 127 = %d (want 1), at code 0 = %d (want 0)",
	          flag127 ? 1 : 0, flag0 ? 1 : 0);

	for (size_t n : {size_t{64}, size_t{511}, size_t{512}, size_t{896}, size_t{4864}}) {
		Rng rng(0x3553314C414E4400ULL + n);
		std::vector<int8_t> stream(n);
		FillRow(stream, RowShape::kRandomWithMinus128, rng);

		// 5.S1 (a): the first candidate is refused mid-row, the second commits.
		std::vector<int8_t> branch(n, 0);
		branch[n / 2] = 127;
		const SslmForwardStatus a = RunResidual("5.S1 first candidate refused mid-row", branch, coarse, stream, fine, unit);
		CHECK_MSG(a == SslmForwardStatus::Ok, "5.S1 n=%zu: the second candidate commits (status %s)", n,
		          SslmForwardStatusName(a));

		// 5.S1 (b): both candidates fail: the first by the landing flag mid-row, the second by the funnel.
		const SslmForwardStatus b =
		    RunResidual("5.S1 both candidates refused", branch, coarse, stream, fine, bad_site_constant);
		CHECK_MSG(b == SslmForwardStatus::CarriedScaleMantissaOutOfDomain,
		          "5.S1 n=%zu: both refused returns the second candidate's status (got %s)", n,
		          SslmForwardStatusName(b));

		// 7.S1c: the row's codes never overflow (every branch code is 0) while code 127 would. The
		// first candidate commits: a flag OR-ed over the whole table would refuse it and commit the
		// second candidate's different scale.
		std::vector<int8_t> zeros(n, 0);
		const SslmForwardStatus c = RunResidual("7.S1c flag read per element present", zeros, coarse, stream, fine, unit);
		CHECK_MSG(c == SslmForwardStatus::Ok, "7.S1c n=%zu: the first candidate commits (status %s)", n,
		          SslmForwardStatusName(c));
		const SiteResult first = RefResidual(zeros.data(), coarse, stream.data(), fine, n, unit);
		CHECK_MSG(first.status == SslmForwardStatus::Ok && first.scale.e != coarse.e,
		          "7.S1c premise n=%zu: the committed scale is the fine candidate's, not the coarse one's", n);
	}
}

// ---- 6.3: the golden pin (the v1.9.0 tag's hash over the digest input set) ----------------------

void TestS1GoldenPin() {
	superslm::Sha256 h;
	uint64_t values = 0;
	auto emit = [&](int64_t v) {
		uint8_t b[8];
		for (int i = 0; i < 8; ++i) b[i] = static_cast<uint8_t>((static_cast<uint64_t>(v) >> (8 * i)) & 0xffU);
		h.Update(b, 8);
		++values;
	};
	const RowCounters c0 = ReadRowCounters();
	superslm_rowsite_cases::RunRowTableCases(emit);
	const RowCounters d = Delta(c0, ReadRowCounters());
	uint8_t digest[32];
	h.Final(digest);
	const std::string hex = superslm::ToHex(digest);
	std::printf("attn-rowsites S1 golden hash: %s (%llu values)\n", hex.c_str(),
	            static_cast<unsigned long long>(values));
	CHECK_MSG(hex == std::string(superslm_test::kAttnRowsiteS1GoldenHash) &&
	              values == superslm_test::kAttnRowsiteS1GoldenValues,
	          "6.3 S1 golden: %s over %llu values, pin %s over %llu (v1.9.0 tag)", hex.c_str(),
	          static_cast<unsigned long long>(values), superslm_test::kAttnRowsiteS1GoldenHash,
	          static_cast<unsigned long long>(superslm_test::kAttnRowsiteS1GoldenValues));
	// The input set reaches both sides of the threshold at every site.
	if (kHaveRowCounters)
		for (int s = 0; s < 3; ++s)
			CHECK_MSG(d.taken[s] > 0 && d.skipped[s] > 0, "6.3: the golden set takes and skips the %s table (+%lld/+%lld)",
			          kRowSiteNames[s], d.taken[s], d.skipped[s]);
}

// ==== Slice S2: prob·V on int16 multiply-add, per head (§4.2, §5.2) ==============================
//
// Every S2 cell calls GemmProbQ15Accumulate directly and asserts two things per call (§8 path rule):
// the output equals a test-side copy of the v1.9.0 loop, and the prob·V path counters moved by
// exactly the delta a test-side copy of the guard names, on the active kernel's own tier only. Which
// kernel runs at all comes first, through a test-side copy of 11.2's selector.

using superslm::detail::GemmTier;
using superslm::detail::SitesKernel;

// The tier the build under test dispatches on: known from the force macro in a forced binary, read
// from the build in the auto binary (the CPU decides there).
GemmTier ExpectedGemmTier() {
#if defined(SUPERSLM_FORCE_SCALAR_MATMUL)
	return GemmTier::kScalar;
#elif defined(SUPERSLM_FORCE_SSE2_MATMUL)
	return GemmTier::kSse2;
#elif defined(SUPERSLM_FORCE_AVX2_MATMUL)
	return GemmTier::kAvx2;
#elif defined(SUPERSLM_FORCE_AVX512_MATMUL)
	return GemmTier::kAvx512;
#else
	return superslm::detail::ActiveGemmTier();
#endif
}

// The test-side switch and compiler identity (§3.2), read from the macros the test itself sees.
#if defined(SUPERSLM_SITES_AVX512_MSVC)
constexpr int kTestSitesAvx512MsvcSwitch = SUPERSLM_SITES_AVX512_MSVC;
#else
constexpr int kTestSitesAvx512MsvcSwitch = 0;
#endif
#if defined(_MSC_VER)
constexpr bool kTestIsMsvcBuild = true;
#else
constexpr bool kTestIsMsvcBuild = false;
#endif

// The test-side copy of 11.2's selector, written from §3.2.
SitesKernel TestSelectSitesKernel(GemmTier tier, int sw, bool msvc) {
	if (tier == GemmTier::kAvx2) return SitesKernel::kAvx2;
	if (tier == GemmTier::kAvx512) return (msvc && sw == 0) ? SitesKernel::kShipped : SitesKernel::kAvx512;
	return SitesKernel::kShipped;
}

SitesKernel ExpectedSitesKernel() {
	return TestSelectSitesKernel(ExpectedGemmTier(), kTestSitesAvx512MsvcSwitch, kTestIsMsvcBuild);
}

const char* SitesKernelName(SitesKernel k) {
	switch (k) {
		case SitesKernel::kShipped: return "v1.9.0 code";
		case SitesKernel::kAvx2: return "AVX2";
		case SitesKernel::kAvx512: return "AVX-512";
	}
	return "?";
}

// The test-side copy of the S2 guard (§4.2): head_dim % 16 == 0 and the int16 condition (every p in
// [0, 32767] and Sum p <= 2^15). The three conjuncts are reported separately so a 2.S2 row can be
// shown to fail exactly one.
struct PvGuard {
	bool hd16 = true, p_nonneg = true, p_le_max = true, sum_le = true;
	bool Fast() const { return hd16 && p_nonneg && p_le_max && sum_le; }
	int Failing() const { return !hd16 + !p_nonneg + !p_le_max + !sum_le; }
};

PvGuard TestPvGuard(const int64_t* probs, size_t width, size_t head_dim) {
	PvGuard g;
	g.hd16 = head_dim % 16 == 0;
	int64_t sum = 0;  // every row of the set is bounded (|p| <= 32,769, width <= 4,097), so int64 is exact
	for (size_t k = 0; k < width; ++k) {
		if (probs[k] < 0) g.p_nonneg = false;
		if (probs[k] > 32767) g.p_le_max = false;
		sum += probs[k];
	}
	g.sum_le = sum <= 32768;
	return g;
}

// The prob·V counters (§3.6), inside the x64 block of the seam.
struct PvCounters {
	long long fast2 = 0, fb2 = 0, fast5 = 0, fb5 = 0;
};
#if defined(SUPERSLM_ENABLE_MATMUL_DISPATCH_INSTRUMENT) && SUPERSLM_MATMUL_HAVE_SIMD_X64
constexpr bool kHavePvCounters = true;
PvCounters ReadPvCounters() {
	PvCounters c;
	c.fast2 = superslm_test::g_pv_fast_avx2.load();
	c.fb2 = superslm_test::g_pv_fallback_avx2.load();
	c.fast5 = superslm_test::g_pv_fast_avx512.load();
	c.fb5 = superslm_test::g_pv_fallback_avx512.load();
	return c;
}
#else
constexpr bool kHavePvCounters = false;
PvCounters ReadPvCounters() { return PvCounters{}; }
#endif

PvCounters PvDelta(const PvCounters& a, const PvCounters& b) {
	return PvCounters{b.fast2 - a.fast2, b.fb2 - a.fb2, b.fast5 - a.fast5, b.fb5 - a.fb5};
}

// The expected delta of `calls` calls of which `fast` take the fast path, on this build (11.1(b)).
PvCounters ExpectedPvDelta(long long fast, long long fallback) {
	PvCounters w;
	switch (ExpectedSitesKernel()) {
		case SitesKernel::kAvx2: w.fast2 = fast; w.fb2 = fallback; break;
		case SitesKernel::kAvx512: w.fast5 = fast; w.fb5 = fallback; break;
		case SitesKernel::kShipped: break;
	}
	return w;
}

void CheckPvDelta(const char* label, const PvCounters& d, const PvCounters& want) {
	if (!kHavePvCounters) return;
	CHECK_MSG(d.fast2 == want.fast2 && d.fb2 == want.fb2 && d.fast5 == want.fast5 && d.fb5 == want.fb5,
	          "%s: pv_fast_avx2 +%lld pv_fallback_avx2 +%lld pv_fast_avx512 +%lld pv_fallback_avx512 +%lld; want "
	          "+%lld/+%lld/+%lld/+%lld (kernel: %s)",
	          label, d.fast2, d.fb2, d.fast5, d.fb5, want.fast2, want.fb2, want.fast5, want.fb5,
	          SitesKernelName(ExpectedSitesKernel()));
}

// The v1.9.0 loop, restated: zero, then key outer, dimension inner, exact int64.
std::vector<int64_t> RefProbV(const int64_t* probs, const int8_t* values, size_t width, size_t head_dim) {
	std::vector<int64_t> out(head_dim, 0);
	for (size_t k = 0; k < width; ++k)
		for (size_t d = 0; d < head_dim; ++d)
			out[d] += probs[k] * static_cast<int64_t>(values[k * head_dim + d]);
	return out;
}

// One call against the reference, with its path assertion. Every buffer is an exact-size heap
// vector, so the hosted ASan leg sees any read or write past one (4.S2's head_dim 60 and 100 rows).
// Returns whether the guard copy expected the fast path.
bool RunPv(const char* label, const std::vector<int64_t>& probs, const std::vector<int8_t>& values, size_t width,
           size_t head_dim) {
	const std::vector<int64_t> want = RefProbV(probs.data(), values.data(), width, head_dim);
	std::vector<int64_t> out(head_dim, superslm_attention_cases::kPvPoison);
	const PvGuard g = TestPvGuard(probs.data(), width, head_dim);
	const PvCounters c0 = ReadPvCounters();
	superslm::GemmProbQ15Accumulate(probs.data(), values.data(), width, head_dim, out.data());
	const PvCounters c1 = ReadPvCounters();
	size_t bad = 0, first = head_dim;
	for (size_t d = 0; d < head_dim; ++d)
		if (out[d] != want[d]) {
			if (first == head_dim) first = d;
			++bad;
		}
	CHECK_MSG(bad == 0, "%s (head_dim %zu, width %zu): %zu of %zu outputs differ from the v1.9.0 loop, first at %zu "
	          "(%lld vs %lld)", label, head_dim, width, bad, head_dim, first,
	          first < head_dim ? static_cast<long long>(out[first]) : 0LL,
	          first < head_dim ? static_cast<long long>(want[first]) : 0LL);
	char full[192];
	std::snprintf(full, sizeof full, "%s (head_dim %zu, width %zu, guard copy: %s)", label, head_dim, width,
	              g.Fast() ? "fast" : "fallback");
	CheckPvDelta(full, PvDelta(c0, c1), ExpectedPvDelta(g.Fast() ? 1 : 0, g.Fast() ? 0 : 1));
	return g.Fast();
}

// ---- 11.2: the MSVC switch, a pure selector with its full truth table, and its wiring ---------------

void TestS2SitesSelector() {
	using superslm::detail::SelectSitesKernel;
	const GemmTier tiers[] = {GemmTier::kScalar, GemmTier::kSse2, GemmTier::kAvx2, GemmTier::kAvx512};
	size_t bad = 0;
	for (GemmTier t : tiers)
		for (bool msvc : {false, true})
			for (int sw : {0, 1}) {
				const SitesKernel got = SelectSitesKernel(t, sw, msvc);
				const SitesKernel want = TestSelectSitesKernel(t, sw, msvc);
				if (got != want) {
					++bad;
					std::printf("  cell 11.2: SelectSitesKernel(tier %d, switch %d, msvc %d) = %s, want %s\n",
					            static_cast<int>(t), sw, msvc ? 1 : 0, SitesKernelName(got), SitesKernelName(want));
				}
			}
	CHECK_MSG(bad == 0, "cell 11.2: %zu of 16 selector rows wrong", bad);
	// The wiring: this build's own switch and compiler identity, on every tier (a hosted runner of any
	// CPU can observe it; the MSVC legs are where the switch's value decides the AVX-512 row).
	for (GemmTier t : tiers)
		CHECK_MSG(superslm::detail::DispatchSitesKernel(t) ==
		              TestSelectSitesKernel(t, kTestSitesAvx512MsvcSwitch, kTestIsMsvcBuild),
		          "cell 11.2 wiring: DispatchSitesKernel(tier %d) = %s, want %s (switch %d, msvc %d)",
		          static_cast<int>(t), SitesKernelName(superslm::detail::DispatchSitesKernel(t)),
		          SitesKernelName(TestSelectSitesKernel(t, kTestSitesAvx512MsvcSwitch, kTestIsMsvcBuild)),
		          kTestSitesAvx512MsvcSwitch, kTestIsMsvcBuild ? 1 : 0);
	std::printf("attn-rowsites S2: tier %d, kernel %s (switch %d, msvc %d), prob-V counters %s\n",
	            static_cast<int>(ExpectedGemmTier()), SitesKernelName(ExpectedSitesKernel()),
	            kTestSitesAvx512MsvcSwitch, kTestIsMsvcBuild ? 1 : 0, kHavePvCounters ? "read" : "not compiled");
}

// ---- 4.S2, 7.S2, 6.1: the shape grid and the int16 condition's corners -----------------------------

void TestS2Grid() {
	size_t fast = 0, fallback = 0;
	superslm_attention_cases::ForEachProbVCase([&](const superslm_attention_cases::PvCase& c) {
		(RunPv(c.label, c.probs, c.values, c.width, c.head_dim) ? fast : fallback) += 1;
	});
	// The grid reaches both sides of every conjunct: fast and fallback both occur.
	CHECK_MSG(fast > 0 && fallback > 0, "4.S2: the set takes the fast path %zu times and falls back %zu times", fast,
	          fallback);
	// The named corners, asserted against the guard copy's own verdict, so a guard copy that drifted
	// from §4.2 cannot quietly move a corner to the other side.
	const struct {
		std::vector<int64_t> p;
		size_t hd;
		bool fast;
		const char* what;
	} corners[] = {
	    {{32767, 1}, 64, true, "p = 32,767 with Sum p = 2^15 is inside"},
	    {{32767, 2}, 64, false, "Sum p = 2^15 + 1 is outside"},
	    {{32768}, 64, false, "the width-1 one-hot row is outside"},
	    {{32767, 1}, 16, true, "head_dim 16 is inside"},
	    {{100, 200}, 60, false, "head_dim 60 is outside"},
	    {{100, 200}, 100, false, "head_dim 100 is outside"},
	};
	for (const auto& k : corners)
		CHECK_MSG(TestPvGuard(k.p.data(), k.p.size(), k.hd).Fast() == k.fast, "7.S2 guard copy: %s", k.what);
}

// ---- 2.S2: hostile rows, each failing exactly one conjunct ------------------------------------------

void TestS2HostileRows() {
	size_t rows = 0;
	superslm_attention_cases::ForEachProbVCase([&](const superslm_attention_cases::PvCase& c) {
		if (std::strncmp(c.label, "2.S2", 4) != 0) return;
		++rows;
		const PvGuard g = TestPvGuard(c.probs.data(), c.width, c.head_dim);
		CHECK_MSG(g.Failing() == 1, "%s: fails %d conjuncts of the guard copy, want exactly 1", c.label, g.Failing());
		// RunPv (in TestS2Grid) already asserted output and fallback +1; the int32-lane premise of the
		// two rows that would wrap is asserted here, so the rows keep deciding what they were built for.
	});
	CHECK_MSG(rows == 4, "2.S2: %zu hostile rows in the set, want 4", rows);
	// The premise of the alternating and the p = 32,767 rows: each lane's true sum, 1,024 x 32,767 x 127,
	// passes INT32_MAX, so a kernel that dropped the conjunct they fail would wrap.
	const int64_t lane = 1024LL * 32767 * 127;
	CHECK_MSG(lane > INT32_MAX, "2.S2 premise: the lane sum %lld exceeds INT32_MAX", static_cast<long long>(lane));
}

// ---- 6.3: the S2 golden pin (the v1.9.0 tag's hash over the prob·V input set) -----------------------

void TestS2GoldenPin() {
	superslm::Sha256 h;
	uint64_t values = 0;
	auto emit = [&](int64_t v) {
		uint8_t b[8];
		for (int i = 0; i < 8; ++i) b[i] = static_cast<uint8_t>((static_cast<uint64_t>(v) >> (8 * i)) & 0xffU);
		h.Update(b, 8);
		++values;
	};
	superslm_attention_cases::RunProbVCases(emit);
	uint8_t digest[32];
	h.Final(digest);
	const std::string hex = superslm::ToHex(digest);
	std::printf("attn-rowsites S2 golden hash: %s (%llu values)\n", hex.c_str(),
	            static_cast<unsigned long long>(values));
	CHECK_MSG(hex == std::string(superslm_test::kAttnRowsiteS2GoldenHash) &&
	              values == superslm_test::kAttnRowsiteS2GoldenValues,
	          "6.3 S2 golden: %s over %llu values, pin %s over %llu (v1.9.0 tag)", hex.c_str(),
	          static_cast<unsigned long long>(values), superslm_test::kAttnRowsiteS2GoldenHash,
	          static_cast<unsigned long long>(superslm_test::kAttnRowsiteS2GoldenValues));
}

// ---- 11.1(d): the 0.5B-width 1-layer artifact through the two layer loops -----------------------

struct TraceCount {
	uint64_t chain = 0;
	std::map<std::string, uint64_t> by_site;
};

void CountTrace(const superslm::SslmChainTraceRecord* chain, const superslm::SslmKvLandingTraceRecord*, void* user) {
	if (!chain) return;
	auto* t = static_cast<TraceCount*>(user);
	++t->chain;
	std::string s(chain->site);
	if (s.rfind("layer", 0) == 0) s = s.substr(s.find('.') + 1);
	++t->by_site[s];
}

bool ReadWholeFile(const char* path, std::vector<uint8_t>& out) {
	std::ifstream f(path, std::ios::binary);
	if (!f) return false;
	out.assign(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
	return !out.empty();
}

// The drive is the plan's (§8 11.1(d)): no ABI, the model's trace hook installed, pinned tokens
// t_i = (37·i + 11) mod 256, a 128-token chunk through RunLayerLoopChunkBatched, then 32 single
// tokens through RunLayerLoop, no final norm and no logits. The artifact is not committed (about
// 16 MB); the cell runs when SUPERSLM_ATTN_ROWSITES_ARTIFACT names it and says SKIPPED otherwise.
void TestS1LayerLoopWindows() {
	const char* path = std::getenv("SUPERSLM_ATTN_ROWSITES_ARTIFACT");
	if (path == nullptr || *path == '\0') {
		std::printf("attn-rowsites 11.1(d): SKIPPED (set SUPERSLM_ATTN_ROWSITES_ARTIFACT to the 0.5B-width 1-layer "
		            "synthetic artifact)\n");
		return;
	}
	using namespace superslm;
	std::vector<uint8_t> bytes;
	CHECK_MSG(ReadWholeFile(path, bytes), "11.1(d): cannot read %s", path);
	if (bytes.empty()) return;
	SslmModelView model;
	std::string error;
	const SslmModelStatus ls = SslmModel::Load(bytes.data(), bytes.size(), model, &error);
	CHECK_MSG(ls == SslmModelStatus::Ok, "11.1(d): load failed: %s", error.c_str());
	if (ls != SslmModelStatus::Ok) return;
	const uint32_t L = model.config.num_hidden_layers;
	const size_t hidden = model.config.hidden_size, head_dim = model.config.head_dim,
	             kv = model.config.num_key_value_heads, H = model.config.num_attention_heads;
	const int64_t cap = model.config.context_cap;
	CHECK_MSG(L == 1 && H == 14 && kv == 2 && head_dim == 64 && hidden == 896 && model.config.intermediate_size == 4864,
	          "11.1(d): the artifact is not the 0.5B-width 1-layer geometry");
	superslm_marshal::PreflightScanWscFolds(model);
	std::vector<superslm_marshal::LayerBacking> backing(L);
	std::vector<LayerWeights> layers(L);
	for (uint32_t l = 0; l < L; ++l)
		CHECK_MSG(superslm_marshal::MarshalLayer(model, l, static_cast<uint32_t>(H), static_cast<uint32_t>(kv),
		                                         backing[l], layers[l], &error),
		          "11.1(d): marshal layer %u: %s", l, error.c_str());
	const int8_t* embed = reinterpret_cast<const int8_t*>(model.weights.Tensor("embed")->data);
	bool ok = true;
	const CarriedScale embed_scale = superslm_marshal::ReadCarriedScale(model.composition_constants, "embed", &ok);
	std::vector<uint8_t> workspace(size_t(L) * size_t(cap) * kv * head_dim * 2);
	TraceCount trace;
	SslmSetTraceHook(model.trace_hook, &CountTrace, &trace);
	const OptionGKLandingMode k_mode =
	    model.option_g_fused_k_landing ? OptionGKLandingMode::kFused : OptionGKLandingMode::kLegacy;
	const auto tok = [](size_t i) { return static_cast<int32_t>((i * 37 + 11) % 256); };
	const size_t T = 128, D = 32;

	// The prefill window.
	const RowCounters p0 = ReadRowCounters();
	const PvCounters v0 = ReadPvCounters();
	const TraceCount tp0 = trace;
	std::vector<int8_t> chunk(T * hidden);
	std::vector<CarriedScale> scales(T);
	for (size_t i = 0; i < T; ++i)
		CHECK_MSG(EmbedEntry(tok(i), static_cast<int32_t>(model.config.vocab_size), embed, hidden, embed_scale,
		                     chunk.data() + i * hidden, &scales[i], "embed", i, &model.trace_hook) ==
		              SslmForwardStatus::Ok,
		          "11.1(d): embed %zu", i);
	SequenceLayerState seq;
	std::vector<int8_t> hc(hidden);
	seq.hidden_codes = hc.data();
	SslmForwardStatus st = RunLayerLoopChunkBatched(
	    chunk.data(), scales.data(), T, layers.data(), L, hidden, head_dim, kv, model.config.intermediate_size, cap, 0,
	    model.rope_tables, workspace.data(), workspace.size(), model.option_g_fused_k_landing, &seq.kv_saturation_count,
	    {}, &model.trace_hook, H * head_dim);
	CHECK_MSG(st == SslmForwardStatus::Ok, "11.1(d): chunk loop status %s", SslmForwardStatusName(st));
	seq.context_length = static_cast<int64_t>(T);
	const RowCounters p1 = ReadRowCounters();
	const PvCounters v1 = ReadPvCounters();
	const TraceCount tp1 = trace;

	// The decode window.
	for (size_t i = 0; i < D && st == SslmForwardStatus::Ok; ++i) {
		CarriedScale sc{};
		CHECK_MSG(EmbedEntry(tok(T + i), static_cast<int32_t>(model.config.vocab_size), embed, hidden, embed_scale,
		                     hc.data(), &sc, "embed", T + i, &model.trace_hook) == SslmForwardStatus::Ok,
		          "11.1(d): embed %zu", T + i);
		seq.hidden_scale = sc;
		seq.layer_index = 0;
		st = RunLayerLoop(seq, layers.data(), L, L, hidden, head_dim, kv, model.config.intermediate_size, cap,
		                  model.rope_tables, workspace.data(), workspace.size(), k_mode, {}, T + i, &model.trace_hook,
		                  H * head_dim);
		CHECK_MSG(st == SslmForwardStatus::Ok, "11.1(d): decode step %zu status %s", i, SslmForwardStatusName(st));
	}
	CHECK_MSG(seq.context_length == static_cast<int64_t>(T + D), "11.1(d): context_length %lld, want %zu",
	          static_cast<long long>(seq.context_length), T + D);
	const RowCounters p2 = ReadRowCounters();
	const PvCounters v2 = ReadPvCounters();
	const TraceCount tp2 = trace;
	SslmSetTraceHook(model.trace_hook, nullptr, nullptr);

	static const char* const kSites[] = {"embed",          "q_proj.requant",   "attn_ctx",        "o_proj.requant",
	                                     "attn_norm",      "attn_residual",    "mlp_norm",        "gate_proj.requant",
	                                     "up_proj.requant", "mlp_act",         "down_proj.requant", "mlp_residual"};
	// The prob·V data terms (§8 11.1(d)): rows failing the int16 condition on this artifact and these
	// pinned tokens, measured on the base (docs/attention-rowsites/s2/pv-data-terms.txt): 15 in the prefill
	// window (position 0's 14 width-1 rows and position 2, head 13's {0, 32,768, 0}), 0 in decode.
	const struct {
		const char* name;
		RowCounters a, b;
		PvCounters va, vb;
		TraceCount ta, tb;
		size_t N;
		long long pv_fallback;
	} windows[] = {{"prefill", p0, p1, v0, v1, tp0, tp1, T, 15}, {"decode", p1, p2, v1, v2, tp1, tp2, D, 0}};
	for (const auto& w : windows) {
		// Structural terms (G26): the closed forms, every width here being >= 512.
		const RowCounters d = Delta(w.a, w.b);
		const long long N = static_cast<long long>(w.N), Ll = static_cast<long long>(L);
		const uint64_t records = w.tb.chain - w.ta.chain;
		CHECK_MSG(records == static_cast<uint64_t>((11 * Ll + 1) * N), "11.1(d) %s: %llu chain trace records, want %lld",
		          w.name, static_cast<unsigned long long>(records), (11 * Ll + 1) * N);
		for (const char* site : kSites) {
			const auto after = w.tb.by_site.find(site);
			const auto before = w.ta.by_site.find(site);
			const uint64_t got = (after == w.tb.by_site.end() ? 0 : after->second) -
			                     (before == w.ta.by_site.end() ? 0 : before->second);
			CHECK_MSG(got == static_cast<uint64_t>(N), "11.1(d) %s: %llu records named %s, want %lld", w.name,
			          static_cast<unsigned long long>(got), site, N);
		}
		// S2: one GemmProbQ15Accumulate call per (layer, head, token pass), L·H·N in all (structural),
		// of which the data term falls back; on the active kernel's tier only (11.1(b)).
		const long long calls = Ll * static_cast<long long>(H) * N;
		char pv_label[64];
		std::snprintf(pv_label, sizeof pv_label, "11.1(d) %s prob-V", w.name);
		CheckPvDelta(pv_label, PvDelta(w.va, w.vb), ExpectedPvDelta(calls - w.pv_fallback, w.pv_fallback));
		if (!kHaveRowCounters) continue;
		const long long want_taken[3] = {2 * Ll * N, Ll * N, 2 * Ll * N};
		for (int s = 0; s < 3; ++s) {
			CHECK_MSG(d.taken[s] == want_taken[s] && d.skipped[s] == 0,
			          "11.1(d) %s: rowtable_%s taken +%lld skipped +%lld, want +%lld/+0", w.name, kRowSiteNames[s],
			          d.taken[s], d.skipped[s], want_taken[s]);
		}
	}
	std::printf("attn-rowsites 11.1(d): prefill and decode windows driven on %s\n", path);
}

}  // namespace

void RunAttnRowsiteCells(int& checks, int& failures) {
	GChecks = 0;
	GFailures = 0;
	if (!kHaveRowCounters)
		std::printf("attn-rowsites: this build has no instrument seam; value checks only, path counters not read\n");
	TestS1SmallGateScalePremise();
	TestS1Grid();
	TestS1ConsecutiveRowsDifferentConstants();
	TestS1ConcurrentCalls();
	TestS1LandingFlag();
	TestS1GoldenPin();
	TestS2SitesSelector();
	TestS2Grid();
	TestS2HostileRows();
	TestS2GoldenPin();
	TestS1LayerLoopWindows();  // 11.1(d): S1's and S2's rows over one drive
	std::printf("attn-rowsites cells (plan slices S1, S2): %d checks, %d failures\n", GChecks, GFailures);
	checks += GChecks;
	failures += GFailures;
}
