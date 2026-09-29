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
// Slice S2 (§4.2): prob·V on int16 multiply-add; slice S3 (§4.3): the requant element loop in 64-bit
// lanes (RequantRowWide), checked against the unchanged per-element RequantTokenCodeWide with sentinel
// fences and exact-size buffers, and counted per tier (requant_row); slice S4 (§4.4): the guarded softmax,
// checked against a test-side restatement of the v1.9.0 SoftmaxRowQ15 body (the unchanged public
// IExpConstruct / IExpEvaluate leaves), bool and output, with the path chosen by a test-side copy of the
// §5.4 guard and the correction rows chosen by a test-side replica of the estimates; slice S5 (§4.5): the Q31
// score row (QkQ31ScoreRow) in three 16-bit pieces, checked against the in-tree scalar reference (the same
// binary's per-key QkQ31Score outside the loader's ratio range), with the path chosen by a test-side copy of
// its guard, and driven through both layer loops on the widened QK-norm fixture (cell 11.1(c)).
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
#include "sslm_c32_softmax_row_width_gate_fixtures.h"
#include "support/attention_cases.h"
#include "support/matmul_dispatch_instrument.h"
#include "support/qk_attention_fixture.h"
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
	// The kernels' own blocking (not in the golden set, so the pin stays the plan's grid): AVX2 runs
	// 16-dimension units four at a time, AVX-512 runs 32-dimension units four at a time plus one
	// 16-dimension tail when head_dim % 32 == 16. These head_dims reach every block count and the tail
	// behind a full block, each at even and odd widths.
	{
		superslm_attention_cases::Rng rng(0x5332424C4F434B53ULL);  // "S2BLOCKS"
		for (size_t hd : {size_t{32}, size_t{48}, size_t{80}, size_t{96}, size_t{144}, size_t{160}, size_t{208},
		                  size_t{224}, size_t{240}})
			for (size_t w : {size_t{1}, size_t{2}, size_t{3}, size_t{64}, size_t{65}}) {
				const std::vector<int64_t> p = superslm_attention_cases::RealisticRow(w, 10, rng);
				const std::vector<int8_t> v = superslm_attention_cases::RandomValues(w * hd, rng);
				(RunPv("4.S2 kernel blocking", p, v, w, hd) ? fast : fallback) += 1;
			}
	}
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

// ==== Slice S3: the requant element loop in 64-bit lanes (§4.3, §5.3) ==============================
//
// Every S3 cell calls the row leaf RequantRowWide directly (or the funnel that calls it) and asserts
// per call (§8 path rule): the codes equal RequantTokenCodeWide element by element (the v1.9.0 leaf,
// unchanged and never the build's row code), nothing outside the row was written, and the requant_row
// counter moved +1 on the selected kernel's own tier and nowhere else. S3 has no runtime guard (§5.3),
// so the expected path is the selector's alone: AVX2 and AVX-512 run the lanes, every other kernel the
// element loop.

struct RqCounters {
	long long avx2 = 0, avx512 = 0;
};
#if defined(SUPERSLM_ENABLE_MATMUL_DISPATCH_INSTRUMENT) && SUPERSLM_MATMUL_HAVE_SIMD_X64
constexpr bool kHaveRqCounters = true;
RqCounters ReadRqCounters() {
	return RqCounters{superslm_test::g_requant_row_avx2.load(), superslm_test::g_requant_row_avx512.load()};
}
#else
constexpr bool kHaveRqCounters = false;
RqCounters ReadRqCounters() { return RqCounters{}; }
#endif

RqCounters RqDelta(const RqCounters& a, const RqCounters& b) { return RqCounters{b.avx2 - a.avx2, b.avx512 - a.avx512}; }

// `calls` row-leaf calls on this build (11.1(b)): the selected kernel's own tier moves, the other does not.
RqCounters ExpectedRqDelta(long long calls) {
	RqCounters w;
	switch (ExpectedSitesKernel()) {
		case SitesKernel::kAvx2: w.avx2 = calls; break;
		case SitesKernel::kAvx512: w.avx512 = calls; break;
		case SitesKernel::kShipped: break;
	}
	return w;
}

bool CheckRqDelta(const char* label, const RqCounters& d, const RqCounters& want) {
	if (!kHaveRqCounters) return true;
	const bool ok = d.avx2 == want.avx2 && d.avx512 == want.avx512;
	CHECK_MSG(ok, "%s: requant_row_avx2 +%lld requant_row_avx512 +%lld; want +%lld/+%lld (kernel: %s)", label, d.avx2,
	          d.avx512, want.avx2, want.avx512, SitesKernelName(ExpectedSitesKernel()));
	return ok;
}

// r and s exactly as the funnel's preflight derives them from d' (checked_chain_funnel.cpp step 4).
struct RqConstants {
	int64_t r;
	int s;
};
RqConstants RqConstantsFor(int64_t d_prime) {
	const superslm::NormalizedScale ns = superslm::NormalizeScale(d_prime);
	return RqConstants{superslm::DynamicScaleReciprocal(ns.dn), ns.s};
}

constexpr size_t kRqFence = 16;
constexpr uint8_t kRqSentinel = 0xA5;

// One row-leaf call against the element loop, with its path assertion. `fenced`: the output row sits
// between 16 sentinel bytes before out[0] and after out[n - 1], checked unchanged after the call (every
// binary). Otherwise the input and the output are exact-size heap buffers, so the hosted ASan leg sees
// any read or write past either (4.S3's second pass). The value assertions are one check each; the
// path assertion is tallied in `path_mismatches` (with the first few printed) and asserted once per
// pass by the caller, so a build whose counter never moves reports one failure per pass, not one per
// row. Returns the number of failed value assertions.
int RunRequantRow(const char* label, const std::vector<int64_t>& row, int64_t d_prime, bool fenced,
                  size_t* path_mismatches) {
	const int f0 = GFailures;
	const size_t n = row.size();
	const RqConstants k = RqConstantsFor(d_prime);
	std::vector<int8_t> want(n);
	for (size_t i = 0; i < n; ++i) want[i] = superslm::RequantTokenCodeWide(row[i], k.r, k.s);
	std::vector<int64_t> x(row);  // exact size
	std::vector<uint8_t> fencedbuf;
	std::vector<int8_t> exact;
	int8_t* out;
	if (fenced) {
		fencedbuf.assign(n + 2 * kRqFence, kRqSentinel);
		out = reinterpret_cast<int8_t*>(fencedbuf.data() + kRqFence);
		std::memset(out, static_cast<uint8_t>(kPoison), n);
	} else {
		exact.assign(n, kPoison);
		out = exact.data();
	}
	const RqCounters c0 = ReadRqCounters();
	superslm::RequantRowWide(x.data(), n, k.r, k.s, out);
	const RqCounters c1 = ReadRqCounters();
	size_t bad = 0, first = n;
	for (size_t i = 0; i < n; ++i)
		if (out[i] != want[i]) {
			if (first == n) first = i;
			++bad;
		}
	CHECK_MSG(bad == 0, "%s (n %zu, d' %lld, r %lld, s %d): %zu of %zu codes differ from RequantTokenCodeWide, first at "
	          "%zu (x %lld: %d vs %d)", label, n, static_cast<long long>(d_prime), static_cast<long long>(k.r), k.s, bad,
	          n, first, first < n ? static_cast<long long>(row[first]) : 0LL, first < n ? out[first] : 0,
	          first < n ? want[first] : 0);
	if (fenced) {
		size_t clobbered = 0;
		for (size_t i = 0; i < kRqFence; ++i) {
			clobbered += fencedbuf[i] != kRqSentinel;
			clobbered += fencedbuf[kRqFence + n + i] != kRqSentinel;
		}
		CHECK_MSG(clobbered == 0, "%s (n %zu, d' %lld): %zu of the 32 sentinel bytes around the row changed", label, n,
		          static_cast<long long>(d_prime), clobbered);
	}
	CHECK_MSG(x == row, "%s (n %zu): the input row changed", label, n);
	const int value_failures = GFailures - f0;
	const RqCounters d = RqDelta(c0, c1), want_d = ExpectedRqDelta(1);
	if (kHaveRqCounters && (d.avx2 != want_d.avx2 || d.avx512 != want_d.avx512)) {
		if (++*path_mismatches <= 3)
			std::printf("  %s (n %zu, d' %lld): requant_row_avx2 +%lld requant_row_avx512 +%lld, want +%lld/+%lld\n", label,
			            n, static_cast<long long>(d_prime), d.avx2, d.avx512, want_d.avx2, want_d.avx512);
	}
	return value_failures;
}

// ---- 7.S3: the P = 2^63 corner's premises (§5.3) --------------------------------------------------

void TestS3CornerPremise() {
	const superslm::NormalizedScale ns = superslm::NormalizeScale(INT64_C(1) << 31);
	CHECK_MSG(ns.dn == (INT64_C(1) << 30) && ns.s == -1, "7.S3 premise: NormalizeScale(2^31) = (%lld, %d), want (2^30, -1)",
	          static_cast<long long>(ns.dn), ns.s);
	const int64_t r = superslm::DynamicScaleReciprocal(ns.dn);
	CHECK_MSG(r == (INT64_C(1) << 32), "7.S3 premise: DynamicScaleReciprocal(2^30) = %lld, want 2^32", static_cast<long long>(r));
	// |x| = 2^31 times r = 2^32 is P = 2^63 exactly: one past INT64_MAX, so a signed lane cannot hold it.
	const uint64_t p = (uint64_t{1} << 31) * static_cast<uint64_t>(r);
	CHECK_MSG(p == (uint64_t{1} << 63), "7.S3 premise: P at the corner is 2^63");
	// Both corner codes are the true +-127 (§5.3: the clamp never fires inside the contract, and the corner
	// itself lands exactly on 127).
	CHECK_MSG(superslm::RequantTokenCodeWide(INT64_C(1) << 31, r, -1) == 127 &&
	              superslm::RequantTokenCodeWide(-(INT64_C(1) << 31), r, -1) == -127,
	          "7.S3 premise: the corner's codes are +-127");
	// Every s the preflight can produce, in [-1, 30], is reached by RequantDPrimes' d' list.
	superslm_rowsite_cases::Rng rng(1);
	bool seen[32] = {};
	for (int64_t dp : superslm_rowsite_cases::RequantDPrimes(rng)) {
		const int s = superslm::NormalizeScale(dp).s;
		if (s >= -1 && s <= 30) seen[s + 1] = true;
	}
	int missing = 0;
	for (bool b : seen) missing += !b;
	CHECK_MSG(missing == 0, "4.S3 premise: %d of the 32 shifts s in [-1, 30] are not reached by the d' list", missing);
}

// ---- 4.S3, 7.S3, 6.1: the shape grid, both fences --------------------------------------------------

// The grid: n {1, 3, 4, 5, 7, 8, 9, 896, 4,864} x d' (1, 2, 2^30, 2^31, their neighbours, and three in
// every octave, so every s in [-1, 30]) x +-d' at every lane position (every position of rows up to 9
// elements, positions 0-8, 15, 16 and the last of the wide rows), the rest drawn from [-d', d']. Then
// random rows: n in [1, 40], d' log-uniform over [1, 2^31], with values biased to +-d' and to rounding
// ties. `fenced` selects the sentinel pass or the exact-size heap pass.
void RunS3Grid(bool fenced) {
	const char* const pass = fenced ? "4.S3 sentinel pass" : "4.S3 exact-size pass";
	superslm_rowsite_cases::Rng rng(fenced ? 0x3453334752494431ULL : 0x3453334752494432ULL);  // "4S3GRID1/2"
	const std::vector<int64_t> dprimes = superslm_rowsite_cases::RequantDPrimes(rng);
	size_t calls = 0, path_bad = 0;
	int failed_calls = 0;
	const RqCounters c0 = ReadRqCounters();
	// A broken body would otherwise print thousands of rows: each loop stops after 20 failing calls.
	for (size_t n : superslm_rowsite_cases::kRequantWidths)
		for (int64_t dp : dprimes) {
			std::vector<size_t> positions;
			if (n <= 9) {
				for (size_t p = 0; p < n; ++p) positions.push_back(p);
			} else {
				positions = {0, 1, 2, 3, 4, 5, 6, 7, 8, 15, 16, n - 1};
			}
			for (size_t p : positions)
				for (int sign : {1, -1}) {
					if (failed_calls > 20) continue;
					std::vector<int64_t> x(n);
					for (auto& v : x) v = rng.InRange(-dp, dp);
					x[p] = sign * dp;
					++calls;
					failed_calls += RunRequantRow(pass, x, dp, fenced, &path_bad) != 0;
				}
		}
	// Every element at +-d' (the corner in every lane at once, at d' = 2^31).
	for (size_t n : superslm_rowsite_cases::kRequantWidths)
		for (int64_t dp : {INT64_C(1), INT64_C(2), INT64_C(1) << 30, INT64_C(1) << 31}) {
			std::vector<int64_t> x(n);
			for (size_t i = 0; i < n; ++i) x[i] = (i % 3 == 1) ? -dp : dp;
			++calls;
			failed_calls += RunRequantRow(pass, x, dp, fenced, &path_bad) != 0;
		}
	// Random rows, and the empty row (reads and writes nothing, still one call).
	for (int t = 0; t < 4000 && failed_calls <= 20; ++t) {
		const size_t n = static_cast<size_t>(rng.InRange(1, 40));
		const int octave = static_cast<int>(rng.InRange(0, 31));
		const int64_t dp = octave == 31 ? (INT64_C(1) << 31) : rng.InRange(INT64_C(1) << octave, (INT64_C(2) << octave) - 1);
		const RqConstants k = RqConstantsFor(dp);
		std::vector<int64_t> x(n);
		for (auto& v : x) {
			switch (rng.InRange(0, 3)) {
				case 0: v = rng.Next() & 1 ? dp : -dp; break;
				case 1: {
					// A value whose |x|·127·r sits next to a rounding tie (e = 62 - s): x = floor(j · 2^(e-1) /
					// (127·r)) + {-1, 0, 1} for a random odd j < 256, clamped into [-d', d']. The quotient is
					// formed exactly in 64 bits: 2^(e-1) = Q·127r + R with R < 127r < 2^39.
					const int e = 62 - k.s;
					const uint64_t j = static_cast<uint64_t>(2 * rng.InRange(0, 127) + 1);
					const uint64_t div = 127u * static_cast<uint64_t>(k.r);
					const uint64_t half = uint64_t{1} << (e - 1);
					int64_t m = static_cast<int64_t>(j * (half / div) + (j * (half % div)) / div);
					m += rng.InRange(-1, 1);
					m = std::clamp<int64_t>(m, 0, dp);
					v = rng.Next() & 1 ? m : -m;
					break;
				}
				default: v = rng.InRange(-dp, dp); break;
			}
		}
		x[static_cast<size_t>(rng.InRange(0, static_cast<int64_t>(n) - 1))] = rng.Next() & 1 ? dp : -dp;
		++calls;
		failed_calls += RunRequantRow(pass, x, dp, fenced, &path_bad) != 0;
	}
	++calls;
	failed_calls += RunRequantRow(pass, std::vector<int64_t>{}, 1, fenced, &path_bad) != 0;
	// The path rule per call (tallied above), then the whole pass as one delta, so a counter that moved on a
	// call the loop above did not see is caught too.
	CHECK_MSG(path_bad == 0, "%s: %zu of %zu row-leaf calls moved the requant_row counters wrongly (kernel: %s)", pass,
	          path_bad, calls, SitesKernelName(ExpectedSitesKernel()));
	CheckRqDelta(pass, RqDelta(c0, ReadRqCounters()), ExpectedRqDelta(static_cast<long long>(calls)));
	std::printf("attn-rowsites S3 %s: %zu row-leaf calls, %d with a wrong code or fence, %zu with a wrong path\n", pass,
	            calls, failed_calls, path_bad);
}

void TestS3Grid() {
	RunS3Grid(/*fenced=*/true);
	RunS3Grid(/*fenced=*/false);
}

// ---- the funnel's call site: one row-leaf call per funnel call that passes its preflight ------------

void TestS3FunnelCallSite() {
	const CarriedScale unit{INT64_C(1073741824), 0};
	superslm_rowsite_cases::Rng rng(0x5333464E4E4C0000ULL);
	for (size_t n : {size_t{0}, size_t{5}, size_t{896}, size_t{4864}}) {
		std::vector<int64_t> x(n);
		for (auto& v : x) v = rng.InRange(-(INT64_C(1) << 31), INT64_C(1) << 31);
		std::vector<int8_t> out(n, kPoison), want(n, kPoison);
		CarriedScale scale{-1, -1};
		const int64_t dp = superslm::MaxAbsReduceWide(x.data(), n);
		const RqConstants k = RqConstantsFor(dp);
		for (size_t i = 0; i < n; ++i) want[i] = superslm::RequantTokenCodeWide(x[i], k.r, k.s);
		const RqCounters c0 = ReadRqCounters();
		const SslmForwardStatus st =
		    superslm::RequantChainChecked(x.data(), n, std::span<const CarriedScale>{}, unit, out.data(), &scale).status;
		const RqCounters c1 = ReadRqCounters();
		CHECK_MSG(st == SslmForwardStatus::Ok && out == want, "S3 funnel n=%zu: status %s, codes %s the element loop's", n,
		          SslmForwardStatusName(st), out == want ? "equal" : "differ from");
		char label[64];
		std::snprintf(label, sizeof label, "S3 funnel call n=%zu", n);
		CheckRqDelta(label, RqDelta(c0, c1), ExpectedRqDelta(1));
	}
	// A refused preflight (d' = 2^31 + 1) writes nothing and never reaches the row leaf.
	std::vector<int64_t> x(896, 3);
	x[100] = (INT64_C(1) << 31) + 1;
	std::vector<int8_t> out(896, kPoison);
	CarriedScale scale{-1, -1};
	const RqCounters c0 = ReadRqCounters();
	const SslmForwardStatus st =
	    superslm::RequantChainChecked(x.data(), x.size(), std::span<const CarriedScale>{}, unit, out.data(), &scale).status;
	CHECK_MSG(st == SslmForwardStatus::ChainInputOutOfDomain &&
	              std::all_of(out.begin(), out.end(), [](int8_t c) { return c == kPoison; }),
	          "S3 funnel refused: status %s, codes untouched", SslmForwardStatusName(st));
	CheckRqDelta("S3 funnel refused by the preflight", RqDelta(c0, ReadRqCounters()), ExpectedRqDelta(0));
}

// ---- 6.3: the S3 golden pin (the v1.9.0 tag's hash over the requant row set) -------------------------

void TestS3GoldenPin() {
	superslm::Sha256 h;
	uint64_t values = 0;
	auto emit = [&](int64_t v) {
		uint8_t b[8];
		for (int i = 0; i < 8; ++i) b[i] = static_cast<uint8_t>((static_cast<uint64_t>(v) >> (8 * i)) & 0xffU);
		h.Update(b, 8);
		++values;
	};
	superslm_rowsite_cases::RunRequantRowCases(emit);
	uint8_t digest[32];
	h.Final(digest);
	const std::string hex = superslm::ToHex(digest);
	std::printf("attn-rowsites S3 golden hash: %s (%llu values)\n", hex.c_str(), static_cast<unsigned long long>(values));
	CHECK_MSG(hex == std::string(superslm_test::kAttnRowsiteS3GoldenHash) &&
	              values == superslm_test::kAttnRowsiteS3GoldenValues,
	          "6.3 S3 golden: %s over %llu values, pin %s over %llu (v1.9.0 tag)", hex.c_str(),
	          static_cast<unsigned long long>(values), superslm_test::kAttnRowsiteS3GoldenHash,
	          static_cast<unsigned long long>(superslm_test::kAttnRowsiteS3GoldenValues));
}

// ==== Slice S4: the guarded softmax (§4.4, §5.4) =====================================================
//
// Every S4 cell calls SoftmaxRowQ15 directly and asserts per call (§8 path rule): the bool and every
// probability equal a test-side restatement of the v1.9.0 body, and the softmax path counters moved by
// exactly the delta that the test-side guard copy (tests/support/attention_cases.h, TestSoftmaxGuard) names,
// on the selected kernel's own tier only. Width 0 counts nowhere (§3.6: "call with width >= 1").

using superslm_attention_cases::SmCase;
using superslm_attention_cases::SoftmaxCorrections;
using superslm_attention_cases::SoftmaxGuard;
using superslm_attention_cases::SoftmaxEstimateReplica;
using superslm_attention_cases::TestSoftmaxGuard;

struct SmCounters {
	long long fast2 = 0, fb2 = 0, fast5 = 0, fb5 = 0;
};
#if defined(SUPERSLM_ENABLE_MATMUL_DISPATCH_INSTRUMENT) && SUPERSLM_MATMUL_HAVE_SIMD_X64
constexpr bool kHaveSmCounters = true;
SmCounters ReadSmCounters() {
	return SmCounters{superslm_test::g_softmax_fast_avx2.load(), superslm_test::g_softmax_fallback_avx2.load(),
	                  superslm_test::g_softmax_fast_avx512.load(), superslm_test::g_softmax_fallback_avx512.load()};
}
#else
constexpr bool kHaveSmCounters = false;
SmCounters ReadSmCounters() { return SmCounters{}; }
#endif

SmCounters SmDelta(const SmCounters& a, const SmCounters& b) {
	return SmCounters{b.fast2 - a.fast2, b.fb2 - a.fb2, b.fast5 - a.fast5, b.fb5 - a.fb5};
}

// `fast` fast and `fallback` fallback calls on this build (11.1(b)).
SmCounters ExpectedSmDelta(long long fast, long long fallback) {
	SmCounters w;
	switch (ExpectedSitesKernel()) {
		case SitesKernel::kAvx2: w.fast2 = fast; w.fb2 = fallback; break;
		case SitesKernel::kAvx512: w.fast5 = fast; w.fb5 = fallback; break;
		case SitesKernel::kShipped: break;
	}
	return w;
}

bool CheckSmDelta(const char* label, const SmCounters& d, const SmCounters& want) {
	if (!kHaveSmCounters) return true;
	const bool ok = d.fast2 == want.fast2 && d.fb2 == want.fb2 && d.fast5 == want.fast5 && d.fb5 == want.fb5;
	CHECK_MSG(ok, "%s: softmax_fast_avx2 +%lld softmax_fallback_avx2 +%lld softmax_fast_avx512 +%lld "
	          "softmax_fallback_avx512 +%lld; want +%lld/+%lld/+%lld/+%lld (kernel: %s)",
	          label, d.fast2, d.fb2, d.fast5, d.fb5, want.fast2, want.fb2, want.fast5, want.fb5,
	          SitesKernelName(ExpectedSitesKernel()));
	return ok;
}

// The v1.9.0 SoftmaxRowQ15 body, restated from the unchanged public leaves: M usable exactly when
// M = q_b^2 + q_c is in [1, 2^47] (read from the guard copy's two M flags, which judge the same 128-bit
// value); max shift; per element IExpConstruct, a kBad* outcome or a value outside [0, M] refuses the element
// (0) and the row (false); denom = max(total, 1); p = (e << 15) / denom.
bool RefSoftmax(const int64_t* scores, size_t width, int64_t q_ln2, int64_t q_b, int64_t q_c, int64_t* out) {
	if (width == 0) return true;
	const SoftmaxGuard g = TestSoftmaxGuard(scores, 1, q_ln2, q_b, q_c);
	const bool m_usable = g.m_ge1 && g.m_le;
	// M fits int64 when usable; formed with unsigned wrap so no intermediate overflows.
	const int64_t m = m_usable ? static_cast<int64_t>(static_cast<uint64_t>(q_b) * static_cast<uint64_t>(q_b) +
	                                                   static_cast<uint64_t>(q_c))
	                           : 0;
	int64_t peak = scores[0];
	for (size_t k = 1; k < width; ++k)
		if (scores[k] > peak) peak = scores[k];
	std::vector<int64_t> e(width, 0);
	int64_t total = 0;
	bool ok = true;
	for (size_t k = 0; k < width; ++k) {
		superslm::IExpConstruction c;
		const superslm::IExpDomain d = superslm::IExpConstruct(scores[k] - peak, q_ln2, q_b, q_c, &c);
		if (d == superslm::IExpDomain::kBadQ || d == superslm::IExpDomain::kBadQLn2 || d == superslm::IExpDomain::kBadQB) {
			ok = false;
			continue;
		}
		const int64_t v = superslm::IExpEvaluate(c);
		if (!m_usable || v < 0 || v > m) {
			ok = false;
			continue;
		}
		e[k] = v;
		total += v;
	}
	const int64_t denom = total > 1 ? total : 1;
	for (size_t k = 0; k < width; ++k) out[k] = (e[k] << superslm::kProbFracBits) / denom;
	return ok;
}

// One call against the reference, with its path assertion. Exact-size heap buffers (the hosted ASan leg).
// Returns whether the guard copy expected the fast path.
bool RunSm(const SmCase& c) {
	const size_t w = c.scores.size();
	std::vector<int64_t> want(w, 0);
	const bool want_ok = RefSoftmax(c.scores.data(), w, c.q_ln2, c.q_b, c.q_c, want.data());
	const SoftmaxGuard g = TestSoftmaxGuard(c.scores.data(), w, c.q_ln2, c.q_b, c.q_c);
	std::vector<int64_t> out = c.aliased ? c.scores : std::vector<int64_t>(w, superslm_attention_cases::kSmPoison);
	const int64_t* in = c.aliased ? out.data() : c.scores.data();
	const SmCounters c0 = ReadSmCounters();
	const bool ok = superslm::SoftmaxRowQ15(in, w, c.q_ln2, c.q_b, c.q_c, out.data());
	const SmCounters c1 = ReadSmCounters();
	size_t bad = 0, first = w;
	for (size_t k = 0; k < w; ++k)
		if (out[k] != want[k]) {
			if (first == w) first = k;
			++bad;
		}
	CHECK_MSG(ok == want_ok && bad == 0,
	          "%s (width %zu, q_ln2 %lld, q_b %lld, q_c %lld): bool %d (v1.9.0 %d); %zu of %zu probabilities differ, "
	          "first at %zu (%lld vs %lld)", c.label, w, static_cast<long long>(c.q_ln2), static_cast<long long>(c.q_b),
	          static_cast<long long>(c.q_c), ok ? 1 : 0, want_ok ? 1 : 0, bad, w, first,
	          first < w ? static_cast<long long>(out[first]) : 0LL, first < w ? static_cast<long long>(want[first]) : 0LL);
	char full[224];
	std::snprintf(full, sizeof full, "%s (width %zu, guard copy: %s)", c.label, w, g.Fast() ? "fast" : "fallback");
	CheckSmDelta(full, SmDelta(c0, c1), ExpectedSmDelta(g.Fast() ? 1 : 0, g.Fast() ? 0 : 1));
	return g.Fast();
}

// ---- 4.S4, 7.S4, 6.1: the grid, the inside corners, the correction rows, the aliased rows ----------

void TestS4Grid() {
	size_t fast = 0, fallback = 0, aliased = 0;
	std::map<std::string, int> corrections;  // label -> rows on which the named correction decides the output
	superslm_attention_cases::ForEachSoftmaxCase([&](const SmCase& c) {
		(RunSm(c) ? fast : fallback) += 1;
		aliased += c.aliased ? 1 : 0;
		const std::string label(c.label);
		if (label.find("correction row") == std::string::npos) return;
		// The row's premise (7.S4b): the replica says the named correction fires on it, and the same
		// arithmetic without that correction gives a different row, so the §9 "skipped" mutant dies here.
		const size_t w = c.scores.size();
		const int skip = label.find("z up") != std::string::npos ? 1 : label.find("p up") != std::string::npos ? 2 : 3;
		std::vector<int64_t> good(w), without(w), want(w);
		SoftmaxCorrections k, unused;
		const bool in = SoftmaxEstimateReplica(c.scores.data(), w, c.q_ln2, c.q_b, c.q_c, good.data(), &k);
		SoftmaxEstimateReplica(c.scores.data(), w, c.q_ln2, c.q_b, c.q_c, without.data(), &unused, skip);
		const bool ref_ok = RefSoftmax(c.scores.data(), w, c.q_ln2, c.q_b, c.q_c, want.data());
		const long long fired = skip == 1 ? k.z_up : skip == 2 ? k.p_up : k.p_down;
		CHECK_MSG(in && ref_ok && fired > 0 && good == want && without != want && k.z_down == 0,
		          "7.S4b premise, %s (width %zu): inside %d, fires %lld times, replica == v1.9.0 %d, skipped differs %d, "
		          "z down %lld", c.label, w, in ? 1 : 0, fired, good == want ? 1 : 0, without != want ? 1 : 0, k.z_down);
		++corrections[label.substr(0, label.find(" (steered"))];
	});
	CHECK_MSG(fast > 0 && fallback > 0 && aliased > 0, "4.S4: %zu fast, %zu fallback, %zu aliased calls", fast, fallback,
	          aliased);
	for (const char* want : {"4.S4 correction row: z up", "4.S4 correction row: p up", "4.S4 correction row: p down"})
		CHECK_MSG(corrections[want] >= 16, "4.S4: %d rows for \"%s\", want at least 16", corrections[want], want);
	// Width 0: true, nothing written, no counter (§3.6 counts calls with width >= 1).
	{
		int64_t sentinel = superslm_attention_cases::kSmPoison;
		const SmCounters c0 = ReadSmCounters();
		const bool ok = superslm::SoftmaxRowQ15(&sentinel, 0, 636211, 1272422, 848665286933, &sentinel);
		const SmCounters c1 = ReadSmCounters();
		CHECK_MSG(ok && sentinel == superslm_attention_cases::kSmPoison, "4.S4 width 0: bool %d, sentinel %s", ok ? 1 : 0,
		          sentinel == superslm_attention_cases::kSmPoison ? "kept" : "written");
		CheckSmDelta("4.S4 width 0", SmDelta(c0, c1), SmCounters{});
	}
	// The named inside corners, asserted against the guard copy's own verdict (a guard copy that drifted
	// from §5.4 cannot quietly move a corner to the other side).
	const int64_t zero = 0;
	const struct {
		int64_t q_ln2, q_b, q_c;
		size_t width;
		int64_t score;
		bool fast;
		const char* what;
	} corners[] = {
	    {1, 0, INT64_C(1) << 47, size_t{1} << 14, 5, true, "q_ln2 = 1 with M = 2^47 at width 2^14 is inside"},
	    {1, 0, (INT64_C(1) << 47) + 1, 1, 0, false, "M = 2^47 + 1 is outside"},
	    {3, 1, 0, 1, 0, true, "M = 1 is inside"},
	    {1, 0, 0, 1, 0, false, "M = 0 is outside"},
	    {9, 4, 0, 1, 0, true, "q_c = 0 is inside"},
	    {9, 4, -1, 1, 0, false, "q_c = -1 is outside"},
	    {9, 4, 0, 1, INT64_C(1) << 61, true, "a score of 2^61 is inside"},
	    {9, 4, 0, 1, -(INT64_C(1) << 61), true, "a score of -2^61 is inside"},
	    {9, 4, 0, 1, (INT64_C(1) << 61) + 1, false, "a score of 2^61 + 1 is outside"},
	    {10, 4, 0, 1, 0, false, "q_ln2 = 2 q_b + 2 is outside"},
	    {9, 4, 0, (size_t{1} << 14) + 1, 0, false, "width 2^14 + 1 is outside"},
	};
	for (const auto& k : corners) {
		std::vector<int64_t> row(k.width, 0);
		row[0] = k.score;
		CHECK_MSG(TestSoftmaxGuard(row.data(), row.size(), k.q_ln2, k.q_b, k.q_c).Fast() == k.fast, "7.S4a guard copy: %s",
		          k.what);
	}
	(void)zero;
}

// ---- 2.S4 and 7.S4c: hostile rows, each failing exactly one conjunct ----------------------------------

void TestS4HostileRows() {
	size_t rows = 0;
	superslm_attention_cases::ForEachSoftmaxCase([&](const SmCase& c) {
		if (std::strncmp(c.label, "2.S4", 4) != 0) return;
		++rows;
		const size_t w = c.scores.size();
		const SoftmaxGuard g = TestSoftmaxGuard(c.scores.data(), w, c.q_ln2, c.q_b, c.q_c);
		const bool witness = std::strstr(c.label, "witness") != nullptr;
		// RunSm (in TestS4Grid) already asserted bool, output and fallback +1.
		if (witness)
			CHECK_MSG(g.Failing() >= 2, "%s: fails %d conjuncts, want more than one", c.label, g.Failing());
		else
			CHECK_MSG(g.Failing() == 1, "%s: fails %d conjuncts of the guard copy, want exactly 1", c.label, g.Failing());
		// Which rows the v1.9.0 body refuses (the bool is the signal there) and which it accepts (the
		// counter is then the only signal): the constant rows refuse; the width and score rows are
		// output-equivalent (§8 2.S4).
		std::vector<int64_t> want(w);
		const bool ref_ok = RefSoftmax(c.scores.data(), w, c.q_ln2, c.q_b, c.q_c, want.data());
		const bool output_equivalent = !g.width_ok || !g.scores_ok;
		CHECK_MSG(ref_ok == output_equivalent, "%s: v1.9.0 bool %d, want %d", c.label, ref_ok ? 1 : 0,
		          output_equivalent ? 1 : 0);
	});
	CHECK_MSG(rows == 10, "2.S4: %zu hostile rows in the set, want 10", rows);
	// The witness restated in the set is the fixture's own.
	const auto& w = superslm_test::kSoftmaxRowOffRatioWitness;
	bool same = false;
	superslm_attention_cases::ForEachSoftmaxCase([&](const SmCase& c) {
		if (std::strstr(c.label, "witness") == nullptr) return;
		same = c.q_ln2 == w.q_ln2 && c.q_b == w.q_b && c.q_c == w.q_c && c.scores.size() == w.width &&
		       c.scores[0] == w.scores[0] && c.scores[1] == w.scores[1] && c.scores[2] == w.scores[2];
	});
	CHECK_MSG(same, "2.S4: the set's off-ratio witness equals kSoftmaxRowOffRatioWitness");
}

// ---- the replica's own premise: equal to the v1.9.0 body on realistic rows inside the guard ------------

void TestS4ReplicaPremise() {
	superslm_attention_cases::Rng rng(0x5334524550524D53ULL);  // "S4REPRMS"
	const auto triples = superslm_attention_cases::SmRealisticConstants(rng, 4);
	size_t rows = 0, bad = 0, z_down = 0;
	for (int it = 0; it < 4000; ++it) {
		const auto& t = triples[static_cast<size_t>(rng.Next() % triples.size())];
		const size_t w = static_cast<size_t>(rng.InRange(1, 200));
		const std::vector<int64_t> s = superslm_attention_cases::SmScoreRow(w, it % 4, t[0], rng);
		std::vector<int64_t> a(w), b(w);
		SoftmaxCorrections k;
		if (!SoftmaxEstimateReplica(s.data(), w, t[0], t[1], t[2], a.data(), &k)) continue;
		++rows;
		z_down += static_cast<size_t>(k.z_down);
		if (!RefSoftmax(s.data(), w, t[0], t[1], t[2], b.data()) || a != b) ++bad;
	}
	CHECK_MSG(rows > 1000 && bad == 0 && z_down == 0,
	          "4.S4 replica premise: %zu rows inside the guard, %zu differ from v1.9.0, z down fired %zu times", rows,
	          bad, z_down);
}

// ---- 6.3: the S4 golden pin (the v1.9.0 tag's hash over the softmax input set) ------------------------

void TestS4GoldenPin() {
	superslm::Sha256 h;
	uint64_t values = 0;
	auto emit = [&](int64_t v) {
		uint8_t b[8];
		for (int i = 0; i < 8; ++i) b[i] = static_cast<uint8_t>((static_cast<uint64_t>(v) >> (8 * i)) & 0xffU);
		h.Update(b, 8);
		++values;
	};
	superslm_attention_cases::RunSoftmaxCases(emit);
	uint8_t digest[32];
	h.Final(digest);
	const std::string hex = superslm::ToHex(digest);
	std::printf("attn-rowsites S4 golden hash: %s (%llu values)\n", hex.c_str(),
	            static_cast<unsigned long long>(values));
	CHECK_MSG(hex == std::string(superslm_test::kAttnRowsiteS4GoldenHash) &&
	              values == superslm_test::kAttnRowsiteS4GoldenValues,
	          "6.3 S4 golden: %s over %llu values, pin %s over %llu (v1.9.0 tag)", hex.c_str(),
	          static_cast<unsigned long long>(values), superslm_test::kAttnRowsiteS4GoldenHash,
	          static_cast<unsigned long long>(superslm_test::kAttnRowsiteS4GoldenValues));
}

// ==== Slice S5: the Q31 score in three 16-bit pieces, per head (§4.5, §5.5) ===========================
//
// Every S5 kernel cell calls QkQ31ScoreRow directly and asserts per call (§8 path rule): every score equals
// the reference, and the q31_row path counters moved by exactly the delta a test-side copy of the guard names
// (head_dim <= 512 and every ratio in [0, 2^32), written from §4.5), on the selected kernel's own tier only.
// The reference is the in-tree scalar reference QkQ31ScoreScalarRef for in-contract ratios, and the SAME
// binary's per-key QkQ31Score for the out-of-contract rows of 2.S5 (§3.3: outside [0, 2^32) the v1.9.0 SIMD
// tiers already differ from the scalar reference, and the fallback is that per-key loop). Cell 11.1(c) drives
// the Q31 call sites of both layer loops on the widened QK-norm fixture.

using superslm_attention_cases::Q31Case;

// The test-side copy of the S5 guard (§4.5), one flag per conjunct so a 2.S5 row can be shown to fail
// exactly one.
struct Q31Guard {
	bool hd_ok = true, ratio_ge0 = true, ratio_lt = true;
	bool Fast() const { return hd_ok && ratio_ge0 && ratio_lt; }
	int Failing() const { return !hd_ok + !ratio_ge0 + !ratio_lt; }
};

Q31Guard TestQ31Guard(const int64_t* ratio, size_t head_dim) {
	Q31Guard g;
	g.hd_ok = head_dim <= 512;
	for (size_t d = 0; d < head_dim; ++d) {
		if (ratio[d] < 0) g.ratio_ge0 = false;
		if (ratio[d] > INT64_C(0xFFFFFFFF)) g.ratio_lt = false;
	}
	return g;
}

struct Q3Counters {
	long long fast2 = 0, fb2 = 0, fast5 = 0, fb5 = 0;
};
#if defined(SUPERSLM_ENABLE_MATMUL_DISPATCH_INSTRUMENT) && SUPERSLM_MATMUL_HAVE_SIMD_X64
constexpr bool kHaveQ3Counters = true;
Q3Counters ReadQ3Counters() {
	return Q3Counters{superslm_test::g_q31_row_fast_avx2.load(), superslm_test::g_q31_row_fallback_avx2.load(),
	                  superslm_test::g_q31_row_fast_avx512.load(), superslm_test::g_q31_row_fallback_avx512.load()};
}
#else
constexpr bool kHaveQ3Counters = false;
Q3Counters ReadQ3Counters() { return Q3Counters{}; }
#endif

Q3Counters Q3Delta(const Q3Counters& a, const Q3Counters& b) {
	return Q3Counters{b.fast2 - a.fast2, b.fb2 - a.fb2, b.fast5 - a.fast5, b.fb5 - a.fb5};
}

// `fast` fast and `fallback` fallback calls on this build (11.1(b)).
Q3Counters ExpectedQ3Delta(long long fast, long long fallback) {
	Q3Counters w;
	switch (ExpectedSitesKernel()) {
		case SitesKernel::kAvx2: w.fast2 = fast; w.fb2 = fallback; break;
		case SitesKernel::kAvx512: w.fast5 = fast; w.fb5 = fallback; break;
		case SitesKernel::kShipped: break;
	}
	return w;
}

bool CheckQ3Delta(const char* label, const Q3Counters& d, const Q3Counters& want) {
	if (!kHaveQ3Counters) return true;
	const bool ok = d.fast2 == want.fast2 && d.fb2 == want.fb2 && d.fast5 == want.fast5 && d.fb5 == want.fb5;
	CHECK_MSG(ok, "%s: q31_row_fast_avx2 +%lld q31_row_fallback_avx2 +%lld q31_row_fast_avx512 +%lld "
	          "q31_row_fallback_avx512 +%lld; want +%lld/+%lld/+%lld/+%lld (kernel: %s)",
	          label, d.fast2, d.fb2, d.fast5, d.fb5, want.fast2, want.fb2, want.fast5, want.fb5,
	          SitesKernelName(ExpectedSitesKernel()));
	return ok;
}

enum class Q31Ref { kScalarRef, kSameBinaryPerKey };

// One call against its reference, with its path assertion. The output is an exact-size heap vector,
// poisoned, so the hosted ASan leg sees any write past the row. Returns whether the guard copy expected
// the fast path.
bool RunQ31(const Q31Case& c, Q31Ref ref) {
	std::vector<int64_t> want(c.width);
	for (size_t j = 0; j < c.width; ++j) {
		const int8_t* k = c.keys.data() + j * c.head_dim;
		want[j] = ref == Q31Ref::kScalarRef
		              ? superslm::QkQ31ScoreScalarRef(c.q.data(), k, c.ratio.data(), c.head_dim)
		              : superslm::QkQ31Score(c.q.data(), k, c.ratio.data(), c.head_dim);
	}
	std::vector<int64_t> out(c.width, superslm_attention_cases::kQ31Poison);
	const Q31Guard g = TestQ31Guard(c.ratio.data(), c.head_dim);
	const Q3Counters c0 = ReadQ3Counters();
	superslm::QkQ31ScoreRow(c.q.data(), c.keys.data(), c.ratio.data(), c.head_dim, c.width, out.data());
	const Q3Counters c1 = ReadQ3Counters();
	size_t bad = 0, first = c.width;
	for (size_t j = 0; j < c.width; ++j)
		if (out[j] != want[j]) {
			if (first == c.width) first = j;
			++bad;
		}
	CHECK_MSG(bad == 0, "%s (head_dim %zu, width %zu): %zu of %zu scores differ from %s, first at key %zu (%lld vs %lld)",
	          c.label, c.head_dim, c.width, bad, c.width,
	          ref == Q31Ref::kScalarRef ? "QkQ31ScoreScalarRef" : "this binary's per-key QkQ31Score", first,
	          first < c.width ? static_cast<long long>(out[first]) : 0LL,
	          first < c.width ? static_cast<long long>(want[first]) : 0LL);
	char full[200];
	std::snprintf(full, sizeof full, "%s (head_dim %zu, width %zu, guard copy: %s)", c.label, c.head_dim, c.width,
	              g.Fast() ? "fast" : "fallback");
	CheckQ3Delta(full, Q3Delta(c0, c1), ExpectedQ3Delta(g.Fast() ? 1 : 0, g.Fast() ? 0 : 1));
	return g.Fast();
}

// §5.5's limbs, restated: w = a2 * 2^30 + a1 * 2^15 + a0, a0 = w & 0x7FFF, a1 = (w >> 15) & 0x7FFF,
// a2 = w >> 30 (arithmetic). Returns a key's three limb sums in int64 (no wrap), for the premise cells.
void LimbSums(const Q31Case& c, size_t key, int64_t sums[3]) {
	sums[0] = sums[1] = sums[2] = 0;
	for (size_t d = 0; d < c.head_dim; ++d) {
		const int64_t w = static_cast<int64_t>(c.q[d]) * c.ratio[d];
		const int64_t k = c.keys[key * c.head_dim + d];
		sums[0] += k * (w & 0x7FFF);
		sums[1] += k * ((w >> 15) & 0x7FFF);
		sums[2] += k * (w >> 30);
	}
}

// ---- 4.S5, 6.1, 7.S5a, 7.S5b, 7.S5c: the shape grid, the margin corners and the ties --------------------

void TestS5Grid() {
	size_t rows = 0, fast = 0, margin = 0, ties = 0;
	superslm_attention_cases::ForEachQ31Case([&](const Q31Case& c) {
		++rows;
		const bool f = RunQ31(c, Q31Ref::kScalarRef);
		fast += f ? 1 : 0;
		CHECK_MSG(f == (c.head_dim <= 512), "%s (head_dim %zu): guard copy says %s, want fast exactly when head_dim "
		          "<= 512 (every ratio of the set is in [1, 2^31])", c.label, c.head_dim, f ? "fast" : "fallback");
		if (std::strncmp(c.label, "7.S5b", 5) == 0) {
			// The premise: every limb sum of every key is the margin row §5.5 names.
			++margin;
			int64_t s[3];
			LimbSums(c, 0, s);
			const int64_t per = static_cast<int64_t>(c.keys[0]) * 32767 * static_cast<int64_t>(c.head_dim);
			CHECK_MSG(s[0] == per && s[1] == per, "%s head_dim %zu: limb sums %lld, %lld, want %lld", c.label,
			          c.head_dim, static_cast<long long>(s[0]), static_cast<long long>(s[1]),
			          static_cast<long long>(per));
			if (c.head_dim == 512 && c.keys[0] == -128)
				CHECK_MSG(per == INT64_C(-2147418112) && per - INT32_MIN == 65536,
				          "7.S5b: the corner's limb sum %lld is not 65,536 above int32's minimum", static_cast<long long>(per));
			if (c.head_dim == 516 && c.keys[0] == -128)
				CHECK_MSG(per < INT32_MIN, "7.S5b: at head_dim 516 the limb sum %lld should pass int32's minimum",
				          static_cast<long long>(per));
		}
		if (std::strncmp(c.label, "7.S5c", 5) == 0) {
			for (size_t j = 0; j < c.width; ++j) {
				int64_t total = 0;
				for (size_t d = 0; d < c.head_dim; ++d)
					total += static_cast<int64_t>(c.q[d]) * c.keys[j * c.head_dim + d] * c.ratio[d];
				const int64_t r = total & ((INT64_C(1) << 31) - 1);
				if (r == (INT64_C(1) << 30)) ++ties;
			}
		}
	});
	CHECK_MSG(rows == 10 * 8 * 3 + 16 + 2, "4.S5: %zu rows in the set, want %d", rows, 10 * 8 * 3 + 16 + 2);
	CHECK_MSG(fast == 8 * 8 * 3 + 8 + 2, "4.S5: %zu rows on the fast side of the guard copy, want %d", fast,
	          8 * 8 * 3 + 8 + 2);
	CHECK_MSG(margin == 16, "7.S5b: %zu margin rows, want 16", margin);
	CHECK_MSG(ties == 16, "7.S5c: %zu keys whose total is a rounding tie (+-2^30 mod 2^31), want 16", ties);
}

// ---- 7.S5d and the guard's inside corners: ratios 0, 2^31 and 2^32 - 1 run fast --------------------------

void TestS5GuardInsideRows() {
	superslm_attention_cases::Rng rng(0x5335494E53494445ULL);  // "S5INSIDE"
	for (size_t hd : {size_t{64}, size_t{128}, size_t{512}})
		for (size_t w : {size_t{9}, size_t{17}})
			for (int64_t r : {INT64_C(0), INT64_C(1) << 31, INT64_C(0xFFFFFFFF)}) {
				Q31Case c = superslm_attention_cases::MakeQ31GridCase(hd, w, 1, rng);
				c.label = "7.S5d inside corner";
				for (size_t d = 0; d < hd; d += 2) c.ratio[d] = r;  // every other channel at the corner
				const bool f = RunQ31(c, Q31Ref::kScalarRef);
				CHECK_MSG(f, "7.S5d: ratio %lld at head_dim %zu should be inside the guard copy", static_cast<long long>(r),
				          hd);
			}
	// Width 0: nothing is written; the call still counts once on its tier (§3.6: once per call).
	Q31Case z = superslm_attention_cases::MakeQ31GridCase(64, 1, 0, rng);
	z.label = "4.S5 width 0";
	z.width = 0;
	z.keys.clear();
	RunQ31(z, Q31Ref::kScalarRef);
}

// ---- 2.S5: hostile rows, each failing exactly one conjunct -----------------------------------------------

void TestS5HostileRows() {
	superslm_attention_cases::Rng rng(0x3253354853544C45ULL);  // "2S5HSTLE"
	size_t rows = 0;
	// Ratio -1 and 2^32 break a kernel that forms w from 32-bit pieces; 2^33 and 2^38 are outside the guard but
	// inside a 64-bit formation's int16 range (§5.5); 2^48 breaks every formation while 128 * 128 * 2^48 = 2^62
	// keeps the scalar reference's int64 sum defined. INT64_MAX is not a row (§5.5). Each at the first, a middle
	// and the last channel (the last is the SIMD tiers' scalar tail at head_dim 63).
	const int64_t hostile[] = {INT64_C(-1), INT64_C(1) << 32, INT64_C(1) << 33, INT64_C(1) << 38, INT64_C(1) << 48};
	for (size_t hd : {size_t{63}, size_t{64}})
		for (int64_t r : hostile)
			for (size_t at : {size_t{0}, hd / 2, hd - 1}) {
				Q31Case c = superslm_attention_cases::MakeQ31GridCase(hd, 17, 0, rng);
				c.label = "2.S5 ratio outside [0, 2^32)";
				c.ratio[at] = r;
				const Q31Guard g = TestQ31Guard(c.ratio.data(), c.head_dim);
				CHECK_MSG(g.Failing() == 1 && (r < 0 ? !g.ratio_ge0 : !g.ratio_lt),
				          "2.S5 ratio %lld: fails %d conjuncts of the guard copy, want exactly the ratio one",
				          static_cast<long long>(r), g.Failing());
				RunQ31(c, Q31Ref::kSameBinaryPerKey);
				++rows;
			}
	for (size_t w : {size_t{1}, size_t{17}}) {
		Q31Case c = superslm_attention_cases::MakeQ31GridCase(513, w, 0, rng);
		c.label = "2.S5 head_dim 513";
		const Q31Guard g = TestQ31Guard(c.ratio.data(), c.head_dim);
		CHECK_MSG(g.Failing() == 1 && !g.hd_ok, "2.S5 head_dim 513: fails %d conjuncts, want exactly head_dim",
		          g.Failing());
		RunQ31(c, Q31Ref::kSameBinaryPerKey);
		++rows;
	}
	CHECK_MSG(rows == 2 * 5 * 3 + 2, "2.S5: %zu hostile rows, want %d", rows, 2 * 5 * 3 + 2);
}

// ---- 6.3: the S5 golden pin (the v1.9.0 tag's hashes over the Q31 set and the QK-norm fixture) ----------

template <class Run>
std::string HashRun(Run run, uint64_t* values) {
	superslm::Sha256 h;
	*values = 0;
	auto emit = [&](int64_t v) {
		uint8_t b[8];
		for (int i = 0; i < 8; ++i) b[i] = static_cast<uint8_t>((static_cast<uint64_t>(v) >> (8 * i)) & 0xffU);
		h.Update(b, 8);
		++*values;
	};
	run(emit);
	uint8_t digest[32];
	h.Final(digest);
	return superslm::ToHex(digest);
}

void TestS5GoldenPin() {
	uint64_t values = 0;
	const std::string hex = HashRun(
	    [](auto& emit) { superslm_attention_cases::RunQ31Cases(emit, superslm::QkQ31ScoreRow); }, &values);
	std::printf("attn-rowsites S5 golden hash: %s (%llu values)\n", hex.c_str(), static_cast<unsigned long long>(values));
	CHECK_MSG(hex == std::string(superslm_test::kAttnRowsiteS5GoldenHash) &&
	              values == superslm_test::kAttnRowsiteS5GoldenValues,
	          "6.3 S5 golden: %s over %llu values, pin %s over %llu (v1.9.0 tag)", hex.c_str(),
	          static_cast<unsigned long long>(values), superslm_test::kAttnRowsiteS5GoldenHash,
	          static_cast<unsigned long long>(superslm_test::kAttnRowsiteS5GoldenValues));
}

// ---- 11.1(c): the QK-norm attention fixture, the Q31 call sites of both loops ----------------------------

// Run (iii)'s sink: every row the decode loop hands it, classified by the test-side guard copies (§5.4's
// softmax guard, §5.2's int16 condition), so the softmax and prob·V counters are exact too.
struct FixtureRows {
	long long observes = 0, sm_fast = 0, sm_fallback = 0, pv_fast = 0, pv_fallback = 0;
};

void ClassifyRow(void* ctx, uint32_t, int64_t, size_t, const int64_t* scores, size_t width, int64_t q_ln2, int64_t q_b,
                 int64_t q_c, const int64_t* probs, const int64_t*, const int64_t*, size_t head_dim) {
	auto* r = static_cast<FixtureRows*>(ctx);
	++r->observes;
	(TestSoftmaxGuard(scores, width, q_ln2, q_b, q_c).Fast() ? r->sm_fast : r->sm_fallback) += 1;
	(TestPvGuard(probs, width, head_dim).Fast() ? r->pv_fast : r->pv_fallback) += 1;
}

struct AllCounters {
	RowCounters row;
	Q3Counters q3;
	SmCounters sm;
	PvCounters pv;
	RqCounters rq;
};

AllCounters ReadAll() {
	return AllCounters{ReadRowCounters(), ReadQ3Counters(), ReadSmCounters(), ReadPvCounters(), ReadRqCounters()};
}

// The per-position structural deltas of §8 11.1(c)'s table, "S5 landed, S6 not" column, over `n`
// positions: q31_row fast L·H = 4 per position and fallback 0; rowtable_norm_skipped 2L + L·H = 6 (two
// hidden norms at 256, four q_norm at 64); silu 1 and landing 2 skipped; every taken 0.
void CheckFixtureStructural(const char* label, const AllCounters& a, const AllCounters& b, long long n) {
	CheckQ3Delta(label, Q3Delta(a.q3, b.q3), ExpectedQ3Delta(4 * n, 0));
	if (!kHaveRowCounters) return;
	const RowCounters d = Delta(a.row, b.row);
	const long long want_skipped[3] = {6 * n, n, 2 * n};
	for (int s = 0; s < 3; ++s)
		CHECK_MSG(d.taken[s] == 0 && d.skipped[s] == want_skipped[s],
		          "%s: rowtable_%s taken +%lld skipped +%lld, want +0/+%lld", label, kRowSiteNames[s], d.taken[s],
		          d.skipped[s], want_skipped[s]);
}

void TestS5QkNormFixture() {
	using superslm::SslmForwardStatus;
	namespace fx = superslm_qk_fixture;
	fx::QkAttentionFixture f;
	CHECK_MSG(f.loaded, "11.1(c): the fixture's own minimal artifact failed to load: %s", f.load_error.c_str());
	if (!f.loaded) return;
	const long long P = static_cast<long long>(fx::kPositions);

	// Run (iii) first: the decode loop with the classifying sink. Its rows give the data terms.
	FixtureRows rows;
	superslm::AttentionCaptureSink sink{&rows, &ClassifyRow};
	AllCounters at = ReadAll();
	const AllCounters start3 = at;
	uint64_t n3 = 0;
	SslmForwardStatus st3 = SslmForwardStatus::Ok;
	const std::string h3 = HashRun(
	    [&](auto& emit) {
		    st3 = fx::RunFixtureDecode(f, emit, &sink, [&](size_t p) {
			    const AllCounters now = ReadAll();
			    char label[64];
			    std::snprintf(label, sizeof label, "11.1(c) run (iii) position %zu", p);
			    CheckFixtureStructural(label, at, now, 1);
			    at = now;
		    });
	    },
	    &n3);
	const AllCounters end3 = ReadAll();
	CHECK_MSG(st3 == SslmForwardStatus::Ok, "11.1(c) run (iii): status %s", SslmForwardStatusName(st3));
	// The premise (checked on the base before any kernel landed): one observe per (position, head), every
	// softmax row inside §5.4's guard; the only rows failing the int16 condition are position 0's width-1 rows.
	CHECK_MSG(rows.observes == 4 * P && rows.sm_fallback == 0 && rows.pv_fallback == 4,
	          "11.1(c) premise: %lld rows observed (want %lld), %lld outside the softmax guard (want 0), %lld failing "
	          "the int16 condition (want 4)", rows.observes, 4 * P, rows.sm_fallback, rows.pv_fallback);

	// Run (i): the decode loop, no sink.
	at = ReadAll();
	const AllCounters start1 = at;
	uint64_t n1 = 0;
	SslmForwardStatus st1 = SslmForwardStatus::Ok;
	const std::string h1 = HashRun(
	    [&](auto& emit) {
		    st1 = fx::RunFixtureDecode(f, emit, nullptr, [&](size_t p) {
			    const AllCounters now = ReadAll();
			    char label[64];
			    std::snprintf(label, sizeof label, "11.1(c) run (i) position %zu", p);
			    CheckFixtureStructural(label, at, now, 1);
			    at = now;
		    });
	    },
	    &n1);
	const AllCounters end1 = ReadAll();
	CHECK_MSG(st1 == SslmForwardStatus::Ok, "11.1(c) run (i): status %s", SslmForwardStatusName(st1));

	// Run (ii): the chunk loop over all 24 positions, with a counting sink installed on the layer (t2701's
	// chunk-mode configuration): the chunk loop never observes.
	FixtureRows chunk_rows;
	superslm::AttentionCaptureSink chunk_sink{&chunk_rows, &ClassifyRow};
	const AllCounters start2 = ReadAll();
	uint64_t n2 = 0;
	SslmForwardStatus st2 = SslmForwardStatus::Ok;
	const std::string h2 = HashRun([&](auto& emit) { st2 = fx::RunFixtureChunk(f, emit, &chunk_sink); }, &n2);
	const AllCounters end2 = ReadAll();
	CHECK_MSG(st2 == SslmForwardStatus::Ok, "11.1(c) run (ii): status %s", SslmForwardStatusName(st2));
	CHECK_MSG(chunk_rows.observes == 0, "11.1(c) run (ii): the chunk loop made %lld observe calls, want 0",
	          chunk_rows.observes);
	CheckFixtureStructural("11.1(c) run (ii), 24 positions", start2, end2, P);

	// Every run hashes to the golden pin's fixture hash (6.3, from the v1.9.0 tag).
	const struct {
		const char* name;
		const std::string& hex;
		uint64_t n;
		const AllCounters &a, &b;
	} runs[] = {{"(i)", h1, n1, start1, end1}, {"(ii)", h2, n2, start2, end2}, {"(iii)", h3, n3, start3, end3}};
	for (const auto& r : runs) {
		CHECK_MSG(r.hex == std::string(superslm_test::kAttnRowsiteS5FixtureGoldenHash) &&
		              r.n == superslm_test::kAttnRowsiteS5FixtureGoldenValues,
		          "11.1(c)/6.3 run %s: fixture hash %s over %llu values, pin %s over %llu (v1.9.0 tag)", r.name,
		          r.hex.c_str(), static_cast<unsigned long long>(r.n), superslm_test::kAttnRowsiteS5FixtureGoldenHash,
		          static_cast<unsigned long long>(superslm_test::kAttnRowsiteS5FixtureGoldenValues));
		// Softmax and prob·V: exact, from run (iii)'s classified rows, on every run.
		char label[64];
		std::snprintf(label, sizeof label, "11.1(c) run %s softmax", r.name);
		CheckSmDelta(label, SmDelta(r.a.sm, r.b.sm), ExpectedSmDelta(rows.sm_fast, rows.sm_fallback));
		std::snprintf(label, sizeof label, "11.1(c) run %s prob-V", r.name);
		CheckPvDelta(label, PvDelta(r.a.pv, r.b.pv), ExpectedPvDelta(rows.pv_fast, rows.pv_fallback));
	}
	std::printf("attn-rowsites 11.1(c) fixture hash: %s (%llu values); %lld rows, softmax %lld/%lld, prob-V %lld/%lld "
	            "(fast/fallback by the guard copies)\n", h1.c_str(), static_cast<unsigned long long>(n1), rows.observes,
	            rows.sm_fast, rows.sm_fallback, rows.pv_fast, rows.pv_fallback);
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
	const RqCounters q0 = ReadRqCounters();
	const SmCounters m0 = ReadSmCounters();
	const Q3Counters x0 = ReadQ3Counters();
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
	const RqCounters q1 = ReadRqCounters();
	const SmCounters m1 = ReadSmCounters();
	const Q3Counters x1 = ReadQ3Counters();
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
	const RqCounters q2 = ReadRqCounters();
	const SmCounters m2 = ReadSmCounters();
	const Q3Counters x2 = ReadQ3Counters();
	const TraceCount tp2 = trace;
	SslmSetTraceHook(model.trace_hook, nullptr, nullptr);

	static const char* const kSites[] = {"embed",          "q_proj.requant",   "attn_ctx",        "o_proj.requant",
	                                     "attn_norm",      "attn_residual",    "mlp_norm",        "gate_proj.requant",
	                                     "up_proj.requant", "mlp_act",         "down_proj.requant", "mlp_residual"};
	// The prob·V data terms (§8 11.1(d)): rows failing the int16 condition on this artifact and these
	// pinned tokens, measured on the base (docs/attention-rowsites/s2/pv-data-terms.txt): 15 in the prefill
	// window (position 0's 14 width-1 rows and position 2, head 13's {0, 32,768, 0}), 0 in decode.
	// The softmax data term: rows outside §5.4's guard, measured on the base (the S3 head) the same way
	// (docs/attention-rowsites/s4/softmax-data-terms.txt): 0 in both windows.
	const struct {
		const char* name;
		RowCounters a, b;
		PvCounters va, vb;
		RqCounters qa, qb;
		SmCounters ma, mb;
		Q3Counters xa, xb;
		TraceCount ta, tb;
		size_t N;
		long long pv_fallback, softmax_fallback;
	} windows[] = {{"prefill", p0, p1, v0, v1, q0, q1, m0, m1, x0, x1, tp0, tp1, T, 15, 0},
	               {"decode", p1, p2, v1, v2, q1, q2, m1, m2, x1, x2, tp1, tp2, D, 0, 0}};
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
		// S3: one row-leaf call per funnel call that passes its preflight, (11·L + 1)·N (structural: the
		// same count as the chain trace records above), on the active kernel's tier only (11.1(b)); 0 on
		// forced SSE2 and on an MSVC build's AVX-512 tier with its switch off, where the records still count.
		std::snprintf(pv_label, sizeof pv_label, "11.1(d) %s requant_row", w.name);
		CheckRqDelta(pv_label, RqDelta(w.qa, w.qb), ExpectedRqDelta((11 * Ll + 1) * N));
		// S4: one SoftmaxRowQ15 call per (layer, head, token pass), L·H·N (structural), of which the data term
		// falls back; on the active kernel's tier only (11.1(b)).
		std::snprintf(pv_label, sizeof pv_label, "11.1(d) %s softmax", w.name);
		CheckSmDelta(pv_label, SmDelta(w.ma, w.mb), ExpectedSmDelta(calls - w.softmax_fallback, w.softmax_fallback));
		// S5: this artifact carries no QK-norm, so the plain score path runs and QkQ31ScoreRow is never called.
		std::snprintf(pv_label, sizeof pv_label, "11.1(d) %s q31_row", w.name);
		CheckQ3Delta(pv_label, Q3Delta(w.xa, w.xb), Q3Counters{});
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
	TestS3CornerPremise();
	TestS3Grid();
	TestS3FunnelCallSite();
	TestS3GoldenPin();
	TestS4ReplicaPremise();
	TestS4Grid();
	TestS4HostileRows();
	TestS4GoldenPin();
	TestS5Grid();
	TestS5GuardInsideRows();
	TestS5HostileRows();
	TestS5GoldenPin();
	TestS5QkNormFixture();     // 11.1(c): the Q31 call sites of both loops
	TestS1LayerLoopWindows();  // 11.1(d): S1's, S2's, S3's, S4's and S5's rows over one drive
	std::printf("attn-rowsites cells (plan slices S1, S2, S3, S4, S5): %d checks, %d failures\n", GChecks, GFailures);
	checks += GChecks;
	failures += GFailures;
}
