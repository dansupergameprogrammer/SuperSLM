// Attention and per-row sites plan, slice S1 (the per-row tables): the fixed input set that the
// digest section `c_rowsites`, the golden-pin generator (tools/gen_attn_rowsite_golden.cpp) and the
// suite's golden cell all run through the three per-row sites. Header-only, so it needs no build
// entry: the digest and the generator include it by relative path, the suite through `tests/`.
//
// Everything here calls only entry points whose signatures are unchanged since v1.9.0
// (RmsNormSite, MlpActSite, ResidualReconcileSite), so the generator can be built against the
// v1.9.0 tag's library and the pin takes no input from the code it grades (plan §3.3 evidence 3).
//
// The set covers both sides of the table threshold (widths 1 to 4,864 around 512), the -128 code
// at the first, middle and last element, uniform rows, a gate scale small enough that
// sig(-128) differs from sig(-127) and from 0 (plan §8 4.S1), residual rows whose first candidate
// is refused mid-row by the landing flag, and rows the funnel refuses. Each call contributes its
// status, its output scale and its whole output row (poisoned before the call, so a refused call's
// untouched row is digested too).
#ifndef SUPERSLM_TESTS_SUPPORT_ROWSITE_CASES_H
#define SUPERSLM_TESTS_SUPPORT_ROWSITE_CASES_H

#include <cstddef>
#include <cstdint>
#include <vector>

#include "superslm/checked_chain_funnel.h"
#include "superslm/forward_sites.h"
#include "superslm/silu_lut_canonical.h"

namespace superslm_rowsite_cases {

// splitmix64, integer-only: the same values on every conforming C++20 implementation.
struct Rng {
	uint64_t s;
	explicit Rng(uint64_t seed) : s(seed) {}
	uint64_t Next() {
		s += 0x9e3779b97f4a7c15ULL;
		uint64_t z = s;
		z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
		z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
		return z ^ (z >> 31);
	}
	int64_t InRange(int64_t lo, int64_t hi) {
		const uint64_t span = static_cast<uint64_t>(hi - lo) + 1ULL;
		return lo + static_cast<int64_t>(Next() % span);
	}
};

// The widths of plan §8 4.S1: both sides of kRowTableMinWidth (512) and the 0.5B's two widths.
inline constexpr size_t kWidths[] = {1, 64, 255, 511, 512, 513, 896, 4864};

// Row shapes: random codes over the full int8 range with -128 placed at the first, middle and last
// element, random codes in [-127, 127] with no -128, all -127, all 127, all -128.
enum class RowShape : int { kRandomWithMinus128 = 0, kRandomNo128, kAllMinus127, kAll127, kAllMinus128 };
inline constexpr RowShape kShapes[] = {RowShape::kRandomWithMinus128, RowShape::kRandomNo128,
                                       RowShape::kAllMinus127, RowShape::kAll127, RowShape::kAllMinus128};

inline void FillRow(std::vector<int8_t>& row, RowShape shape, Rng& rng) {
	const size_t n = row.size();
	for (size_t i = 0; i < n; ++i) {
		switch (shape) {
		case RowShape::kRandomWithMinus128: row[i] = static_cast<int8_t>(rng.InRange(-128, 127)); break;
		case RowShape::kRandomNo128: row[i] = static_cast<int8_t>(rng.InRange(-127, 127)); break;
		case RowShape::kAllMinus127: row[i] = -127; break;
		case RowShape::kAll127: row[i] = 127; break;
		case RowShape::kAllMinus128: row[i] = -128; break;
		}
	}
	if (shape == RowShape::kRandomWithMinus128 && n > 0) {
		row[0] = -128;
		row[n / 2] = -128;
		row[n - 1] = -128;
	}
}

// One site call's observable result, emitted as int64 values through `emit`: status, scale (m, e),
// then every output code. `out` was poisoned with 0x5A before the call.
template <class Emit>
void EmitResult(Emit& emit, superslm::SslmForwardStatus status, const superslm::CarriedScale& scale,
                const std::vector<int8_t>& out) {
	emit(static_cast<int64_t>(status));
	emit(scale.m);
	emit(scale.e);
	for (int8_t c : out) emit(static_cast<int64_t>(c));
}

inline constexpr int8_t kPoison = 0x5A;

// RMSNorm: every width x shape, gains drawn per case (one small range the funnel accepts at every
// width, one wide range it refuses at some), and two site constants.
template <class Emit>
void RunNormCases(Emit& emit) {
	using superslm::CarriedScale;
	Rng rng(0x5331524F574E4F52ULL);  // "S1ROWNOR"
	const CarriedScale site_constants[] = {{INT64_C(1073741824), 0}, {INT64_C(1518500250), -3}};
	const int64_t gain_limits[] = {300, 40000};
	for (size_t n : kWidths)
		for (RowShape shape : kShapes)
			for (int64_t gain_limit : gain_limits)
				for (const CarriedScale& site_constant : site_constants) {
					std::vector<int8_t> h(n);
					FillRow(h, shape, rng);
					std::vector<int32_t> g(n);
					for (auto& v : g) v = static_cast<int32_t>(rng.InRange(-gain_limit, gain_limit));
					std::vector<int8_t> out(n, kPoison);
					CarriedScale scale{-1, -1};
					const auto st = superslm::RmsNormSite(h.data(), g.data(), n, CarriedScale{}, site_constant,
					                                      out.data(), &scale);
					EmitResult(emit, st, scale, out);
				}
}

// SwiGLU: every width x shape of the gate row, random up rows, and gate scales from the forward's
// range plus the small scale (m = 2^30, e = -35) at which sig(-128) = 589, sig(-127) = 608.
template <class Emit>
void RunSiluCases(Emit& emit) {
	using superslm::CarriedScale;
	Rng rng(0x533153494C555F43ULL);  // "S1SILU_C"
	const CarriedScale gate_scales[] = {{INT64_C(1073741824), -35}, {INT64_C(1073741824), -34},
	                                    {INT64_C(1631069115), -32}, {INT64_C(1216311211), -28},
	                                    {INT64_C(2147483647), -36}};
	const CarriedScale up_scale{INT64_C(1340958474), -18};
	const CarriedScale site_constant{INT64_C(1073741824), 0};
	for (size_t n : kWidths)
		for (RowShape shape : kShapes)
			for (const CarriedScale& gate_scale : gate_scales) {
				std::vector<int8_t> gate(n), up(n);
				FillRow(gate, shape, rng);
				FillRow(up, RowShape::kRandomNo128, rng);
				std::vector<int8_t> out(n, kPoison);
				CarriedScale scale{-1, -1};
				const auto st = superslm::MlpActSite(gate.data(), gate_scale, up.data(), up_scale, n,
				                                     superslm::kSiluLutCanonicalTable, site_constant,
				                                     out.data(), &scale);
				EmitResult(emit, st, scale, out);
			}
}

// A residual scale pair and what it is for.
struct ResidualScalePair {
	superslm::CarriedScale branch;
	superslm::CarriedScale stream;
};

// Residual: every width x shape of both rows over scale pairs from the forward's range (exponent
// gaps 0 to 20, both signs of the mantissa, both candidate orders), gaps beyond 31 both ways, and
// the landing-flag rows: the first candidate's "other" row is all 0 except one 127 at the middle
// element (refused mid-row, the second candidate commits), all 0 (never refused although code 127
// would be), and a refused first candidate with a second candidate the funnel refuses too.
template <class Emit>
void RunResidualCases(Emit& emit) {
	using superslm::CarriedScale;
	Rng rng(0x5331524553494455ULL);  // "S1RESIDU"
	const CarriedScale site_constant{INT64_C(1073741824), 0};
	const ResidualScalePair pairs[] = {
	    {{INT64_C(1518500250), -40}, {INT64_C(1073741824), -40}},
	    {{INT64_C(1234567890), -37}, {INT64_C(1987654321), -44}},
	    {{INT64_C(1987654321), -44}, {INT64_C(1234567890), -37}},
	    {{INT64_C(-1400000000), -39}, {INT64_C(1100000000), -41}},
	    {{INT64_C(1100000000), -52}, {INT64_C(-1900000000), -32}},
	    {{INT64_C(1300000000), -70}, {INT64_C(1300000000), -36}},  // gap 34: stream selected first
	    {{INT64_C(1300000000), -36}, {INT64_C(1300000000), -70}},  // gap 34: branch selected first
	};
	for (size_t n : kWidths)
		for (RowShape shape : kShapes)
			for (const ResidualScalePair& p : pairs) {
				std::vector<int8_t> branch(n), stream(n);
				FillRow(branch, shape, rng);
				FillRow(stream, RowShape::kRandomWithMinus128, rng);
				std::vector<int8_t> out(n, kPoison);
				CarriedScale scale{-1, -1};
				const auto st = superslm::ResidualReconcileSite(branch.data(), p.branch, stream.data(), p.stream,
				                                                n, site_constant, out.data(), &scale);
				EmitResult(emit, st, scale, out);
			}

	// The landing-flag rows. Branch e = 30, stream e = -30: the gap exceeds 31, so the stream
	// (finer) candidate is built first and lands the branch codes; code 127 overflows int64 there
	// and code 0 does not. The branch row is the "other" row of that first candidate.
	const CarriedScale coarse{INT64_C(1300000000), 30};
	const CarriedScale fine{INT64_C(1300000000), -30};
	const CarriedScale bad_site_constant{INT64_C(4294967296), 0};  // the funnel refuses it (m > int32)
	for (size_t n : kWidths) {
		std::vector<int8_t> stream(n);
		FillRow(stream, RowShape::kRandomWithMinus128, rng);
		for (int variant = 0; variant < 3; ++variant) {
			std::vector<int8_t> branch(n, 0);
			if (variant != 1) branch[n / 2] = 127;  // variants 0 and 2: refused mid-row
			const CarriedScale sc = variant == 2 ? bad_site_constant : site_constant;
			std::vector<int8_t> out(n, kPoison);
			CarriedScale scale{-1, -1};
			const auto st = superslm::ResidualReconcileSite(branch.data(), coarse, stream.data(), fine, n, sc,
			                                                out.data(), &scale);
			EmitResult(emit, st, scale, out);
		}
	}
}

// The whole S1 set, in a fixed order.
template <class Emit>
void RunRowTableCases(Emit& emit) {
	RunNormCases(emit);
	RunSiluCases(emit);
	RunResidualCases(emit);
}

}  // namespace superslm_rowsite_cases

#endif  // SUPERSLM_TESTS_SUPPORT_ROWSITE_CASES_H
