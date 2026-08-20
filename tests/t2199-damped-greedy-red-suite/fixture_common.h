// T-2199 (Curie) -- shared CHECK harness and fixture helpers for the damped greedy decoding
// (Phase A anti-LM / Phase C score-and-select) red suite. Reuses this repo's own established
// CHECK/CHECK_MSG/SKIP_MSG counter-triple convention (tests/t2130-g5-red-suite/
// fixture_common.h's own shape, itself tests/t2112-gpu-1p0-red-suite's, itself
// tests/test_main.cpp's) -- no new harness invented.
//
// Unlike t2130/t2112, this suite needs NO real model artifact: Phase A (the anti-LM) and
// Phase C (the score-and-select primitive) are both "no engine dependency, buildable and
// testable standalone" per the plan's own Sec8 phase-ordering rationale. Every fixture here
// is either hand-computed or adversarially constructed directly against the certified
// sub-primitives (intmath.h, checked_chain_funnel.h) the plan's own citations name -- the
// same standalone-probe shape Loki's strike-6 probe
// (Claude/Loki/t2199-strike6-probe-fsd-mask.cpp, Wizard repo) already used successfully for
// this exact mechanism, reused here as this suite's own construction precedent.
#ifndef SSLM_T2199_FIXTURE_COMMON_H
#define SSLM_T2199_FIXTURE_COMMON_H

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "superslm/intmath.h"
#include "superslm/checked_chain_funnel.h"
#include "sslm_damped_greedy.h"

static int GChecks = 0;
static int GFailures = 0;
static int GSkips = 0;

#define CHECK(cond) \
	do { \
		++GChecks; \
		if (!(cond)) { \
			++GFailures; \
			std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
		} \
	} while (0)

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

#define SKIP_MSG(...) \
	do { \
		++GSkips; \
		std::printf("SKIP %s:%d -- ", __FILE__, __LINE__); \
		std::printf(__VA_ARGS__); \
		std::printf("\n"); \
	} while (0)

namespace t2199fixture {

using namespace superslm;

// The real vocab width the plan cites throughout (Sec2.3, Sec9 dim4) -- Qwen2.5's own
// 151,936-row LM head. Fixtures that do not need the real width use a small synthetic V
// instead (stated at each call site) so the row can be hand-inspected in a FAIL message.
inline constexpr int32_t kRealVocabSize = 151936;

// ---- packed mask helpers (1 bit/vocab position, LSB-first per byte -- the same layout
// checked_chain_funnel.h's own masking half and Loki's strike-6 probe both use) ----
inline std::vector<uint8_t> MakeMask(int32_t vocab_size, bool all_legal) {
	return std::vector<uint8_t>((vocab_size + 7) / 8, all_legal ? 0xFF : 0x00);
}
inline void SetLegal(std::vector<uint8_t>& mask, int32_t token) {
	mask[static_cast<size_t>(token) >> 3] |= static_cast<uint8_t>(1u << (token & 7));
}
inline void ClearLegal(std::vector<uint8_t>& mask, int32_t token) {
	mask[static_cast<size_t>(token) >> 3] &= static_cast<uint8_t>(~(1u << (token & 7)));
}
inline bool IsLegal(const std::vector<uint8_t>& mask, int32_t token) {
	return ((mask[static_cast<size_t>(token) >> 3] >> (token & 7)) & 1u) != 0;
}

// Applies a packed mask to a plain int32 logit row, floor-narrowing every masked position to
// INT32_MIN -- the plan's own "mask first" convention (Sec6), mirroring
// ApplyMaskAndArgmax's masking half exactly (Loki strike-6 probe's own ApplyMaskHalf).
inline void ApplyMaskFloor(std::vector<int32_t>& row, const std::vector<uint8_t>& mask) {
	for (size_t t = 0; t < row.size(); ++t)
		if (!IsLegal(mask, static_cast<int32_t>(t))) row[t] = kInt32Min;
}

// ---- (q_ln2, q_b, q_c) derivation -----------------------------------------------------
// Derives a real, domain-valid (q_ln2, q_b, q_c) triple via the certified IExpScaleConstants
// (intmath.h), searching (m, e) for the plan's own corrected starting scale, q_ln2 = 493
// (plan Sec5.6.2) -- the SAME calibration Loki's strike-6 probe used, reused here as this
// suite's own known-valid derivation rather than hand-typed integers (StandardsDocument.md
// Sec5.4: verified at source or by execution, never by construction). Cells that need a
// DIFFERENT q_ln2 (the erasure-zone cell, Sec9 dim6's qspread-collapse construction:
// "q_ln2 >= 23,378") search for that target instead; `target_q_ln2` selects which.
inline bool DeriveScaleConstants(int64_t target_q_ln2, int64_t* out_q_ln2, int64_t* out_q_b,
                                  int64_t* out_q_c) {
	for (int64_t e = -80; e <= 0; ++e) {
		for (int64_t m = (int64_t)1 << 30; m < ((int64_t)1 << 31); m += ((int64_t)1 << 20)) {
			int64_t a = 0, b = 0, c = 0;
			if (IExpScaleConstants(m, e, kIExpLn2Q, 30, kIExpBQ, 30, kIExpCaQ, 30, &a, &b, &c) !=
			    IExpScaleDomain::kOk)
				continue;
			if (a == target_q_ln2) {
				*out_q_ln2 = a;
				*out_q_b = b;
				*out_q_c = c;
				return true;
			}
		}
	}
	return false;
}

// The plan's own corrected starting scale (Sec5.6.2), reused as this suite's default
// "faithful calibration band" fixture -- q_ln2 = 493 is well inside the faithful band Sec9
// dim7's own alpha=0-identity cell cites (q_ln2 <= 7,893).
inline bool DeriveDefaultScaleConstants(int64_t* out_q_ln2, int64_t* out_q_b, int64_t* out_q_c) {
	return DeriveScaleConstants(493, out_q_ln2, out_q_b, out_q_c);
}

// A hand-inspectable, small synthetic row with a controlled, well-separated top-k -- used by
// cells that pin exact computed values rather than merely a legal/illegal classification.
// Position `t`'s logit is a deterministic pseudo-random spread in [-2000, 2000), with
// `top_tokens` (already sorted, most-favoured first) overridden to strictly descending,
// well-separated values above that spread so top-k selection is unambiguous.
inline std::vector<int32_t> MakeSyntheticRow(int32_t vocab_size,
                                              const std::vector<int32_t>& top_tokens) {
	std::vector<int32_t> row(static_cast<size_t>(vocab_size));
	for (int32_t t = 0; t < vocab_size; ++t)
		row[static_cast<size_t>(t)] = static_cast<int32_t>((t * 2654435761u) % 4000) - 2000;
	int32_t v = 20000;
	for (int32_t tok : top_tokens) {
		row[static_cast<size_t>(tok)] = v;
		v -= 1000;  // strictly descending, 1000 apart -- no accidental ties
	}
	return row;
}

}  // namespace t2199fixture

#endif  // SSLM_T2199_FIXTURE_COMMON_H
