// SuperSLM SwiGLU SiLU via a fixed-point sigmoid LUT (C10, S2.4).
//
// See include/superslm/silu_lut.h for the runtime contract and
// SuperSLM_S2.4_SiLU_LUT_Design §5 (index derivation) / §6 (interpolation) for the
// construction. Entirely integer, entirely division-free: the two roundings reuse the shipped
// RoundingDivideByPOT primitive (int64 to place the sub-node index when e+k+Q_idx < 0, int32 to
// interpolate), so C3's tie rule is not re-implemented here. The table is generated offline in
// double precision by the converter and read from the artifact's SIL1 section; the caller
// materializes it into the native-endian int32 array this lookup indexes.
#include "superslm/silu_lut.h"

#include "superslm/intmath.h"  // RoundingDivideByPOT (int32 + int64 overloads), the C3 tie rule

namespace superslm {

int32_t SiluSigmoidQ15(const int32_t* table, int8_t code, int64_t m, int e) {
	// §5 — the real-valued table position pos = (code·realscale + X)·K, in Q(kSiluLutQIdx) fixed
	// point. realscale = m·2^e and K = N/2X = 2^kSiluLutLog2K, so pos's scale contribution is
	// code·m·2^(e+k) — shift-and-add, never a general multiply-then-divide.
	const int64_t term = static_cast<int64_t>(code) * m;  // exact int64: |term| < 127·2^31 < 2^39
	const int shift = e + kSiluLutLog2K + kSiluLutQIdx;    // e + 5 + 12

	// Place the sub-node position. The branch is chosen by shift's sign: shift >= 0 (i.e.
	// e >= -(k+Q_idx) = -17) is the exact left-shift; shift < 0 rounds under C3 via the int64
	// RoundingDivideByPOT (the S2.4 prep overload). Two-sided caller-ensures precondition (§10
	// item 3), the same convention the other kernels use — a scale outside the swept range is a
	// site-level rejection (C29), not this leaf's guard: on the left branch `shift < 24` keeps
	// `term << shift` exact within int64 (|term| < 2^38, so |result| < 2^62); on the right branch
	// `-shift <= 63` stays in RoundingDivideByPOT's exponent domain. For every real calibrated
	// (m,e) the branch is shift < 0, with -shift measured in [15,19] over the full calibrated
	// corpus (§10 item 3 width sweep, 28 layers x 2 forwards); the only shift >= 0 inputs are
	// code == 0 (term 0) or code extremes that saturate below. The real compiled kernel overflows
	// the left branch at shift == 26 and hits the RoundingDivideByPOT domain limit at -shift == 64,
	// both far outside the corpus range (margin >= 2 shifts to the left edge, >= 44 to the right).
	int64_t pos_fixed = shift >= 0 ? (term << shift) : RoundingDivideByPOT(term, -shift);

	// + N/2 (in Q(kSiluLutQIdx)), then saturate to the table domain [0, N<<Q_idx] BEFORE the split.
	const int64_t domain_hi = static_cast<int64_t>(kSiluLutN) << kSiluLutQIdx;  // N << Q_idx
	pos_fixed += domain_hi / 2;                                                 // (N<<Q_idx)/2 (N even)
	if (pos_fixed < 0) pos_fixed = 0;
	else if (pos_fixed > domain_hi) pos_fixed = domain_hi;

	// §5 correction (§14 amendment 4): clamp the node index FIRST — the N-th node has no "hi"
	// neighbour past it — then derive frac against the CLAMPED index, so the two are one reference
	// frame. i0 in [0, N-1]; frac in [0, 2^Q_idx], reaching 2^Q_idx exactly at high saturation
	// (§6 resolves that to table[N]).
	int64_t i0_64 = pos_fixed >> kSiluLutQIdx;
	if (i0_64 > kSiluLutN - 1) i0_64 = kSiluLutN - 1;
	const int i0 = static_cast<int>(i0_64);
	const int64_t frac = pos_fixed - (static_cast<int64_t>(i0) << kSiluLutQIdx);

	// §6 — linear interpolation, exactly one C3 rounding. table[i0+1] always addresses a real
	// entry (i0 <= N-1, and the table has N+1 entries for exactly this reason). Under the
	// canonical table (ParseSigmoidLut's pinned-content check, S-HARDEN-1 F20/F22 — the ONLY
	// table this function is ever called with in the shipped path) |frac*diff| <= 2^12 * 2^15 =
	// 2^27, which fits int32 exactly as the design derives. `diff`/`product`/`delta` are widened
	// to int64_t as DEFENCE IN DEPTH beneath that content-pinning guarantee, not because the
	// canonical table needs it: the guarantee is load-bearing at the parser, not re-derived here,
	// but the two unvalidated-artifact-operand UB chain (F22) is the standing argument for never
	// letting a downstream arithmetic step be the last line of defense on its own.
	const int64_t lo = table[i0];
	const int64_t hi = table[i0 + 1];
	const int64_t diff = hi - lo;
	const int64_t product = frac * diff;
	const int64_t delta = RoundingDivideByPOT(product, kSiluLutQIdx);
	return static_cast<int32_t>(lo + delta);
}

}  // namespace superslm
