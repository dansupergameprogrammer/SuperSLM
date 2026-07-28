// SuperSLM checked chain funnel — the S3.1 green-phase construction.
//
// See include/superslm/checked_chain_funnel.h for the contract (§7.2, §5.5, §7.2's
// second limb). This file is the funnel's own translation unit: the only place in
// the whole S3a forward composition permitted to call MaxAbsReduceWide/
// RowBoundsWide/NormalizeScale/DynamicScaleReciprocal/RequantTokenCodeWide/
// NarrowAccumulatorToI32 directly (§7.3's CI source check enforces this
// structurally on every other forward TU).
//
// C28's derived-operand pair predicate (CheckRoundingDivideByPotExponentDomain) is
// S3.2's own build (Claude/Curie/superslm-s3.2-weightless-and-projection-sites-
// test-design-2026-07-28.md §9 item 4). It is declared in the header above,
// re-staged from its original S3.1 header-contract declaration (32aca0c), and
// its body below is the real 0 <= q_B + 62 + e_a <= 63 comparison (S3.2 green
// phase).
//
// C32/D-SLM366's derived-operand predicate (CheckSoftmaxRowWidthDomain) is
// S3.3's own green-phase construction (Claude/Curie/superslm-s3.3-attention-
// interior-test-design-2026-07-28.md §6.2, §11) — see its own comment.
#include "superslm/checked_chain_funnel.h"

#include "superslm/intmath.h"
#include "superslm/matmul.h"
#include "superslm/silu_lut.h"
#include "superslm/trace_hook.h"

namespace superslm {

const char* SslmForwardStatusName(SslmForwardStatus s) noexcept {
	switch (s) {
		case SslmForwardStatus::Ok: return "Ok";
		case SslmForwardStatus::ChainInputOutOfDomain: return "ChainInputOutOfDomain";
		case SslmForwardStatus::LogitNarrowingOverflow: return "LogitNarrowingOverflow";
		case SslmForwardStatus::IExpConstantsOutOfDomain: return "IExpConstantsOutOfDomain";
		case SslmForwardStatus::CarriedScaleMantissaOutOfDomain: return "CarriedScaleMantissaOutOfDomain";
		case SslmForwardStatus::SiluCompositionScaleOutOfDomain: return "SiluCompositionScaleOutOfDomain";
		case SslmForwardStatus::RoundingDivideByPotExponentOutOfDomain:
			return "RoundingDivideByPotExponentOutOfDomain";
		case SslmForwardStatus::SoftmaxRowWidthOutOfDomain: return "SoftmaxRowWidthOutOfDomain";
		case SslmForwardStatus::TokenIdOutOfRange: return "TokenIdOutOfRange";
		case SslmForwardStatus::PositionOverCap: return "PositionOverCap";
		case SslmForwardStatus::WorkspaceTooSmall: return "WorkspaceTooSmall";
		case SslmForwardStatus::KvCapacityExhausted: return "KvCapacityExhausted";
		case SslmForwardStatus::KvPrecisionUnsupported: return "KvPrecisionUnsupported";
		case SslmForwardStatus::InvalidLayerBudget: return "InvalidLayerBudget";
	}
	return "?";
}

namespace {

// C26's carried-scale product step, generalized to combine any two CarriedScale
// operands (SuperSLM_Plan.md C26 row): one C1/C2 high-mul per combination
// (SaturatingRoundingDoublingHighMul, ties toward +infinity), then renormalize the
// Q31-style mantissa back into [2^30, 2^31) by an EXACT single shift (no rounding)
// when the high-mul result lands below the canonical floor; the high-mul's own
// product is < 2^62 and int64-safe, matching C26's stated width.
//
// **Precondition: both operands' mantissas fit int32_t's own range** (ac34677 S5,
// corrected — the prior comment here claimed this always held because "canonical
// range is a strict subset of int32's positive range", which is true of a CANONICAL
// operand but does not hold of a mid-composition one, and the header
// (checked_chain_funnel.h's CarriedScale doc) states mid-composition values are not
// required to stay canonical). An operand outside int32_t's range silently
// truncates through the unconditional cast below — executed at m = 2^31, one past
// INT32_MAX, which produced a negative mantissa with no diagnostic. This function
// does not check the precondition itself (it has no failure return); the caller,
// RequantChainChecked, checks every operand against it before folding (step 0), and
// checks the fold's own running product after every combine (380b75f review N1),
// returning CarriedScaleMantissaOutOfDomain rather than calling this function again
// -- or trusting its result -- when the precondition does not hold.
//
// **The renormalization's `m < 2^30` comparison is signed, and that is not a second
// defect** (380b75f review N1's own question, resolved by construction below): a
// negative high-mul result is always less than the positive floor, so it is
// unconditionally doubled regardless of how large its magnitude already is --
// executed at `CombineCarriedScale({INT32_MIN, 0}, {INT32_MAX, 0})`, which doubles
// an already-int32-representable m = -2147483647 into m = -4294967294. But
// doubling m and decrementing e in lockstep is EXACT and value-preserving
// (`m * 2^e` is unchanged by the transform, in int64 arithmetic, whichever sign m
// carries) -- the redundant double changes nothing this function itself computes
// wrong. The only place the doubled value can do damage is the NEXT combine's
// unconditional `static_cast<int32_t>` above, which is exactly the precondition
// this function has always documented and exactly what the caller now checks
// after every fold step, on `running`, before it is ever used as that next
// operand. A magnitude-aware comparison here would suppress this specific
// witness's overflow without removing the underlying gap: any other pair of
// operands whose high-mul lands further outside [-2^30, 2^30) still doubles into
// a value the same caller-side check must catch regardless. The fix belongs on
// the channel every case funnels through -- the caller's own check on `running` --
// not on this comparison.
CarriedScale CombineCarriedScale(CarriedScale a, CarriedScale b) {
	const int32_t ma = static_cast<int32_t>(a.m);
	const int32_t mb = static_cast<int32_t>(b.m);
	int64_t e = a.e + b.e + 31;
	int64_t m = static_cast<int64_t>(SaturatingRoundingDoublingHighMul(ma, mb));
	if (m < (int64_t{1} << 30)) {
		m <<= 1;
		e -= 1;
	}
	return CarriedScale{m, e};
}

// ac34677 S5's fix: the precondition CombineCarriedScale actually needs. Checked
// on every `incoming` factor and `site_constant` before folding starts (step 0),
// and on the fold's own running product after every combine (380b75f review N1) --
// `running` is CombineCarriedScale's left operand on every fold after the first,
// and step 0 alone never sees it.
bool CarriedScaleMantissaFitsInt32(const CarriedScale& c) {
	return c.m >= static_cast<int64_t>(kInt32Min) && c.m <= static_cast<int64_t>(kInt32Max);
}

// CheckSoftmaxRowWidthDomain's own portable 128-bit facility (Poirot
// 2026-07-28 finding 1's fix): forming `q_b*q_b + q_c` must not reproduce the
// exact int64 overflow intmath.h:391-395 documents as unsafe for a caller to
// re-derive (D-SLM81). Signed 128-bit, two's-complement -- mirroring
// intmath.cpp's own private S128 facility and forward_sites.cpp's own U128
// precedent for the identical reason: neither is reachable across this
// file's translation-unit boundary, and both of those own comments make the
// same point about their own local copies (a small, self-contained facility
// per TU, not a shared instance).
struct S128 { uint64_t lo, hi; };  // two's complement

inline S128 S128FromI64(int64_t v) {
	return S128{static_cast<uint64_t>(v), v < 0 ? ~uint64_t{0} : uint64_t{0}};
}

inline S128 S128Mul(int64_t a, int64_t b) {
	const uint64_t ua = a < 0 ? (~static_cast<uint64_t>(a) + 1u) : static_cast<uint64_t>(a);
	const uint64_t ub = b < 0 ? (~static_cast<uint64_t>(b) + 1u) : static_cast<uint64_t>(b);
	const uint64_t ll = (ua & 0xFFFFFFFFull) * (ub & 0xFFFFFFFFull);
	const uint64_t lh = (ua & 0xFFFFFFFFull) * (ub >> 32);
	const uint64_t hl = (ua >> 32) * (ub & 0xFFFFFFFFull);
	const uint64_t hh = (ua >> 32) * (ub >> 32);
	const uint64_t mid = (ll >> 32) + (lh & 0xFFFFFFFFull) + (hl & 0xFFFFFFFFull);
	const uint64_t lo = (ll & 0xFFFFFFFFull) | (mid << 32);
	const uint64_t hi = hh + (lh >> 32) + (hl >> 32) + (mid >> 32);
	S128 r{lo, hi};
	if ((a < 0) ^ (b < 0)) {  // two's-complement negate
		r.lo = ~r.lo;
		r.hi = ~r.hi;
		if (++r.lo == 0) ++r.hi;
	}
	return r;
}

inline S128 S128Add(S128 a, S128 b) {
	const uint64_t lo = a.lo + b.lo;
	const uint64_t carry = (lo < a.lo) ? 1u : 0u;
	return S128{lo, a.hi + b.hi + carry};
}

inline bool S128Ge(S128 a, S128 b) {  // signed a >= b
	const int64_t ah = static_cast<int64_t>(a.hi), bh = static_cast<int64_t>(b.hi);
	if (ah != bh) return ah > bh;
	return a.lo >= b.lo;
}

// Representable in int64_t iff sign-extending the low 64 bits reproduces the
// full 128-bit value -- the same test SShrToI64's own callers rely on
// implicitly elsewhere in this tree, made explicit here because this
// predicate needs the yes/no rather than a narrowed value.
inline bool S128FitsI64(S128 v) {
	const int64_t lo_signed = static_cast<int64_t>(v.lo);
	const uint64_t expected_hi = lo_signed < 0 ? ~uint64_t{0} : uint64_t{0};
	return v.hi == expected_hi;
}

}  // namespace

ChainResult RequantChainChecked(const int64_t* wide_row, size_t n,
                                 std::span<const CarriedScale> incoming,
                                 CarriedScale site_constant, int8_t* out_codes,
                                 CarriedScale* out_scale,
                                 std::string_view site, size_t token_index,
                                 SslmTraceHookState* trace_hook_state) {
	// Step 0 (ac34677 S5): every CarriedScale that will reach CombineCarriedScale
	// below must fit that function's own precondition (mantissa within int32_t's
	// range) before anything else runs — checked here, ahead of step 1, so a
	// rejection at this step leaves out_codes and *out_scale untouched exactly like
	// every other rejection path (step 5 has not run yet).
	for (const CarriedScale& factor : incoming) {
		if (!CarriedScaleMantissaFitsInt32(factor)) {
			return ChainResult{SslmForwardStatus::CarriedScaleMantissaOutOfDomain};
		}
	}
	if (!CarriedScaleMantissaFitsInt32(site_constant)) {
		return ChainResult{SslmForwardStatus::CarriedScaleMantissaOutOfDomain};
	}

	// Steps 1-2 (§7.2): MaxAbsReduceWide already returns D' with C20's all-zero-row
	// guard (D' = max(D, 1)) baked in — same contract shape as the narrow
	// MaxAbsReduce sibling.
	const int64_t d_prime = MaxAbsReduceWide(wide_row, n);

	// Step 3: C29's own domain check — C21's precondition for NormalizeScale, and
	// nothing else (T-1254's correction: this entitles step 4 alone, never a later
	// narrowing to int32).
	if (d_prime > (int64_t{1} << 31)) {
		return ChainResult{SslmForwardStatus::ChainInputOutOfDomain};
	}

	// Step 4: NormalizeScale -> DynamicScaleReciprocal, both already int64-domain.
	const NormalizedScale ns = NormalizeScale(d_prime);
	const int64_t r = DynamicScaleReciprocal(ns.dn);

	// Step 5's own carried_scale_product, computed here -- ahead of step 6's
	// per-element write below -- in C26's pinned LEFT-ASSOCIATED order: the
	// incoming carried scale(s) first, then the site constant, then this token's
	// own D'-factor. D' itself is already exact and canonical from NormalizeScale's
	// own decomposition (Dn = D' << s for s >= 0, or D' >> 1 at the single s == -1
	// case), so D' == Dn * 2^(-s) with no further rounding: the D'-factor is
	// CarriedScale{ns.dn, -ns.s} exactly, needing no separate derivation.
	//
	// 380b75f review N1: step 0 above checks every `incoming` factor and
	// `site_constant` against CombineCarriedScale's own precondition, but never
	// `running` -- the fold's own left operand on every combine after the first,
	// and exactly what step 0 cannot see because it does not exist until the fold
	// runs. Checked here instead, after every fold step and before `running` is
	// ever used as the next combine's operand or written to `*out_scale`: a fold
	// whose own product drifts out of int32_t's range is rejected with the same
	// CarriedScaleMantissaOutOfDomain status step 0 uses. Computing and validating
	// the fold before step 6's loop -- rather than after, in the order the two
	// steps' own numbering might otherwise suggest -- means this rejection path
	// leaves out_codes untouched exactly like every other rejection (step 6 has
	// not run yet); step 5 and step 6 do not depend on each other's outputs, so
	// running the fold first changes nothing step 6 itself does.
	const CarriedScale d_prime_factor{ns.dn, -static_cast<int64_t>(ns.s)};
	bool have_running = false;
	bool fold_in_domain = true;
	CarriedScale running{};
	auto fold_in = [&](const CarriedScale& next) {
		if (!fold_in_domain) return;
		running = have_running ? CombineCarriedScale(running, next) : next;
		have_running = true;
		fold_in_domain = CarriedScaleMantissaFitsInt32(running);
	};
	for (const CarriedScale& factor : incoming) fold_in(factor);
	fold_in(site_constant);
	fold_in(d_prime_factor);
	if (!fold_in_domain) {
		return ChainResult{SslmForwardStatus::CarriedScaleMantissaOutOfDomain};
	}

	// Step 6: RequantTokenCodeWide per element, directly on the int64 row — never
	// narrowed to int32 first (T-1254's fold).
	for (size_t i = 0; i < n; ++i) {
		out_codes[i] = RequantTokenCodeWide(wide_row[i], r, ns.s);
	}
	*out_scale = running;

	// §11 S3.1a's instrumentation seam (trace_hook.h), attached to this
	// already-green funnel per the sub-slot's own routing option. Runs
	// strictly after every write above and reads only what those writes
	// already produced -- wide_row/n, d_prime, ns.dn/ns.s, r, out_codes, and
	// the just-written *out_scale. It writes none of them, and does not run
	// at all when no hook is installed, so ChainResult/out_codes/*out_scale
	// are identical whether or not a hook is installed (§10.3's
	// instrumentation axis). `trace_hook_state` is the caller's own model
	// handle's state (D-SLM353) -- a null pointer (no handle passed) means no
	// tracing, exactly like a handle whose hook is uninstalled.
	if (trace_hook_state != nullptr && SslmTraceHookInstalled(*trace_hook_state)) {
		SslmChainTraceRecord record;
		record.site = site;
		record.token_index = token_index;
		record.x_int = std::span<const int64_t>(wide_row, n);
		record.d_prime = d_prime;
		record.dn = ns.dn;
		record.s = ns.s;
		record.r = r;
		record.codes = std::span<const int8_t>(out_codes, n);
		record.m_out = running.m;
		record.e_out = running.e;
		SslmEmitChainTrace(*trace_hook_state, record);
	}

	return ChainResult{SslmForwardStatus::Ok};
}

SslmForwardStatus NarrowRowChecked(const int64_t* wide_row, size_t n, int32_t* out_i32) {
	// Step 1 (§7.2, §5.5): the row's signed extremes — NOT MaxAbsReduceWide, which
	// expresses only magnitude and cannot see the asymmetric int32 target range.
	int64_t row_max = 0;
	int64_t row_min = 0;
	RowBoundsWide(wide_row, n, &row_max, &row_min);

	// Step 2: C35's own domain check — a different check, on a different
	// quantity, from C29's magnitude bound above (T-1254).
	if (row_max > static_cast<int64_t>(kInt32Max) || row_min < static_cast<int64_t>(kInt32Min)) {
		return SslmForwardStatus::LogitNarrowingOverflow;
	}

	// Step 3: NarrowAccumulatorToI32, genuinely sound now that every element has
	// been proven, individually and by its own sign, to lie in int32's
	// representable range.
	NarrowAccumulatorToI32(wide_row, n, out_i32);
	return SslmForwardStatus::Ok;
}

SslmForwardStatus CheckIExpConstantsDomain(int64_t q, int64_t q_ln2, int64_t q_b, int64_t q_c) {
	// Encodes no threshold of its own (§7.2's second limb, D-SLM348's ruling): the
	// already-shipped IExpConstantsInDomain is the total, sole domain authority.
	return IExpConstantsInDomain(q, q_ln2, q_b, q_c) ? SslmForwardStatus::Ok
	                                                  : SslmForwardStatus::IExpConstantsOutOfDomain;
}

namespace {

// The runtime no-UB domain's own ceiling on `e`, derived from SiluSigmoidQ15's two
// shift placements (silu_lut.cpp:21,35): `shift = e + kSiluLutLog2K + kSiluLutQIdx`.
// The left branch (shift >= 0) is exact only while `term << shift` stays in int64,
// i.e. strictly below kSiluLutTermLeftShiftOverflowExponent; the right branch
// (shift < 0) calls RoundingDivideByPOT(int64_t, int), defined only up to
// kRoundingDivideByPotExponentMaxI64. Named constants, not literals, so a change to
// either primitive's own domain is caught here at compile time (§5.4).
constexpr int kSiluCompositionRuntimeMaxE =
    kSiluLutTermLeftShiftOverflowExponent - kSiluLutLog2K - kSiluLutQIdx - 1;  // 8
constexpr int kSiluCompositionRuntimeMinE =
    -(kRoundingDivideByPotExponentMaxI64 + kSiluLutLog2K + kSiluLutQIdx);  // -80

// The mirror §5.4 calls for: proves at compile time that S-HARDEN-1's load-time
// ceiling (kCompositionScaleMaxE/MinE, silu_lut.h) never exceeds this predicate's
// own runtime ceiling — the ordering the loader's own model.cpp:602,606 asserts on
// its own literal, restated here from the runtime side so a change to either the
// loader's floor or this predicate's domain fails the build instead of drifting
// silently out of the containment relation §5.4 requires.
static_assert(kCompositionScaleMaxE <= kSiluCompositionRuntimeMaxE,
              "the load-time CompositionConstants ceiling on e must not exceed the "
              "runtime SwiGLU no-UB ceiling (C34, ac34677 S11) — containment would "
              "otherwise fail on the loader's own accepted upper range");
static_assert(kCompositionScaleMinE >= kSiluCompositionRuntimeMinE,
              "the load-time CompositionConstants floor on e must not fall below the "
              "runtime SwiGLU no-UB floor (C34, ac34677 S11) — containment would "
              "otherwise fail on the loader's own accepted lower range");

}  // namespace

SslmForwardStatus CheckRoundingDivideByPotExponentDomain(int64_t q_B, int64_t e_a) {
	// C28's own derived-operand pair predicate (§7.2 second limb, §4.4; S3.2):
	// 0 <= q_B + 62 + e_a <= 63, named against RoundingDivideByPOT's own int64
	// exponent domain constants rather than the literals 0 and 63, so neither
	// side can drift from the primitive without failing a compile.
	const int64_t k = q_B + 62 + e_a;
	if (k < kRoundingDivideByPotExponentMinI64 || k > kRoundingDivideByPotExponentMaxI64) {
		return SslmForwardStatus::RoundingDivideByPotExponentOutOfDomain;
	}
	return SslmForwardStatus::Ok;
}

SslmForwardStatus CheckSiluCompositionScaleDomain(int64_t m, int64_t e) {
	// The no-UB domain itself (§5.4): `|m|` must stay within kCompositionScaleMaxAbsM
	// (the same symmetric bound SiluSigmoidQ15's `term = code * m` needs to stay
	// int64-exact, |term| < 2^39, and the same one the loader's own gate uses), and
	// `e` must keep both of that function's shift placements in range. Wider
	// than S-HARDEN-1's load-time descriptor on the upper branch by construction
	// (kSiluCompositionRuntimeMaxE = 8 > kCompositionScaleMaxE = 7) — e = 8 is the
	// one point of difference §5.4 executes: rejected at load time, accepted here.
	if (m < -kCompositionScaleMaxAbsM || m > kCompositionScaleMaxAbsM) {
		return SslmForwardStatus::SiluCompositionScaleOutOfDomain;
	}
	if (e < kSiluCompositionRuntimeMinE || e > kSiluCompositionRuntimeMaxE) {
		return SslmForwardStatus::SiluCompositionScaleOutOfDomain;
	}
	return SslmForwardStatus::Ok;
}

// S3.3's own green-phase construction (Claude/Curie/superslm-s3.3-attention-
// interior-test-design-2026-07-28.md §6.2, §11). D-SLM365's closed form:
// M = q_b^2 + q_c is the row's own i-exp value at q = 0, where ShiftByMax
// puts the row maximum -- the numerator ceiling C32's Q15 divide needs.
// D-SLM367 ruled kSoftmaxRowMaxSafeExponent (2^47) as the numerator bound on
// every path (§3).
//
// **Poirot 2026-07-28 finding 1 (CRITICAL), fixed here.** Forming
// `q_b*q_b + q_c` in int64 is the exact computation intmath.h:391-395
// documents as unsafe for a caller to perform ("the obvious check ...
// squares base in int64 and itself overflows once q_b exceeds ~3.04e9 ...
// Callers therefore use this predicate; they do not re-derive it",
// D-SLM81) -- this function was that re-derivation. 108 points over the
// canonical mantissa domain produce a fully int64_t-representable
// (q_b, q_c) pair whose sum is not, wrapping the guard's own threshold
// negative and defeating both limbs at once. The fix, per the same comment's
// own instruction: evaluate at 128-bit width, the same domain
// IExpConstantsInDomain already uses internally, rather than re-deriving a
// second int64 guard in front of the first.
SslmForwardStatus CheckSoftmaxRowWidthDomain(int64_t q_b, int64_t q_c, size_t width) {
	// **`q_c >= 0`, checked first (Poirot/Popper 2026-07-28, this pass).**
	// Both Null 2 witnesses (`checked_chain_funnel.h`'s own comment above)
	// reach `M <= 0` by having a NEGATIVE `q_c` cancel against `q_b*q_b` --
	// `q_b=10, q_c=-100` gives `M=0`, and the sign-asymmetry witness reaches
	// `M<0` the same way. Rejecting `q_c < 0` outright removes that
	// cancellation at its source: with `q_c >= 0`, `M = q_b*q_b + q_c >= 0`
	// unconditionally (a non-negative square plus a non-negative addend),
	// so the sum check below can never see a negative `m` and the
	// signed-to-`size_t` cast Null 2's second bullet exploited
	// (`INT64_MAX / m` with `m < 0`) is no longer reachable through this
	// function -- not patched, removed. `q_c` is checked in plain int64_t
	// (no overflow risk: a signed comparison against 0 needs no wide
	// arithmetic).
	if (q_c < 0) {
		return SslmForwardStatus::SoftmaxRowWidthOutOfDomain;
	}
	const S128 m128 = S128Add(S128Mul(q_b, q_b), S128FromI64(q_c));
	// Representable in int64_t AND within the ratified ceiling -- both
	// judged at 128-bit width so neither test can itself overflow the way
	// the int64 re-derivation did.
	if (!S128FitsI64(m128) || !S128Ge(S128FromI64(kSoftmaxRowMaxSafeExponent), m128)) {
		return SslmForwardStatus::SoftmaxRowWidthOutOfDomain;
	}
	const int64_t m = static_cast<int64_t>(m128.lo);
	// The sum `width * M` is checked without forming it (an overflowing
	// multiply is itself UB), mirroring the comment this predicate's own
	// declaration specifies. `m` is now PROVEN non-negative by the `q_c >= 0`
	// guard above (not merely unreached-on-the-checked-population, as the
	// prior version of this comment claimed) -- a non-negative square plus a
	// non-negative addend cannot be negative, in exact 128-bit arithmetic,
	// for any `q_b`.
	//
	// **This bound is necessary but not sufficient for the row's real
	// safety** (Popper 2026-07-28 Null 1): `M` is the closed form's value at
	// the row's shifted-max element (`q = 0`), and it bounds every OTHER
	// element only under the ratio `2*q_b >= q_ln2 - 1` (intmath.h:414) --
	// this predicate has no `q_ln2` parameter with which to check that ratio,
	// so a `q_c >= 0`, in-ceiling `M` can still be an unsound stand-in for a
	// row that is off that ratio. `SoftmaxRowQ15` (intmath.cpp) closes that
	// gap independently: it recomputes this same `M` from its own `q_b`/
	// `q_c` arguments and enforces it as a real ceiling against the row's
	// ACTUAL evaluated per-element values -- the observation this
	// closed-form predicate structurally cannot make.
	if (m != 0 && width > static_cast<size_t>(INT64_MAX / m)) {
		return SslmForwardStatus::SoftmaxRowWidthOutOfDomain;
	}
	return SslmForwardStatus::Ok;
}

// C33's own position-cap guard (§11 S3.3's own gate line: "a position ==
// context_cap is rejected before a table read"; Board T-1308). Declared here,
// in this file's own §7.2 second-limb predicate family, so a follow-up Curie
// pass can attach a red cell against a real, callable symbol -- T-1308 named
// the absence of any callable predicate or stub at either sub-slot (S3.3 or
// S3.6) as the blocker itself, not merely a routing question (no `tests/`
// edit accompanies this declaration; tests/ stays read-only to this
// campaign). `position` is a host/runtime-supplied sequence position;
// `context_cap` is the artifact's own config field (model.h). Rejects when
// `position` is outside `[0, context_cap)` -- the cap is an EXCLUSIVE upper
// bound, matching the plan's own "position == context_cap is rejected"
// wording (equality with the cap is already one past the last valid slot).
// Wiring this into an actual forward call site against a loaded model's real
// context_cap remains S3.6's own job (§9.1); this predicate only makes the
// comparison callable.
SslmForwardStatus CheckPositionOverCap(int64_t position, int64_t context_cap) {
	if (position < 0 || position >= context_cap) {
		return SslmForwardStatus::PositionOverCap;
	}
	return SslmForwardStatus::Ok;
}

}  // namespace superslm
