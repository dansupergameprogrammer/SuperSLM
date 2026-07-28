// SuperSLM checked chain funnel — the S3.1 green-phase construction.
//
// See include/superslm/checked_chain_funnel.h for the contract (§7.2, §5.5, §7.2's
// second limb). This file is the funnel's own translation unit: the only place in
// the whole S3a forward composition permitted to call MaxAbsReduceWide/
// RowBoundsWide/NormalizeScale/DynamicScaleReciprocal/RequantTokenCodeWide/
// NarrowAccumulatorToI32 directly (§7.3's CI source check enforces this
// structurally on every other forward TU).
//
// CheckRoundingDivideByPotExponentDomain (C28) is declared in the header ahead of
// its own build (§11 S3.2) but is NOT implemented here — S3.1's own scope line
// names only the two derived-operand predicates for C30 and C34, and C28's site
// and predicate are S3.2's (Claude/Curie/superslm-s3.1-checked-chain-funnel-test-
// design-2026-07-28.md §9.2). Its stub body is left unchanged.
#include "superslm/checked_chain_funnel.h"

#include "superslm/intmath.h"
#include "superslm/matmul.h"
#include "superslm/trace_hook.h"

namespace superslm {

const char* SslmForwardStatusName(SslmForwardStatus s) noexcept {
	switch (s) {
		case SslmForwardStatus::Ok: return "Ok";
		case SslmForwardStatus::ChainInputOutOfDomain: return "ChainInputOutOfDomain";
		case SslmForwardStatus::LogitNarrowingOverflow: return "LogitNarrowingOverflow";
		case SslmForwardStatus::IExpConstantsOutOfDomain: return "IExpConstantsOutOfDomain";
		case SslmForwardStatus::RoundingDivideByPotExponentOutOfDomain:
			return "RoundingDivideByPotExponentOutOfDomain";
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

// C26's carried-scale product step, generalized to combine any two canonical
// CarriedScale operands (SuperSLM_Plan.md C26 row): one C1/C2 high-mul per
// combination (SaturatingRoundingDoublingHighMul, ties toward +infinity), then
// renormalize the Q31-style mantissa back into [2^30, 2^31) by an EXACT single
// shift (no rounding) when the high-mul result lands below the canonical floor.
// Both operands' mantissas fit int32_t (canonical range is a strict subset of
// int32's positive range); the high-mul's own product is < 2^62 and int64-safe,
// matching C26's stated width.
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

}  // namespace

ChainResult RequantChainChecked(const int64_t* wide_row, size_t n,
                                 std::span<const CarriedScale> incoming,
                                 CarriedScale site_constant, int8_t* out_codes,
                                 CarriedScale* out_scale,
                                 std::string_view site, size_t token_index) {
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

	// Step 5: RequantTokenCodeWide per element, directly on the int64 row — never
	// narrowed to int32 first (T-1254's fold).
	for (size_t i = 0; i < n; ++i) {
		out_codes[i] = RequantTokenCodeWide(wide_row[i], r, ns.s);
	}

	// Step 6: carried_scale_product in C26's pinned LEFT-ASSOCIATED order — the
	// incoming carried scale(s) first, then the site constant, then this token's
	// own D'-factor. D' itself is already exact and canonical from NormalizeScale's
	// own decomposition (Dn = D' << s for s >= 0, or D' >> 1 at the single s == -1
	// case), so D' == Dn * 2^(-s) with no further rounding: the D'-factor is
	// CarriedScale{ns.dn, -ns.s} exactly, needing no separate derivation.
	const CarriedScale d_prime_factor{ns.dn, -static_cast<int64_t>(ns.s)};
	bool have_running = false;
	CarriedScale running{};
	auto fold_in = [&](const CarriedScale& next) {
		if (!have_running) {
			running = next;
			have_running = true;
		} else {
			running = CombineCarriedScale(running, next);
		}
	};
	for (const CarriedScale& factor : incoming) fold_in(factor);
	fold_in(site_constant);
	fold_in(d_prime_factor);
	*out_scale = running;

	// §11 S3.1a's instrumentation seam (trace_hook.h), attached to this
	// already-green funnel per the sub-slot's own routing option. Runs
	// strictly after every write above and reads only what those writes
	// already produced -- wide_row/n, d_prime, ns.dn/ns.s, r, out_codes, and
	// the just-written *out_scale. It writes none of them, and does not run
	// at all when no hook is installed, so ChainResult/out_codes/*out_scale
	// are identical whether or not a hook is installed (§10.3's
	// instrumentation axis).
	if (SslmTraceHookInstalled()) {
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
		SslmEmitChainTrace(record);
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

SslmForwardStatus CheckRoundingDivideByPotExponentDomain(int64_t, int64_t) {
	// Not this pass's to implement — S3.2's own build (§9.2 of the test-design
	// record cited above). Stub body unchanged.
	return SslmForwardStatus::WorkspaceTooSmall;  // stub
}

}  // namespace superslm
