// SuperSLM checked chain funnel — STUB (S3.1 red-phase).
//
// See include/superslm/checked_chain_funnel.h for the contract. These are
// deliberately-wrong sentinel bodies so a follow-up red suite (Claude/Curie/
// superslm-s3.1-checked-chain-funnel-test-design-2026-07-28.md §4) compiles, links,
// and FAILS against the real construction. Brunel replaces each body with §7.2's
// pinned composition in the S3.1 green phase.
//
// Every function below that can reject returns SslmForwardStatus::WorkspaceTooSmall
// unconditionally on the checked paths — a status none of this contract's own cells
// ever expects from these functions (their real answers are Ok, ChainInputOutOfDomain,
// or LogitNarrowingOverflow), so every case fails on an actual value mismatch rather
// than coincidentally passing. Output parameters are left untouched, matching the
// real contract's own "on rejection, neither is touched" rule — a caller comparing
// them against an expected accept-path value also gets a clean mismatch.
#include "superslm/checked_chain_funnel.h"

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

ChainResult RequantChainChecked(const int64_t*, size_t, std::span<const CarriedScale>,
                                 CarriedScale, int8_t*, CarriedScale*) {
	return ChainResult{SslmForwardStatus::WorkspaceTooSmall};  // stub
}

SslmForwardStatus NarrowRowChecked(const int64_t*, size_t, int32_t*) {
	return SslmForwardStatus::WorkspaceTooSmall;  // stub
}

SslmForwardStatus CheckIExpConstantsDomain(int64_t, int64_t, int64_t, int64_t) {
	// Stub encodes neither the design's textual C30 rule nor IExpConstantsInDomain's
	// own answer — the two are known to disagree over part of the swept domain and a
	// planner ruling on which governs is pending (see the header comment above).
	return SslmForwardStatus::WorkspaceTooSmall;
}

SslmForwardStatus CheckRoundingDivideByPotExponentDomain(int64_t, int64_t) {
	return SslmForwardStatus::WorkspaceTooSmall;  // stub
}

}  // namespace superslm
