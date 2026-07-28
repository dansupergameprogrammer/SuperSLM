// SuperSLM S3.2 site compositions — STUB (S3.2 red-phase).
//
// See include/superslm/forward_sites.h for the contract. These are
// deliberately-wrong sentinel bodies so a follow-up red suite (Claude/Curie/
// superslm-s3.2-weightless-and-projection-sites-test-design-2026-07-28.md §4)
// compiles, links, and FAILS against the real construction. Brunel replaces each
// body with its real composition in the S3.2 green phase.
//
// Every SslmForwardStatus-returning function below returns WorkspaceTooSmall
// unconditionally — a status none of this contract's own expected outcomes
// (Ok, TokenIdOutOfRange) ever is, matching the S3.1/S2.1 stub precedent
// exactly (32aca0c). Every int64_t-returning function returns 0 unconditionally,
// matching the same precedent (3baad98) for value-returning kernels. Output
// parameters are left untouched on every path, matching the funnel's own
// "on rejection, neither is touched" convention.
#include "superslm/forward_sites.h"

namespace superslm {

int64_t FloorDivI64(int64_t /*a*/, int64_t /*b*/) {
	return 0;  // stub
}

SslmForwardStatus RmsNormSite(const int8_t* /*h*/, const int32_t* /*g*/,
                               size_t /*hidden_size*/, CarriedScale /*incoming_scale*/,
                               CarriedScale /*site_constant*/, int8_t* /*out_codes*/,
                               CarriedScale* /*out_scale*/) {
	return SslmForwardStatus::WorkspaceTooSmall;  // stub
}

int64_t ApplyWeightScaleFold(int64_t /*acc*/, int32_t /*identity*/, int32_t /*mult*/,
                              int32_t /*shift*/) {
	return 0;  // stub
}

int64_t BiasReconcile(int64_t /*b*/, int64_t /*q_b*/, int64_t /*r_a*/, int64_t /*e_a*/) {
	return 0;  // stub
}

SslmForwardStatus EmbedEntry(int32_t /*token_id*/, int32_t /*vocab_size*/,
                              const int8_t* /*embed_weights*/, size_t /*hidden_size*/,
                              CarriedScale /*site_constant*/, int8_t* /*out_codes*/,
                              CarriedScale* /*out_scale*/) {
	return SslmForwardStatus::WorkspaceTooSmall;  // stub
}

}  // namespace superslm
