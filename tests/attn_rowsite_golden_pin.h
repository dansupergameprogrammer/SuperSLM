// GENERATED FILE. Do not hand-edit.
//
// Produced by tools/gen_attn_rowsite_golden.cpp built against the v1.9.0 tag's library (the
// normative per-element code), over tests/support/rowsite_cases.h's and
// tests/support/attention_cases.h's fixed input sets, one hash per slice. Attention and per-row
// sites plan, §3.3 evidence 3, coverage cell 6.3. Re-running the generator against v1.9.0 must
// reproduce this file byte-for-byte.
#ifndef SUPERSLM_TESTS_ATTN_ROWSITE_GOLDEN_PIN_H
#define SUPERSLM_TESTS_ATTN_ROWSITE_GOLDEN_PIN_H

#include <cstdint>

namespace superslm_test {

// Slice S1: RmsNormSite, MlpActSite and ResidualReconcileSite over RunRowTableCases.
inline constexpr const char* kAttnRowsiteS1GoldenHash =
    "8836d5eb32a4badb492a8bcdf11e00222ad59a1e4b98013a3b8cb0c059d98ec8";
inline constexpr uint64_t kAttnRowsiteS1GoldenValues = 634120ULL;

// Slice S2: GemmProbQ15Accumulate over RunProbVCases.
inline constexpr const char* kAttnRowsiteS2GoldenHash =
    "b0d1a6cd065347e799e5bb9857ce5db1f51ff351c8d4edde22896f11974506ed";
inline constexpr uint64_t kAttnRowsiteS2GoldenValues = 30100ULL;

// Slice S3: RequantChainChecked's element loop over RunRequantRowCases.
inline constexpr const char* kAttnRowsiteS3GoldenHash =
    "3e3abed7c746191e8745c89ad38019076eff290aa7f4ffb57fb51c4527fdb3b9";
inline constexpr uint64_t kAttnRowsiteS3GoldenValues = 3567018ULL;

}  // namespace superslm_test

#endif  // SUPERSLM_TESTS_ATTN_ROWSITE_GOLDEN_PIN_H
