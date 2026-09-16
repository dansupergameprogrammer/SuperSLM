// GENERATED FILE. Do not hand-edit.
// Produced by tests/reference/t2704_residual_oracle.py.
// Re-running the generator must reproduce this file byte-for-byte.
#ifndef SUPERSLM_TESTS_SSLM_T2704_RESIDUAL_ORACLE_FIXTURES_H
#define SUPERSLM_TESTS_SSLM_T2704_RESIDUAL_ORACLE_FIXTURES_H

#include <cstdint>

namespace superslm_test {
struct T2704ResidualOracleRow {
  int64_t branch_m, branch_e, stream_m, stream_e;
  int8_t branch_code, stream_code;
  int64_t selected_magnitude, normalized_denominator, normalization_shift, reciprocal;
  int64_t selected_direct, nonselected_raw, nonselected_oriented, wide_sum;
};
inline constexpr T2704ResidualOracleRow kT2704M1ExactTie = {
  1LL, -17LL, 1LL, -17LL, -127, -127,
  1LL, 1073741824LL, 30LL, 4294967296LL,
  -127LL, -127LL, -127LL, -254LL,
};
struct T2704ResidualRetryRow {
  int64_t branch_m, branch_e, stream_m, stream_e;
  int8_t branch_code, stream_code;
  int64_t fine_normalized_denominator, fine_normalization_shift, fine_reciprocal;
  int64_t coarse_normalized_denominator, coarse_normalization_shift, coarse_reciprocal;
  int64_t coarse_raw, coarse_wide;
};
inline constexpr T2704ResidualRetryRow kT2704FineRejectCoarseCommit = {
  1073741824LL, 56LL, 1392366989LL, -40LL, 127, 0,
  1392366989LL, 0LL, 3312119617LL,
  1073741824LL, 0LL, 4294967296LL,
  0LL, 127LL,
};
}  // namespace superslm_test

#endif  // SUPERSLM_TESTS_SSLM_T2704_RESIDUAL_ORACLE_FIXTURES_H
