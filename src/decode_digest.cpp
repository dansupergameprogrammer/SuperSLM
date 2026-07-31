// STUB (T-1389): declared and stubbed by the test-design pass authoring the
// S3.6 red suite (Claude/Curie/superslm-s3.6-head-and-greedy-decode-test-
// design-2026-07-31.md; decode_digest.h). Neither function computes: each
// writes 32 zero bytes to `out_digest`, a value SHA-256 never produces for
// any input (Sha256Hash's own known-answer vectors, tests/test_main.cpp
// TestSha256KnownVectors, confirm this), so a cell comparing against any
// real expected digest fails loudly rather than by chance. The build seat
// replaces both bodies with the real count-prefixed little-endian
// SHA-256 composition decode_digest.h's own doc comment states.
#include "superslm/decode_digest.h"

#include <cstring>

namespace superslm {

void ComputeTokenDigest(const int32_t* /*tokens*/, size_t /*token_count*/,
                         uint8_t out_digest[32]) {
	std::memset(out_digest, 0, 32);
}

void ComputeFinalLogitDigest(const int32_t* /*logit_rows*/, size_t /*num_positions*/,
                              size_t /*vocab_size*/, uint8_t out_digest[32]) {
	std::memset(out_digest, 0, 32);
}

}  // namespace superslm
