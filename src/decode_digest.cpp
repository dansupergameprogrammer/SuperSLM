// §10.1's real count-prefixed little-endian SHA-256 composition (T-1389;
// built against Claude/Curie/superslm-s3.6-head-and-greedy-decode-test-
// design-2026-07-31.md's red suite, replacing the 32-zero-byte stub the
// test-design pass landed). Both functions assemble the exact byte layout
// decode_digest.h's own doc comment states -- a little-endian int32 count,
// then the little-endian int32 payload in order -- into one buffer, and hash
// it with the already-shipped, already-known-answer-tested Sha256Hash
// (sha256.h). Neither function reuses any part of the test suite's own
// independent oracle (`ExpectedTokenDigest`/`ExpectedFinalLogitDigest`,
// tests/test_main.cpp) -- the two are grounded in §10.1's text separately, so
// a defect in one is not masked by the other sharing it.
#include "superslm/decode_digest.h"

#include <vector>

#include "superslm/sha256.h"

namespace superslm {

namespace {

// Little-endian int32 append -- the one byte-layout primitive both digests
// share (§10.1's own "count-prefixed little-endian int32" text, both items).
void AppendLE32(std::vector<uint8_t>& out, int32_t v) {
	const uint32_t u = static_cast<uint32_t>(v);
	out.push_back(static_cast<uint8_t>(u & 0xFF));
	out.push_back(static_cast<uint8_t>((u >> 8) & 0xFF));
	out.push_back(static_cast<uint8_t>((u >> 16) & 0xFF));
	out.push_back(static_cast<uint8_t>((u >> 24) & 0xFF));
}

}  // namespace

void ComputeTokenDigest(const int32_t* tokens, size_t token_count, uint8_t out_digest[32]) {
	std::vector<uint8_t> bytes;
	bytes.reserve(4 + token_count * 4);
	AppendLE32(bytes, static_cast<int32_t>(token_count));
	for (size_t i = 0; i < token_count; ++i) AppendLE32(bytes, tokens[i]);
	Sha256Hash(bytes.data(), bytes.size(), out_digest);
}

void ComputeFinalLogitDigest(const int32_t* logit_rows, size_t num_positions,
                              size_t vocab_size, uint8_t out_digest[32]) {
	std::vector<uint8_t> bytes;
	const size_t total = num_positions * vocab_size;
	bytes.reserve(4 + total * 4);
	AppendLE32(bytes, static_cast<int32_t>(num_positions));
	for (size_t i = 0; i < total; ++i) AppendLE32(bytes, logit_rows[i]);
	Sha256Hash(bytes.data(), bytes.size(), out_digest);
}

}  // namespace superslm
