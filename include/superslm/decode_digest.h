// SuperSLM decode parity digests — a cheap, coarse oracle computed over the decode loop's own
// visible output, distinct from the trace instrument (trace_hook.h), which records per-site
// chain arithmetic for bit-equality checking against a reference.
//
// Two digests are pinned per (artifact, prompt):
//   1. The token digest — SHA-256 over the count-prefixed little-endian int32 sequence of
//      emitted token ids.
//   2. The final-logit digest — SHA-256 over the count-prefixed little-endian int32 final
//      logit row at each generated position.
//
// "Count-prefixed" (both): the first four bytes are the element count (token digest) or the
// position count (logit digest), encoded little-endian int32, followed by the payload — never
// a bare concatenation with no length, which would let two different-length sequences that
// happen to share a prefix collide under a naive comparison of "the hash so far."
//
// Both functions take only `tokens`/`logit_rows` as input, never a sequence's internal state or
// a workspace — an excluded field (saturation count, band verdict, decode-step status, layer
// budget, decode-call count, any pointer or workspace byte) literally cannot reach the hash
// through this API.
#ifndef SUPERSLM_DECODE_DIGEST_H
#define SUPERSLM_DECODE_DIGEST_H

#include <cstddef>
#include <cstdint>

namespace superslm {

// §10.1 item 1. `tokens` has `token_count` elements. Byte layout: 4
// little-endian bytes encoding `token_count` (as int32; caller-ensures
// `token_count` fits int32 — the same magnitude every other count in this
// tree's format is bound to, docs/sslm_format.md), then `token_count` groups
// of 4 little-endian bytes, one per token id, in sequence order (position 0
// first). `token_count == 0` is legal (the empty sequence's own digest,
// SHA-256 of the 4-byte zero count alone) and is not this function's call to
// reject — an empty decode is a caller-level question (§9.1's own loop
// contract), not this function's.
void ComputeTokenDigest(const int32_t* tokens, size_t token_count, uint8_t out_digest[32]);

// §10.1 item 2. `logit_rows` has `num_positions * vocab_size` elements,
// row-major: row `p` (0-indexed within this call, `p` in `[0, num_positions)`)
// is the full int32 logit row at the p-th GENERATED position — never a
// prompt position, which the reference's own trace never emits a logit row
// for either (§6.4's "Logits" step runs once per produced token, not once
// per prompt token). Byte layout: 4 little-endian bytes encoding
// `num_positions` (as int32), then, for each position in order, `vocab_size`
// groups of 4 little-endian bytes, one per logit — never length-prefixed a
// second time per row, because `vocab_size` is a fact of the artifact both
// sides of any comparison already know (CFG1's own field), not part of the
// variable-length axis §10.1's count-prefix protects.
void ComputeFinalLogitDigest(const int32_t* logit_rows, size_t num_positions,
                              size_t vocab_size, uint8_t out_digest[32]);

}  // namespace superslm

#endif  // SUPERSLM_DECODE_DIGEST_H
