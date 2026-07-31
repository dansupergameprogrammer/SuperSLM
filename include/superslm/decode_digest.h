// SuperSLM S3a's single-build parity oracle -- what is hashed
// (SuperSLM_S3a_WalkingSkeleton_Plan.md §10.1; §11 S3.6, whose own red-cell
// text is the first to require an emitted token be checked "in both
// digests," because S3.6 is the first sub-slot where a real token/logit
// sequence exists to hash. Distinct from the trace instrument
// (trace_hook.h), which records per-site chain arithmetic for
// bit-equality-to-reference (§12 criterion 2); these two digests are the
// PARITY GATE's own oracle (§12 criterion 3, §11 S3.8) — a much cheaper,
// coarser check computed over the decode loop's own visible output.
//
// §10.1 pins two digests per (artifact, prompt):
//   1. The token digest — SHA-256 over the count-prefixed little-endian
//      int32 sequence of emitted token ids.
//   2. The final-logit digest — SHA-256 over the count-prefixed
//      little-endian int32 final logit row at each generated position.
//
// "Count-prefixed" (both): the first four bytes are the element count
// (token digest) or the position count (logit digest), encoded little-endian
// int32, followed by the payload — never a bare concatenation with no
// length, which would let two different-length sequences that happen to
// share a prefix collide under a naive comparison of "the hash so far."
//
// Neither digest exists anywhere in this tree before S3.6 — grep confirms no
// prior sub-slot declares or builds one (§10 states the INTENT that both are
// "wired from the first commit"; the mechanism itself is first buildable
// once a real token/logit sequence exists to feed it, which S3.6 is the
// first sub-slot to produce). §10.2's exclusions (saturation count, band
// verdict, decode-step status, layer budget, decode-call count, any pointer
// or workspace byte) are properties of what these two functions do NOT take
// as input — they take only `tokens`/`logit_rows`, never a `SequenceLayerState`
// or a workspace, so an excluded field literally cannot reach the hash
// through this API, which is the exclusion table's own each-row claim made
// structural rather than a discipline the caller must remember.
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
