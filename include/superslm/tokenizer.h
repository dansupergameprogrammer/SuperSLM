// Integer-only byte-level BPE tokenizer over a loaded `.sslm`.
//
// Deterministic and dependency-free: NFC normalization and the \p{L}/\p{N}/\s
// property classes come from the pinned Unicode tables in the artifact's
// UnicodeTables section (never a platform Unicode or regex library — D-SLM13). The
// algorithm mirrors tools/convert_tokenizer.py's reference encode(), which is proven
// bit-for-bit against the upstream HF tokenizer (SuperSLM_Plan.md §10).
#ifndef SUPERSLM_TOKENIZER_H
#define SUPERSLM_TOKENIZER_H

#include "superslm/artifact.h"

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace superslm {

class TokenizerView {
public:
	TokenizerView() = default;

	// Bind to a loaded artifact by parsing its Tokenizer + UnicodeTables sections.
	// Returns false (and sets *err, if non-null) when either section is absent or
	// malformed; the artifact's integrity is already verified by the loader, so this
	// parse trusts the bytes and only sanity-checks structure.
	static bool Open(const SslmArtifact& artifact, TokenizerView& out, std::string* err);

	bool Ok() const noexcept { return ok_; }
	int32_t VocabSize() const noexcept { return vocab_count_; }

	// text -> token ids. A special-token's content appearing in the text is matched
	// (longest first) and emitted as its id; every other span is NFC-normalized,
	// pre-tokenized by the fixed Qwen/GPT pattern, byte-level encoded, and BPE-merged.
	// No BOS/EOS/chat markers are added — that is the caller's (or a template's) job.
	std::vector<int32_t> Encode(std::string_view text) const;

	// ids -> UTF-8 text: the byte-level bytes each token carries, concatenated and
	// interpreted as UTF-8 (invalid sequences pass through as replacement chars).
	std::string Decode(const std::vector<int32_t>& ids) const;

private:
	bool ok_ = false;
	int32_t vocab_count_ = 0;
	const SslmArtifact* artifact_ = nullptr; // sections are views into it; must outlive
	// Parsed table views are filled in by Open when the tokenizer is built.
};

} // namespace superslm

#endif // SUPERSLM_TOKENIZER_H
