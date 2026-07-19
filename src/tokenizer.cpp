#include "superslm/tokenizer.h"

namespace superslm {

// ---------------------------------------------------------------------------
// S1.3 RED-FIRST STUB. The byte-level BPE tokenizer is authored test-first: Curie's
// red suite loads the golden fixture and asserts Encode reproduces the upstream ids,
// and must fail red before Brunel builds the body. Until then Open reports the
// tokenizer is unbuilt and Encode/Decode yield nothing.
// (SuperSLM_Plan.md §10/§15; Curie owns the S1 tokenizer red suite.)
// ---------------------------------------------------------------------------

bool TokenizerView::Open(const SslmArtifact& /*artifact*/, TokenizerView& out,
                         std::string* err) {
	out = TokenizerView{};
	if (err) *err = "TokenizerView::Open: not implemented (S1.3 red-first stub)";
	return false;
}

std::vector<int32_t> TokenizerView::Encode(std::string_view /*text*/) const {
	return {};
}

std::string TokenizerView::Decode(const std::vector<int32_t>& /*ids*/) const {
	return {};
}

} // namespace superslm
