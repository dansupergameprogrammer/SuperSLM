// Internal (not installed, not exported): a read-only view of the byte tables a parsed TOK1 carries,
// for the engine's own consumers inside this library. Schema fast-decode plan (rev 1.2.1) §5.1: the
// forced-next table is built at model map from the stored token DFA and these tables (each token's
// bytes, the single-byte token of each byte, the special ids), with Encode for next_forced. The
// tables are the TokenizerView's own (src/tokenizer.cpp parses them once); this header only names
// them, so the table builder never re-parses TOK1.
#ifndef SUPERSLM_SRC_TOKENIZER_TABLES_H
#define SUPERSLM_SRC_TOKENIZER_TABLES_H

#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "superslm/tokenizer.h"

namespace superslm {
namespace internal {

struct TokenizerTables {
	const uint32_t* byte_to_id = nullptr;    // 256 entries: the single-byte token of each byte
	const uint8_t* vocab_offsets_le = nullptr;  // vocab_count + 1 LE u32 offsets into vocab_blob
	const uint8_t* vocab_blob = nullptr;
	uint32_t vocab_count = 0;                // ids at and above have no bytes
	const std::vector<std::pair<std::string, int32_t>>* specials = nullptr;  // (content, id)
};

// False (and *out untouched) when the view holds no parsed tokenizer. The pointers stay valid for
// the view's lifetime, which is its artifact's. Does not allocate or throw.
bool ReadTokenizerTables(const TokenizerView& view, TokenizerTables* out) noexcept;

}  // namespace internal
}  // namespace superslm

#endif  // SUPERSLM_SRC_TOKENIZER_TABLES_H
