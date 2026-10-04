// T-3103 S0: load an attached G1 SCM1 section with the product's own SchemaMasksTable::Parse
// (include/superslm/schema_masks.h), then decide by exhaustive search whether a document is in the
// loaded token language. A document is accepted when some sequence of real vocabulary tokens spells its
// exact bytes, every token is permitted by the mask and the transition table at its state, and the walk
// ends in an accepting state. The search is memoized on (byte position, DFA state).
// Inputs: <scm1.bin> <vocab.hex> <vocab size> <accept.json> <reject.json>
//   vocab.hex: one line per token id, the token's raw bytes in hex (empty line for an empty token).
// Prints one JSON line. Exit 0 only when the section parses, intent_gate is present, the accept document has
// an accepting path and the reject document has none.
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iterator>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "superslm/schema_masks.h"

namespace
{
std::vector<uint8_t> ReadAll(const char* path)
{
	std::ifstream in(path, std::ios::binary);
	return std::vector<uint8_t>((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
}

std::vector<std::string> ReadVocab(const char* path)
{
	std::ifstream in(path);
	std::vector<std::string> vocab;
	std::string line;
	while (std::getline(in, line))
	{
		std::string bytes;
		for (size_t i = 0; i + 1 < line.size(); i += 2)
			bytes.push_back(static_cast<char>(std::stoul(line.substr(i, 2), nullptr, 16)));
		vocab.push_back(bytes);
	}
	return vocab;
}

struct Search
{
	const superslm::SchemaMasksTable& t;
	const superslm::SchemaEntry& e;
	const std::vector<std::string>& vocab;
	std::string doc;
	std::set<std::pair<size_t, uint32_t>> dead;
	size_t tokens = 0;

	bool Accepting(uint32_t s) const
	{
		for (uint32_t a = 0; a < e.accepting_count; ++a)
			if (superslm::schema_masks_detail::ReadLE32(e.accepting_le + 4 * a) == s) return true;
		return false;
	}

	bool From(size_t pos, uint32_t state, size_t depth)
	{
		if (pos == doc.size()) return Accepting(state);
		if (dead.count({pos, state})) return false;
		for (uint32_t id = 0; id < vocab.size(); ++id)
		{
			const std::string& tok = vocab[id];
			if (tok.empty() || tok.size() > doc.size() - pos || doc.compare(pos, tok.size(), tok) != 0) continue;
			if (!t.MaskBit(e, state, id)) continue;
			uint32_t next = 0;
			if (!t.Transition(e, state, id, &next)) continue;
			if (From(pos + tok.size(), next, depth + 1))
			{
				tokens = depth + 1 > tokens ? depth + 1 : tokens;
				return true;
			}
		}
		dead.insert({pos, state});
		return false;
	}
};
} // namespace

int main(int argc, char** argv)
{
	if (argc != 6)
	{
		std::fprintf(stderr, "usage: g1_attach_check <scm1.bin> <vocab.hex> <vocab_size> <accept.json> <reject.json>\n");
		return 2;
	}
	std::vector<uint8_t> data = ReadAll(argv[1]);
	std::vector<std::string> vocab = ReadVocab(argv[2]);
	const uint32_t vocab_size = static_cast<uint32_t>(std::stoul(argv[3]));
	superslm::SchemaMasksTable table;
	std::string err;
	if (vocab.size() != vocab_size || !superslm::SchemaMasksTable::Parse(data.data(), data.size(), vocab_size, table, &err))
	{
		std::printf("{\"parse\":false,\"vocab_lines\":%zu,\"error\":\"%s\"}\n", vocab.size(), err.c_str());
		return 1;
	}
	const superslm::SchemaEntry* e = table.ByName("intent_gate");
	if (e == nullptr)
	{
		std::printf("{\"parse\":true,\"intent_gate\":false}\n");
		return 1;
	}
	std::vector<uint8_t> acc = ReadAll(argv[4]);
	std::vector<uint8_t> rej = ReadAll(argv[5]);
	Search sa{table, *e, vocab, std::string(acc.begin(), acc.end()), {}, 0};
	Search sr{table, *e, vocab, std::string(rej.begin(), rej.end()), {}, 0};
	const bool accepted = sa.From(0, 0, 0);
	const bool rejected_has_path = sr.From(0, 0, 0);
	std::printf("{\"parse\":true,\"schemas\":%zu,\"states\":%u,\"transitions\":%u,\"accepting\":%u,"
		"\"accept_doc_in_language\":%s,\"accept_path_tokens\":%zu,\"reject_doc_in_language\":%s,"
		"\"reject_dead_pairs\":%zu}\n",
		table.Count(), e->state_count, e->transition_count, e->accepting_count, accepted ? "true" : "false",
		sa.tokens, rejected_has_path ? "true" : "false", sr.dead.size());
	return (accepted && !rejected_has_path) ? 0 : 1;
}
