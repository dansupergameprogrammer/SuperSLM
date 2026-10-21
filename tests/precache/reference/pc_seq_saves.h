// Precached-prefixes plan, cell 9.9 [E5] (rev 1.18.1, audit G14 (c)): the sequence-save cases 1.13.1 writes
// and 1.14.0 must read, shared by the capture tool and the cell so both build each state the same way.
//
// capture_v1p13p1_seq_saves.cpp compiles this header against the frozen v1.13.1 tree (d697927; see
// tools/build_precache_v1p13p1_captures.sh) and the cell (tests/precache/unpinned/e5_unpinned.cpp) against
// the tree under work. Every verb used here exists unchanged from 1.13.0 to the release tree, so the one
// script runs on both. The cell rebuilds only the hint (the nested prefix chain) with it; the saved
// states, their bytes and their continuations are the 1.13.1 build's own output, so no code of the tree
// under work grades the cell.
//
// The cases (plan 9.9): SSB5 saves of a whole_reserve sequence, and SSB6 saves of budget sequences: a
// root, an adopter of a nested prefix with origin > 0, and one mid-turn, each with and without a bound
// schema. "Mid-turn" is read here as saved between the layer calls of one decode step (the blob's
// layer_index > 0: PKV's MIDTOKEN resting state), the one sequence state the other three never save;
// the plan does not define the word, and the red-suite record routes that reading to the planner. A bound
// state's continuation is graded in outcome form (Continue below): plan 9.9's "64 tokens" cannot hold on
// pkv_def's one schema, which dead-ends after one token; that is routed too.
#ifndef SUPERSLM_PC_SEQ_SAVES_H
#define SUPERSLM_PC_SEQ_SAVES_H

#include "superslm/sslm_abi.h"

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace pcsave {

enum class Kind { kWholeReserve, kRoot, kNestedAdopter, kMidToken };

struct Case {
	const char* name;
	Kind kind;
	bool schema;
	const char* magic;  // what 1.13.1 writes for this state (plan 9.9)
};

constexpr Case kCases[] = {
    {"ssb5_whole_reserve", Kind::kWholeReserve, false, "SSB5"},
    {"ssb5_whole_reserve_schema", Kind::kWholeReserve, true, "SSB5"},
    {"ssb6_root", Kind::kRoot, false, "SSB6"},
    {"ssb6_root_schema", Kind::kRoot, true, "SSB6"},
    {"ssb6_nested_adopter", Kind::kNestedAdopter, false, "SSB6"},
    {"ssb6_nested_adopter_schema", Kind::kNestedAdopter, true, "SSB6"},
    {"ssb6_mid_turn", Kind::kMidToken, false, "SSB6"},
    {"ssb6_mid_turn_schema", Kind::kMidToken, true, "SSB6"},
};
constexpr size_t kCaseCount = sizeof kCases / sizeof kCases[0];

constexpr const char* kSchemaName = "g5_minimal_one_field";  // pkv_def's one schema
constexpr int32_t kSeqBudget = 160;     // room for the prompt, 4 decoded tokens and the 64-token continuation
constexpr int32_t kParentBudget = 192;   // the nested chain's root
constexpr int32_t kChildBudget = 64;
constexpr int32_t kParentPrompt = 150;   // past one page at pkv_def's rule B (128), so the hint shares a page
constexpr int32_t kChildPrompt = 20;     // the nested prefix: origin = 150 > 0
constexpr int32_t kSeqPrompt = 40;
constexpr int32_t kAdopterPrompt = 8;
constexpr int32_t kDecodeBeforeSave = 4;
constexpr int32_t kContinue = 64;        // plan 9.9: 64 steps equal to the pinned continuation
constexpr uint32_t kPoolPages = 96;
constexpr int32_t kVocab = 256;

// Deterministic prompt streams (pkv_scenarios.h's LCG, restated so this header stands alone).
inline std::vector<int32_t> Prompt(uint32_t seed, int32_t n) {
	std::vector<int32_t> t(static_cast<size_t>(n));
	uint32_t x = seed * 2654435761u + 12345u;
	for (int32_t i = 0; i < n; ++i) {
		x = x * 1664525u + 1013904223u;
		t[static_cast<size_t>(i)] = static_cast<int32_t>((x >> 8) % static_cast<uint32_t>(kVocab));
	}
	return t;
}

struct Run {
	sslm_model model = nullptr;
	sslm_kv_pool pool = nullptr;
	std::string fail;
	bool Ok(sslm_status s, const char* what) {
		if (s == SSLM_OK) return true;
		if (fail.empty()) fail = std::string(what) + " -> " + std::to_string(static_cast<int>(s));
		return false;
	}
};

inline bool PrefillSeq(Run& r, sslm_seq s, const std::vector<int32_t>& t) {
	size_t at = 0;
	while (at < t.size()) {
		int32_t consumed = 0;
		if (!r.Ok(sslm_prefill(r.model, s, t.data() + at, static_cast<int32_t>(t.size() - at), 8,
		                       SSLM_SPAN_PROMPT, nullptr, &consumed),
		          "prefill"))
			return false;
		if (consumed <= 0) return r.Ok(SSLM_INVALID_ARGUMENT, "prefill consumed 0");
		at += static_cast<size_t>(consumed);
	}
	return true;
}

inline bool PrefillPrefix(Run& r, sslm_prefix p, const std::vector<int32_t>& t) {
	size_t at = 0;
	while (at < t.size()) {
		int32_t consumed = 0;
		if (!r.Ok(sslm_prefix_prefill(r.model, p, t.data() + at, static_cast<int32_t>(t.size() - at), 8,
		                              SSLM_SPAN_PROMPT, nullptr, &consumed),
		          "prefix_prefill"))
			return false;
		if (consumed <= 0) return r.Ok(SSLM_INVALID_ARGUMENT, "prefix_prefill consumed 0");
		at += static_cast<size_t>(consumed);
	}
	return true;
}

// One decode-step call, one layer: returns the status and writes the token (-1 while mid-token).
inline sslm_status StepOneLayer(Run& r, sslm_seq s, int32_t* tok) {
	sslm_decode_params p{};
	p.layer_budget = 1;
	sslm_seq b[1] = {s};
	*tok = -1;
	return sslm_decode_step(r.model, b, 1, &p, nullptr, tok);
}

// `n` greedy decode steps, one layer per call, one outcome per step: the token, the schema dead end's -2,
// or -1000 - status for a refused call. Neither a dead end nor a refusal stops the run (paged-KV's 8.16
// outcome form, g1e_cell_9_6.cpp's StepOutcomes): pkv_def's one schema admits a single token and then
// dead-ends, so a bound state has no 64-token greedy continuation, and its outcomes are what both builds
// must agree on. An unbound state's outcomes are all tokens.
inline std::vector<int32_t> Continue(Run& r, sslm_seq s, int32_t n) {
	std::vector<int32_t> out;
	for (int32_t i = 0; i < n; ++i) {
		int32_t tok = -1;
		sslm_status st = SSLM_OK;
		for (int guard = 0; tok == -1 && st == SSLM_OK && guard < 4096; ++guard) st = StepOneLayer(r, s, &tok);
		out.push_back(st != SSLM_OK ? -1000 - static_cast<int32_t>(st) : tok);
	}
	return out;
}

// True when every outcome is a token (no dead end, no refusal).
inline bool AllTokens(const std::vector<int32_t>& outcomes) {
	for (int32_t t : outcomes)
		if (t < 0) return false;
	return true;
}

inline bool SaveSeq(Run& r, sslm_seq s, std::vector<uint8_t>* blob) {
	size_t need = 0;
	if (sslm_seq_save(s, nullptr, &need) != SSLM_BUFFER_TOO_SMALL || need == 0)
		return r.Ok(SSLM_INVALID_ARGUMENT, "seq_save size query");
	blob->assign(need, 0);
	size_t wrote = blob->size();
	if (!r.Ok(sslm_seq_save(s, blob->data(), &wrote), "seq_save")) return false;
	blob->resize(wrote);
	return true;
}

inline bool LookupSchema(Run& r, sslm_schema* schema) {
	*schema = nullptr;
	return r.Ok(sslm_schema_lookup(r.model, kSchemaName, schema), "schema_lookup") && *schema;
}

// The nested chain: a budget root prefix of kParentPrompt tokens (schema bound before its prefill when
// `schema`), frozen; a child begun from it (origin = kParentPrompt, the binding inherited) with
// kChildPrompt more tokens, frozen. The child is the adopter's prefix and, on the release tree, the hint.
inline bool NestedChain(Run& r, bool schema, sslm_prefix* parent, sslm_prefix* child) {
	*parent = nullptr;
	*child = nullptr;
	if (!r.Ok(sslm_prefix_begin_budgeted(r.model, &r.pool, kParentBudget, parent), "prefix_begin_budgeted"))
		return false;
	if (schema) {
		sslm_schema sc = nullptr;
		if (!LookupSchema(r, &sc) || !r.Ok(sslm_prefix_set_schema(*parent, sc), "prefix_set_schema")) return false;
	}
	if (!PrefillPrefix(r, *parent, Prompt(91, kParentPrompt)) || !r.Ok(sslm_prefix_freeze(*parent), "freeze parent"))
		return false;
	if (!r.Ok(sslm_prefix_begin_from(*parent, kChildBudget, child), "prefix_begin_from")) return false;
	return PrefillPrefix(r, *child, Prompt(92, kChildPrompt)) && r.Ok(sslm_prefix_freeze(*child), "freeze child");
}

// The state at the save point. `parent` and `child` receive the nested chain (kNestedAdopter only); the
// caller releases the sequence before the prefixes.
inline bool Build(Run& r, const Case& c, sslm_seq* seq, sslm_prefix* parent, sslm_prefix* child) {
	*seq = nullptr;
	*parent = nullptr;
	*child = nullptr;
	sslm_schema sc = nullptr;
	if (c.schema && !LookupSchema(r, &sc)) return false;
	if (c.kind == Kind::kWholeReserve) {
		if (!r.Ok(sslm_seq_create(r.model, &r.pool, seq), "seq_create")) return false;
	} else {
		if (!r.Ok(sslm_seq_create_budgeted(r.model, &r.pool, kSeqBudget, seq), "seq_create_budgeted")) return false;
	}
	if (c.schema && !r.Ok(sslm_seq_set_schema(*seq, sc), "seq_set_schema")) return false;
	if (c.kind == Kind::kNestedAdopter) {
		if (!NestedChain(r, c.schema, parent, child)) return false;
		if (!r.Ok(sslm_seq_adopt_prefix(*seq, *child), "adopt nested prefix")) return false;
		if (!PrefillSeq(r, *seq, Prompt(93, kAdopterPrompt))) return false;
	} else {
		if (!PrefillSeq(r, *seq, Prompt(94, kSeqPrompt))) return false;
	}
	// Unbound: kDecodeBeforeSave greedy tokens first. Bound: none, since pkv_def's schema dead-ends after its
	// one admitted token; the bound state is saved after the prompt (the mid-turn one inside that token).
	if (!c.schema) {
		const std::vector<int32_t> before = Continue(r, *seq, kDecodeBeforeSave);
		if (before.size() != static_cast<size_t>(kDecodeBeforeSave) || !AllTokens(before))
			return r.Ok(SSLM_INVALID_ARGUMENT, "decode before save");
	}
	if (c.kind == Kind::kMidToken) {
		// A bound state is still ready after its prompt: its pending (admitted) token is emitted first, as
		// PKV's EnterMidToken does, so the one-layer step below starts the next token.
		if (c.schema) {
			const std::vector<int32_t> pending = Continue(r, *seq, 1);
			if (!AllTokens(pending)) return r.Ok(SSLM_INVALID_ARGUMENT, "mid-turn: the pending token");
		}
		int32_t tok = -1;
		if (!r.Ok(StepOneLayer(r, *seq, &tok), "mid-turn layer step")) return false;
		if (tok >= 0) return r.Ok(SSLM_INVALID_ARGUMENT, "mid-turn step finished a token (one-layer model?)");
	}
	return true;
}

}  // namespace pcsave

#endif  // SUPERSLM_PC_SEQ_SAVES_H
