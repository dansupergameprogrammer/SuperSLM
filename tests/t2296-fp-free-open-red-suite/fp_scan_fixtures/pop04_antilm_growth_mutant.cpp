// T-2326 (Curie) -- Sec7 dimension 11's fourth commissioning population: a
// mutant that reintroduces std::unordered_map-shaped growth on the
// damped_greedy_antilm.cpp `tables_`/`counts` containers, reached via
// sslm_seq_restore (fold round 5, D-SLM4310, closing T-2270's own fracture --
// Claude/Loki/t2270-probe/: 26 FP instructions executing on this exact path
// while the pre-fold-5 scan reported clean, because its own root-and-closure
// walk never reached sslm_abi.cpp's `sslm_seq_restore`).
//
// PROVENANCE. T-2270's own probe directory carries analysis of the REAL,
// then-compiled damped_greedy_antilm.cpp (a disassembly capture,
// antilm-update-disasm.txt) rather than a standalone constructed .cpp -- the
// strike worked directly against production source, which this design's own
// remedy (§3.6) has since replaced: reading src/damped_greedy_antilm.cpp today
// (this session, read-only per this ticket's field 3 -- "the rest of the
// SuperSLM engine tree") confirms `tables_` is now
// std::vector<superslm::detail::GrowableContextMap<ContextEntry>> and
// ContextEntry::counts is superslm::detail::GrowableIntMap<...> (that file's
// own line-33 comment: "T-2296 (2026-08-26) replaced tables_/counts"). The
// mutant this population's own must-reject claim needs is therefore a
// deliberate REGRESSION relative to today's clean state, not a fixture that
// exists anywhere in the tree today.
//
// This file is a fresh, standalone reproduction of the PRE-T-2296 shape --
// never a modification of the real file (out of this ticket's writable scope,
// and Curie never implements, per Implement/Curie/Curie.md's own charter) --
// mirroring the real ContextEntry's own two fields (a per-candidate-token
// count map, keyed by an int32-ish token id, growing across many distinct
// contexts and many distinct candidates within a context) with plain
// std::unordered_map in place of the replacement's GrowableIntMap/
// GrowableContextMap, and populated through a restore-shaped replay loop (many
// contexts, each with many candidate tokens) matching sslm_seq_restore's own
// call pattern into AntiLmUpdate (src/damped_greedy_antilm.cpp:121-128).
//
// Build: MSVC cl.exe -- std::unordered_map, same STL-version constraint as
// pop01_pop03 (clang 18.1.8 + MSVC STL headers refuses below clang 19).

#include <cstdint>
#include <unordered_map>
#include <vector>

struct ContextEntryMutant {
    int64_t total = 0;
    std::unordered_map<int32_t, int64_t> counts;  // pre-T-2296 shape
};

// Mirrors sslm_seq_restore's own replay: many distinct contexts (one
// std::unordered_map entry each), each accumulating many distinct candidate
// tokens -- the exact shape whose growth this population's own claim requires
// be caught, reached via a "restore" call rather than live decode-time
// generation, per Sec7 dimension 1's own (c) leg naming this coordinate.
extern "C" __declspec(dllexport)
void AntiLmRestoreGrowthMutant(const int64_t* contexts, const int32_t* tokens,
                                size_t n_contexts, size_t n_tokens_per_context) {
    std::unordered_map<int64_t, ContextEntryMutant> tables;
    for (size_t c = 0; c < n_contexts; ++c) {
        ContextEntryMutant& entry = tables[contexts[c]];  // pre-T-2296: operator[] growth
        for (size_t t = 0; t < n_tokens_per_context; ++t) {
            entry.counts[tokens[t]] += 1;
            entry.total += 1;
        }
    }
}
