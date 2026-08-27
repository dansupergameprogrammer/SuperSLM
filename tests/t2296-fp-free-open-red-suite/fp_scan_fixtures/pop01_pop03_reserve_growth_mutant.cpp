// T-2326 (Curie) -- Sec7 dimension 11's first and third commissioning
// populations: a mutant that reintroduces reserve()-shaped std::unordered_map/
// set growth, scanned byte-for-byte.
//
// PROVENANCE. The design's own text names two distinct sites for this shape:
//   - Population one (Sec4.1's own must-reject text, "TE-32's own Part B/C"):
//     the ORIGINAL two replacement sites (tokenizer.cpp's `merges`, model.cpp's
//     `seen_names`) -- TE-32's own historical artifact lives at
//     Claude/Loki/te32-probe/, which is OUTSIDE this session's granted
//     read-only scope (Claude/Loki/te32-probe/ is not among the probe
//     directories this ticket names). Not reproduced byte-for-byte; see this
//     campaign's case file for the disposition.
//   - Population three (fold round 3, D-SLM4308, closing T-2268/D-SLM4260-4267
//     -- Claude/Loki/t2268-probe/t2268_insert_growth_fp.cpp, its own "seqreg"
//     leg): the EXTENDED site sslm_abi.cpp's `g_live_seqs`, an
//     std::unordered_set<void*> grown by plain insert() and shrunk by erase(),
//     never reserved explicitly -- T-2268's own finding was that reserve() is
//     one entry point into the mechanism (max_load_factor()'s own float
//     divide/ceil inside libstdc++/MSVC-STL's rehash-sizing path), not the only
//     one: an ordinary insert() sequence that grows the bucket array also
//     reaches it.
//
// This file realizes population three's own construction directly (the
// insert()-driven growth T-2268 struck), reproduced fresh rather than reusing
// a committed binary (this design's own git-archive-only sourcing discipline,
// Claude/Vitruvius/t2265-superslm-fp-free-open-design-2026-08-24.md Sec4.1
// throughout) -- and, by the identical mechanism, discharges population one's
// own claim at the level that actually matters to a byte-level scan: the scan
// operates on compiled machine code and does not distinguish "this container is
// named g_live_seqs in sslm_abi.cpp" from "this container is named merges in
// tokenizer.cpp" -- both compile to the identical std::unordered_map/set
// rehash-sizing machinery, and BOTH must-reject constructions exercise the same
// classifier property (checks (A)/(B) rejecting the SSE-class divide/ceil
// instructions that machinery lowers to). The population that would remain
// open if this file's own construction passed but TE-32's own historical
// artifact failed is a claim about WHICH SOURCE FILE the mutant lives in, not
// about the scan's own mechanism -- filed as its own residual in the case file,
// not silently treated as fully commissioned.
//
// Build: MSVC cl.exe (/std:c++20 /O2 /EHsc), matching T-2268's own build.bat.
// clang was tried first and rejected for this file specifically: clang 18.1.8
// targeting *-windows-msvc pulls MSVC's own STL headers, and MSVC's STL (14.44)
// refuses any clang below 19.0.0 (confirmed by direct attempt, error STL1000) --
// std::unordered_set has no STL-free equivalent to fall back to the way
// pop07_fpblind.cpp's std::floor/std::fma did, so this fixture is MSVC-only,
// gated (SKIP, not FAIL) when no VsDevCmd.bat is found.

#include <cstdint>
#include <cstdlib>
#include <unordered_set>

// Mirrors sslm_abi.cpp's own g_live_seqs shape: a heap-pointer set grown by
// plain insert() and shrunk by erase(), never reserve()'d.
extern "C" __declspec(dllexport) void GLiveSeqsGrowthMutant(uint64_t n) {
    std::unordered_set<void*> live;
    for (uint64_t i = 0; i < n; ++i) {
        void* p = std::malloc(64);
        live.insert(p);
        if ((i % 7) == 6) {
            live.erase(p);
            std::free(p);
        }
    }
    for (void* p : live) std::free(p);
}

// The original two sites' own shape (tokenizer.cpp's merges / model.cpp's
// seen_names): reserve() then a bounded emplace loop -- population one's own
// defining property, realized here rather than left unbuilt.
extern "C" __declspec(dllexport) void MergesReserveGrowthMutant(uint64_t n) {
    std::unordered_set<uint64_t> merges;
    merges.reserve(n);
    for (uint64_t i = 0; i < n; ++i) merges.insert(i * 2654435761ULL);
}
