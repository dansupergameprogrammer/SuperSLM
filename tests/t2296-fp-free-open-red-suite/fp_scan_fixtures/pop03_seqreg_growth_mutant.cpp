// T-2333 (Curie) -- Sec7 dimension 11's THIRD commissioning population: a
// mutant that reintroduces reserve()-shaped std::unordered_set growth,
// scanned byte-for-byte.
//
// PROVENANCE (fold round 3, D-SLM4308, closing T-2268/D-SLM4260-4267 --
// Claude/Loki/t2268-probe/t2268_insert_growth_fp.cpp, its own "seqreg" leg):
// the EXTENDED site sslm_abi.cpp's `g_live_seqs`, an std::unordered_set<void*>
// grown by plain insert() and shrunk by erase(), never reserved explicitly --
// T-2268's own finding was that reserve() is one entry point into the
// mechanism (max_load_factor()'s own float divide/ceil inside libstdc++/
// MSVC-STL's rehash-sizing path), not the only one: an ordinary insert()
// sequence that grows the bucket array also reaches it.
//
// NOTE (T-2333): this file previously ALSO carried a fresh substitute for
// population ONE (the original two sites, tokenizer.cpp/model.cpp), authored
// when Claude/Loki/te32-probe/ was outside T-2326's own granted read-only
// scope. That scope is now granted (T-2333's own brief, item 5); population
// one is realized as its own, faithful reproduction of TE-32's own
// construction at pop01_te32_reserve.cpp, and the placeholder previously
// here (`MergesReserveGrowthMutant`) is removed rather than left stale.
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
