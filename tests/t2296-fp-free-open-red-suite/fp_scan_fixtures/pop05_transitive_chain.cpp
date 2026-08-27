// T-2333/T-2326 (Curie) -- Sec7 dimension 11's fifth commissioning
// population: a synthetic multi-hop transitive-closure proof.
//
// PROVENANCE. The design's own text (Sec7 dim 11, fold round 6, D-SLM4335)
// specifies this population but names no execution artifact or file path for
// it anywhere in the design's 30+ fold rounds. Authored fresh (T-2326),
// unchanged in construction by T-2333 -- only the build toolchain changes
// (MSVC cl.exe + dumpbin, not clang), so this fixture's own real disassembly
// text can be read by the ratified `diagnostic_walk(roots, call_graph)`
// surface's own real call graph, matching population six's own toolchain
// exactly.
//
// CONSTRUCTION (design's own text, verbatim requirement): "root -> A -> B -> C,
// where C is three or more hops from the chosen root, contains a flagged
// instruction, and no shorter path from the root to C exists." A scan built on
// a genuine transitive walk (BFS/DFS over the full call graph) reaches
// FlaggedLeaf via Root; a scan built on a bounded-depth approximation (a
// one-hop-plus-one-hop check, or a hand-extended two-hop grep) does not, since
// FlaggedLeaf is unreachable within two hops of Root by construction (no
// shorter path exists: Root calls ONLY HopA, HopA calls ONLY HopB, HopB calls
// ONLY FlaggedLeaf -- no direct or two-hop edge from Root to FlaggedLeaf).
//
// DISPOSITION, per Sec7 dim 11's own fold-round-12 text: this population
// remains "genuinely valuable... a diagnostic that silently loses reachability
// ... misattributes which container a future reject belongs to" but "no
// longer commission[s] the gate" (the ACCEPT/REJECT verdict does not depend
// on the walk since fold round 8, only Sec7 dimensions 1-3's own "when does
// this site run" reasoning does).
//
// Build: MSVC cl.exe (/std:c++20 /O2 /EHsc), then dumpbin /disasm -- matching
// population six's own toolchain (T-2333), so both populations' own real
// disassembly text is read by the same ratified corpus-building surfaces
// (build_call_graph/diagnostic_walk).

#if defined(_MSC_VER)
#define T2333_NOINLINE __declspec(noinline)
#else
#define T2333_NOINLINE __attribute__((noinline))
#endif

extern "C" T2333_NOINLINE double FlaggedLeaf(double x, double y) {
    return x / y;  // the flagged instruction (a genuine divsd)
}

extern "C" T2333_NOINLINE double HopB(double x, double y) {
    return FlaggedLeaf(x, y) + 1.0;
}

extern "C" T2333_NOINLINE double HopA(double x, double y) {
    return HopB(x, y) * 2.0;
}

// Root calls ONLY HopA -- no direct or two-hop edge from Root to FlaggedLeaf
// exists anywhere in this translation unit.
extern "C" double Root(double x, double y) {
    // Stored through a volatile before returning: forces MSVC /O2 to emit a
    // genuine `call HopA` here rather than a tail `jmp HopA` (confirmed by
    // direct dumpbin inspection this session -- the straight-line `return
    // HopA(x, y);` form compiles to a tail jmp, which this population's own
    // reachability claim does not need to exercise; the call-graph builder
    // this fixture is read by (fold33-probe's own build_call_graph) only
    // recognises `call` edges, not tail-jmp -- filed as a residual against
    // that reference implementation in the case file, not silently worked
    // around by relying on jmp-edge support that is not yet specified).
    volatile double r = HopA(x, y);
    return r;
}
