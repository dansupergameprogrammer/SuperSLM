// T-2326 (Curie) -- Sec7 dimension 11's fifth commissioning population: a
// synthetic multi-hop transitive-closure proof.
//
// PROVENANCE. The design's own text (Sec7 dim 11, fold round 6, D-SLM4335)
// specifies this population but names no execution artifact or file path for
// it anywhere in the design's 30 fold rounds -- unlike populations seven
// through fourteen, which each cite a probe directory this design's own fold
// history actually built and ran. This population was demoted to a
// diagnostic-only role at fold round 8 (D-SLM4369) before it was ever
// realized as code: the transitive-closure walk it would commission stopped
// deciding the instrument's own pass/fail verdict at that fold (membership is
// now the closed symbol table, Sec4.1), so nothing in this design's own
// history built it. Authored fresh here, never a reproduction of an existing
// artifact.
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
// on the walk since fold round 8, only §7 dimensions 1-3's own "when does this
// site run" reasoning does). This test's own assertion is scoped accordingly
// -- see test_check_fp_free_scan.py's own docstring for population five.
//
// Build: clang, either ELF or COFF target -- no STL, no toolchain constraint.

extern "C" __attribute__((noinline)) double FlaggedLeaf(double x, double y) {
    return x / y;  // the flagged instruction (a genuine divsd)
}

extern "C" __attribute__((noinline)) double HopB(double x, double y) {
    return FlaggedLeaf(x, y) + 1.0;
}

extern "C" __attribute__((noinline)) double HopA(double x, double y) {
    return HopB(x, y) * 2.0;
}

// Root calls ONLY HopA -- no direct or two-hop edge from Root to FlaggedLeaf
// exists anywhere in this translation unit.
extern "C" double Root(double x, double y) {
    return HopA(x, y);
}
