// T-2326 (Curie) -- Sec7 dimension 11's sixth commissioning population:
// translation-unit-set desync detection (fold round 6, D-SLM4335). See
// pop06_tu_b_added.cpp's own header for the full construction; this is the
// first of the mini build's two translation units -- the one every scan
// enumeration (hand-maintained or genuinely derived) already includes.
//
// PROVENANCE: no execution artifact exists in this design's own fold history
// for this population, identically to population five -- see
// pop05_transitive_chain.cpp's own header. Authored fresh.

extern "C" double CalleeInOtherTU(double x, double y);

extern "C" double Root(double x, double y) {
    return CalleeInOtherTU(x, y);
}
