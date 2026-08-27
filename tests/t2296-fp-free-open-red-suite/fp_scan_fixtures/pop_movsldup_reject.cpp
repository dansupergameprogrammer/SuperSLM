// T-2342 (Curie) -- the boundary Poirot's own remedy names alongside the
// movsd fix: "for symmetry with the AVX half already present, movsldup/
// movshdup are NOT movement-only and must stay out." Both duplicate one lane
// of a packed-single value into its neighbour -- a real data-rearrangement
// operation on floating-point-typed lanes, not a pure copy the way movsd/
// movss/movaps are (the design's own VEC_MOVE_ALLOW rationale: "pure data
// movement... with no arithmetic, comparison, rounding, or conversion
// semantics" -- movsldup/movshdup select and duplicate a LANE, which is a
// real rearrangement operation, not a copy of the whole register). If a
// future repair widens VEC_MOVE_ALLOW to admit movsd by a loose pattern
// (e.g. "any mnemonic starting with movs") rather than by the exact literal
// name, this construction is what catches the overreach: it must still
// REJECT after movsd is fixed.

#include <pmmintrin.h>

extern "C" __m128 DuplicateLowLane(__m128 x) {
    return _mm_moveldup_ps(x);
}

extern "C" __m128 DuplicateHighLane(__m128 x) {
    return _mm_movehdup_ps(x);
}
