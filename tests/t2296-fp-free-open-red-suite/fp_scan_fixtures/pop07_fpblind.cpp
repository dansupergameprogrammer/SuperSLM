// T-2326 (Curie) -- Sec7 dimension 11's seventh commissioning population, adapted
// for a portable clang build.
//
// PROVENANCE: T-2271's own eight-object construction (design Sec4.1, D-SLM4350;
// Claude/Vitruvius/t2265-fold7-probe/fpblind.cpp, adversary-authored, independent
// of the instrument's maker per StandardsDocument.md Sec5.4's commissioning
// rule). Four functions, each performing a genuine floating-point operation the
// pre-fold-7 36-mnemonic detector could not see, isolating the CLASSIFIER's own
// blind spot rather than the walk.
//
// ADAPTED, NOT A BYTE-IDENTICAL COPY: the original includes <cmath>, which pulls
// MSVC's own STL headers when compiled with clang -target *-windows-msvc, and
// MSVC's STL (14.44) refuses any clang below 19.0.0 (this toolchain is 18.1.8) --
// confirmed by direct compile attempt, error STL1000. Replacing std::floor/
// std::fma with the equivalent compiler builtins (__builtin_floorf/
// __builtin_fmaf) removes the STL dependency without changing the population's
// own defining property: each function still performs a genuine floating-point
// operation invisible to the pre-fold-7 mnemonic/libm-name detector (a builtin
// lowers to the identical machine instruction a libm call or an intrinsic would).
// BodyRound/BodyHadd are unchanged from the original (immintrin.h carries no STL
// dependency).
//
// Falsifying property: every one of these four bodies, at this ISA tier, must
// REJECT under the deciding instrument's checks (A)/(B) (a vector/FP register
// touched by an instruction not on VEC_MOVE_ALLOW). A scan whose classifier
// still enumerates only 36 of 159 mnemonic classes (the pre-fold-7 shape T-2271
// struck) ACCEPTs all four; the amended default-deny classifier REJECTs all
// four (D-SLM4350: 8 of 8 REJECT across two ISA tiers in the original strike).

#include <immintrin.h>

// A -- a plain floor() on a float. The shape a future open-path sizing/ratio
//      computation most plausibly takes.
extern "C" float BodyFloor(float x) { return __builtin_floorf(x); }

// B -- fused multiply-add.
extern "C" float BodyFma(float a, float b, float c) { return __builtin_fmaf(a, b, c); }

// C -- SSE4.1 round-to-nearest with the precision exception NOT suppressed
//      (imm8 bit 3 clear). Raises PE on any operand that actually rounds.
extern "C" float BodyRound(float x) {
    __m128 v = _mm_set_ss(x);
    return _mm_cvtss_f32(_mm_round_ss(v, v, _MM_FROUND_TO_NEAREST_INT));
}

// D -- horizontal add: the shape an auto-vectorized reduction takes.
extern "C" float BodyHadd(float a, float b) {
    __m128 v = _mm_set_ps(a, b, a, b);
    return _mm_cvtss_f32(_mm_hadd_ps(v, v));
}
