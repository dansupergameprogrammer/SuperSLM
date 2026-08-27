// T-2342 (Curie) -- Sec4.1 gap (e)'s AArch64 half: the BFloat16 family
// (bfdot/bfmmla/etc.) touches a NEON v<n> register, carries no leading `f`
// (it carries `bf`, indistinguishable from a `b`-prefixed integer mnemonic
// like `bic` under the printed structural rule), and is not one of the three
// named domain-crossing conversions -- so the rule as printed ACCEPTs it
// structurally, and every one of these is genuine BFloat16 floating-point
// arithmetic. Proven through the real production route this session
// (matching Claude/Popper/t2340-fp-scan-instrument-commissioning-2026-08-27.md
// Sec4.4, D-SLM4850): compiled by clang for
// --target=aarch64-linux-gnu -march=armv8.6-a+bf16 -O2, a real ELF object,
// decoded and confirmed via capstone this session ("bfdot v0.4s, v1.8h,
// v2.8h" / "bfmmla v0.4s, v1.8h, v2.8h"), and scan_object (the built
// instrument, unmodified) returns ACCEPT for both -- the falsifying,
// currently-wrong verdict this cell pins as red.

#include <arm_neon.h>

__attribute__((noinline)) float32x4_t Bf16DotProduct(float32x4_t r, bfloat16x8_t a, bfloat16x8_t b) {
    return vbfdotq_f32(r, a, b);
}

__attribute__((noinline)) float32x4_t Bf16MatrixMultiplyAccumulate(float32x4_t r, bfloat16x8_t a, bfloat16x8_t b) {
    return vbfmmlaq_f32(r, a, b);
}
