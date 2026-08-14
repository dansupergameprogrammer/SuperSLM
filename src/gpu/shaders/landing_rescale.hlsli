// T-2032 (Sec5.6/Sec11 B4 well-scoped next checkpoint): LandingRescaleGpu --
// C27's K/V landing composite (forward_sites.cpp:489-629, LandingRescale),
// bit-exact port. This is the single largest new porting item the T-2031
// composed-pipeline analysis names (Claude/Brunel/t2025-gpu-serial-build-
// 2026-08-13.md Sec11.2): forward_sites.cpp's OWN U128 family (wide128.hlsli's
// UAddWide/UShlWide/UOneShlWide/UShrWideFull, added alongside this file), NOT
// intmath.cpp's U128/S128 family the funnel guards already use.
#ifndef SSLM_LANDING_RESCALE_HLSLI
#define SSLM_LANDING_RESCALE_HLSLI

#include "wide128.hlsli"

// forward_sites.cpp SaturatingSub64 -- unsigned-wraparound compute, then the
// standard two's-complement overflow test, saturating toward the direction
// the true (unclamped) result would have overflowed.
int64_t SaturatingSub64Wide(int64_t a, int64_t b)
{
    uint64_t ua = (uint64_t)a;
    uint64_t ub = (uint64_t)b;
    int64_t out_val = (int64_t)(ua - ub);
    bool a_neg = a < 0;
    bool b_neg = b < 0;
    bool out_neg = out_val < 0;
    bool overflow = (a_neg != b_neg) && (out_neg != a_neg);
    if (!overflow) return out_val;
    return a_neg ? (int64_t)0x8000000000000000ULL : 0x7FFFFFFFFFFFFFFFLL;
}

// forward_sites.cpp ComposedExponent -- k = clamp(62 - (e_a - e_t), +-4096).
int64_t ComposedExponentGpu(int64_t e_a, int64_t e_t)
{
    int64_t diff = SaturatingSub64Wide(e_a, e_t);
    int64_t k = SaturatingSub64Wide((int64_t)62, diff);
    const int64_t kClamp = 4096;
    if (k > kClamp) k = kClamp;
    if (k < -kClamp) k = -kClamp;
    return k;
}

// forward_sites.cpp LandingRescale, bit-exact: rounds/shifts
// branch_code*m_a*r_t by 2^k (k = ComposedExponent(e_a, e_t)), magnitude-then-
// sign, and reports (via out params) whether the caller's own
// clamp(-127,127) would have clamped this raw value (the T-518 saturation
// signal). This checkpoint's own build does not need
// out_magnitude_exceeded_int64 (Option-G-only, unreachable at this design's
// build target, D-SLM2994) -- omitted here rather than plumbed unused.
int64_t LandingRescaleGpu(int64_t branch_code, int64_t m_a, int64_t r_t, int64_t e_a, int64_t e_t,
                          out bool out_would_clamp)
{
    bool branch_negative = branch_code < 0;
    bool m_a_negative = m_a < 0;
    bool negative = branch_negative != m_a_negative;
    uint64_t abs_branch = branch_negative ? (~(uint64_t)branch_code + 1ULL) : (uint64_t)branch_code;
    uint64_t abs_m_a = m_a_negative ? (~(uint64_t)m_a + 1ULL) : (uint64_t)m_a;

    U128 mag = UMulWide(UMul(abs_branch, abs_m_a), (uint64_t)r_t);  // intmath.cpp-family 64x64->128, then *scalar

    int64_t k = ComposedExponentGpu(e_a, e_t);
    int64_t raw;
    bool magnitude_exceeds_clamp = false;

    if (k >= 0)
    {
        U128 doubled = UAddWide(mag, mag);
        U128 rounded = UAddWide(doubled, UOneShlWide((int)k));
        U128 quotient = UShrWideFull(rounded, (int)k + 1);
        magnitude_exceeds_clamp = (quotient.hi != 0ULL) || (quotient.lo > 127ULL);
        raw = (int64_t)quotient.lo;
    }
    else
    {
        int shift = (int)(-k);
        if (shift >= 128)
        {
            magnitude_exceeds_clamp = (mag.lo != 0ULL || mag.hi != 0ULL);
            raw = 0;
        }
        else
        {
            U128 shifted = UShlWide(mag, shift);
            U128 verify = UShrWideFull(shifted, shift);
            bool shift_lost_bits = (verify.lo != mag.lo) || (verify.hi != mag.hi);
            magnitude_exceeds_clamp = shift_lost_bits || (shifted.hi != 0ULL) || (shifted.lo > 127ULL);
            raw = (int64_t)shifted.lo;
        }
    }

    if (negative) raw = (int64_t)(0ULL - (uint64_t)raw);

    out_would_clamp = magnitude_exceeds_clamp || raw < -127 || raw > 127;
    return raw;
}

#endif  // SSLM_LANDING_RESCALE_HLSLI
