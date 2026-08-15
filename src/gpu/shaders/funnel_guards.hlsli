// T-1986 GPU-serial port, B2 (Sec6, Sec11 B2): shared guard-domain helpers for
// the funnel's own checked entry points, reused by both
// CheckRoundingDivideByPotExponentDomainGpu and RequantChainCheckedGpu (B2
// does not depend on B5 in Sec11's own dependency graph, so this file's own
// MaxAbsReduceWide-shaped reduction is a self-contained, single-thread guard-
// tier reduction -- NOT the production, parallelized two-schedule reduction
// B5 proves separately for the shipped composed sites). Every function here
// has a named C++ sibling; see src/intmath.cpp and
// src/forward/checked_chain_funnel.cpp.
#ifndef SSLM_FUNNEL_GUARDS_HLSLI
#define SSLM_FUNNEL_GUARDS_HLSLI

#include "wide128.hlsli"

static const int kRoundingDivideByPotExponentMinI64_ = 0;
static const int kRoundingDivideByPotExponentMaxI64_ = 63;
static const int64_t kINT32_MIN_ = -2147483648LL;
static const int64_t kINT32_MAX_ = 2147483647LL;

// intmath.cpp RoundingDivideByPotComposedExponentInDomain (checked_chain_funnel.cpp:373-387).
bool RoundingDivideByPotComposedExponentInDomainGpu(int64_t q_B, int64_t e_a, out int64_t out_exponent)
{
    S128 sum = SAdd(SAdd(SFromI64(q_B), SFromI64(62)), SFromI64(e_a));
    bool in_domain =
        SGe(sum, SFromI64((int64_t)kRoundingDivideByPotExponentMinI64_)) &&
        SLt(sum, SFromI64((int64_t)kRoundingDivideByPotExponentMaxI64_ + 1));
    out_exponent = SLow64(sum);
    return in_domain;
}

// intmath.cpp RoundingDivideByPOTWide (checked_chain_funnel.cpp:357-370 area / intmath.cpp:357-370).
S128 RoundingDivideByPOTWideGpu(S128 x, int exponent)
{
    uint64_t mask = (uint64_t(1) << exponent) - 1ULL;
    uint64_t x_lo = x.lo;
    bool x_negative = ((int64_t)x.hi) < 0;
    uint64_t remainder = x_lo & mask;
    uint64_t threshold = (mask >> 1) + (x_negative ? 1ULL : 0ULL);
    S128 shifted = SShrFull(x, exponent);
    if (remainder > threshold) return SAdd(shifted, SFromI64(1));
    return shifted;
}

// checked_chain_funnel.h CarriedScale.m's own int32 precondition
// (ac34677 S5) -- checked_chain_funnel.cpp's private CarriedScaleMantissaFitsInt32.
bool CarriedScaleMantissaFitsInt32Gpu(int64_t m)
{
    return m >= kINT32_MIN_ && m <= kINT32_MAX_;
}

// intmath.cpp SaturatingAdd64 (also checked_chain_funnel.cpp:93-105, identical body).
int64_t SaturatingAdd64Gpu_(int64_t a, int64_t b)
{
    uint64_t ua = (uint64_t)a;
    uint64_t ub = (uint64_t)b;
    int64_t out_val = (int64_t)(ua + ub);
    bool overflow = ((a >= 0) == (b >= 0)) && ((out_val >= 0) != (a >= 0));
    if (!overflow) return out_val;
    return (a >= 0) ? 0x7FFFFFFFFFFFFFFFLL : (int64_t)0x8000000000000000ULL;
}

// intmath.cpp SaturatingRoundingDoublingHighMul (C1).
int SaturatingRoundingDoublingHighMulGpu_(int a, int b)
{
    int64_t ab = (int64_t)a * (int64_t)b;
    int64_t result = (ab + (int64_t(1) << 30)) >> 31;
    return (result > kINT32_MAX_) ? (int)kINT32_MAX_ : (int)result;
}

// checked_chain_funnel.cpp CombineCarriedScale (C26's carried-scale product step).
void CombineCarriedScaleGpu_(int64_t am, int64_t ae, int64_t bm, int64_t be, out int64_t out_m,
                              out int64_t out_e)
{
    int ma = (int)am;
    int mb = (int)bm;
    int64_t e = SaturatingAdd64Gpu_(SaturatingAdd64Gpu_(ae, be), 31);
    int64_t m = (int64_t)SaturatingRoundingDoublingHighMulGpu_(ma, mb);
    if (m < (int64_t(1) << 30))
    {
        m = m << 1;
        e = e - 1;
    }
    out_m = m;
    out_e = e;
}

// intmath.cpp DynamicScaleReciprocal -- unused by B2's own guard status (r
// never gates a rejection), retained here only if a future caller needs it;
// not called by RequantChainCheckedGpu below (guard-only tier skips step 6).

// intmath.cpp Clz64 (std::countl_zero) -- domain [1, 2^64-1], matching
// NormalizeScale's own precondition (its caller, MaxAbsReduceWide's D' >= 1
// guard).
int Clz64Gpu(uint64_t n)
{
    uint hi = (uint)(n >> 32);
    uint lo = (uint)(n & 0xFFFFFFFFULL);
    if (hi != 0u) return 31 - (int)firstbithigh(hi);
    return 32 + (31 - (int)firstbithigh(lo));
}

// intmath.cpp NormalizeScale (C21).
void NormalizeScaleGpu(int64_t d_prime, out int64_t out_dn, out int out_s)
{
    int p = 63 - Clz64Gpu((uint64_t)d_prime);
    int s = 30 - p;
    out_dn = (s >= 0) ? (d_prime << s) : (d_prime >> 1);
    out_s = s;
}

#endif  // SSLM_FUNNEL_GUARDS_HLSLI
