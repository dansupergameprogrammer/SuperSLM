// T-1986 GPU-serial port, B1 (Sec7.1): BiasReconcileWideGpu.
// Bit-exact port of src/intmath.cpp's BiasReconcileWide plus its own
// RoundingDivideByPotComposedExponentInDomain / RoundingDivideByPOTWide
// helpers (checked_chain_funnel.cpp:373-410 co-located those two; ported
// together here since BiasReconcileWideGpu is this design's only GPU
// surface for either).
#include "wide128.hlsli"

ByteAddressBuffer   In  : register(t0);  // int64 b, int64 q_b, int64 r_a, int64 e_a
RWByteAddressBuffer Out : register(u0);  // int64 ok (0/1), int64 out

static const int kRoundingDivideByPotExponentMinI64 = 0;
static const int kRoundingDivideByPotExponentMaxI64 = 63;

bool RoundingDivideByPotComposedExponentInDomainGpu(int64_t q_B, int64_t e_a, out int64_t out_exponent)
{
    S128 sum = SAdd(SAdd(SFromI64(q_B), SFromI64(62)), SFromI64(e_a));
    bool in_domain =
        SGe(sum, SFromI64((int64_t)kRoundingDivideByPotExponentMinI64)) &&
        SLt(sum, SFromI64((int64_t)kRoundingDivideByPotExponentMaxI64 + 1));
    out_exponent = SLow64(sum);
    return in_domain;
}

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

[numthreads(1, 1, 1)]
void main(uint3 dtid : SV_DispatchThreadID)
{
    int64_t b = In.Load<int64_t>(0);
    int64_t q_b = In.Load<int64_t>(8);
    int64_t r_a = In.Load<int64_t>(16);
    int64_t e_a = In.Load<int64_t>(24);

    int64_t exponent = 0;
    bool ok = false;
    int64_t out_val = 0;

    if (!RoundingDivideByPotComposedExponentInDomainGpu(q_b, e_a, exponent))
    {
        out_val = 0;
        ok = false;
    }
    else
    {
        S128 wide = SMul(b, r_a);
        S128 rounded = RoundingDivideByPOTWideGpu(wide, (int)exponent);
        ok = SFitsI64(rounded);
        out_val = SLow64(rounded);
    }

    Out.Store<int64_t>(0, ok ? int64_t(1) : int64_t(0));
    Out.Store<int64_t>(8, out_val);
}
