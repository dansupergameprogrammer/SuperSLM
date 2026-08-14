// T-1986 GPU-serial port, B1 (Sec7.1): CombineCarriedScaleGpu.
// Bit-exact port of src/forward/checked_chain_funnel.cpp's CombineCarriedScale
// plus its own SaturatingAdd64 helper (co-located, checked_chain_funnel.cpp:
// 93-176) and src/intmath.cpp's SaturatingRoundingDoublingHighMul (C1).
// CombineCarriedScale never rejects (Sec6, D-SLM2906) -- this kernel has no
// status word, only the saturated (m, e) pair.
#include "wide128.hlsli"

ByteAddressBuffer   In  : register(t0);  // int64 a.m, a.e, b.m, b.e
RWByteAddressBuffer Out : register(u0);  // int64 m, int64 e

static const int64_t kINT64_MAX = 0x7FFFFFFFFFFFFFFFLL;
static const int64_t kINT64_MIN = (int64_t)0x8000000000000000ULL;
static const int kINT32_MAX = 0x7FFFFFFF;

int64_t SaturatingAdd64Gpu(int64_t a, int64_t b)
{
    uint64_t ua = (uint64_t)a;
    uint64_t ub = (uint64_t)b;
    int64_t out_val = (int64_t)(ua + ub);
    bool overflow = ((a >= 0) == (b >= 0)) && ((out_val >= 0) != (a >= 0));
    if (!overflow) return out_val;
    return (a >= 0) ? kINT64_MAX : kINT64_MIN;
}

int SaturatingRoundingDoublingHighMulGpu(int a, int b)
{
    int64_t ab = (int64_t)a * (int64_t)b;
    int64_t result = (ab + (int64_t(1) << 30)) >> 31;
    return (result > (int64_t)kINT32_MAX) ? kINT32_MAX : (int)result;
}

[numthreads(1, 1, 1)]
void main(uint3 dtid : SV_DispatchThreadID)
{
    int64_t am = In.Load<int64_t>(0);
    int64_t ae = In.Load<int64_t>(8);
    int64_t bm = In.Load<int64_t>(16);
    int64_t be = In.Load<int64_t>(24);

    int ma = (int)am;
    int mb = (int)bm;
    int64_t e = SaturatingAdd64Gpu(SaturatingAdd64Gpu(ae, be), 31);
    int64_t m = (int64_t)SaturatingRoundingDoublingHighMulGpu(ma, mb);
    if (m < (int64_t(1) << 30))
    {
        m = m << 1;
        e = e - 1;
    }

    Out.Store<int64_t>(0, m);
    Out.Store<int64_t>(8, e);
}
