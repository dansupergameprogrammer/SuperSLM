// T-1986 GPU-serial port, B1 (Sec7.1): DynamicScaleReciprocalGpu.
// Bit-exact port of src/intmath.cpp's DynamicScaleReciprocal -- 3 Newton
// iterations plus 2 unconditional branch-free correction steps, all
// data-independent op counts (the CPU header's own C19/C14 discipline). One
// scalar dispatch per call (single thread) -- the test suite issues one
// dispatch per swept fixture, matching B1's own "one GPU dispatch per call"
// shape (Sec11 B1).
#include "wide128.hlsli"

ByteAddressBuffer   In  : register(t0);  // 1 x int64 dn
RWByteAddressBuffer Out : register(u0);  // 1 x int64 y

int64_t DynamicScaleReciprocalGpuImpl(int64_t dn)
{
    // round_half_up(48*2^31/17), round_half_up(32*2^31/17) -- same literal
    // arithmetic as intmath.cpp; both fit int64_t with no overflow risk.
    const int64_t kC32 = (2 * (int64_t(48) << 31) + 17) / 34;
    const int64_t kC32_2 = (2 * (int64_t(32) << 31) + 17) / 34;

    int64_t y = kC32 - ((kC32_2 * dn) >> 31);

    [unroll]
    for (int i = 0; i < 3; ++i)
    {
        int64_t dn_y = SShrToI64(SMul(dn, y), 31);
        int64_t delta = (int64_t(1) << 32) - dn_y;
        y = SShrToI64(SMul(y, delta), 31);
    }

    [unroll]
    for (int j = 0; j < 2; ++j)
    {
        S128 residual_2x = STwice(SSub(SFromI64(int64_t(1) << 62), SMul(y, dn)));
        if (SGe(residual_2x, SFromI64(dn)))
        {
            y += 1;
        }
        else if (SLt(residual_2x, SFromI64(-dn)))
        {
            y -= 1;
        }
    }
    return y;
}

[numthreads(1, 1, 1)]
void main(uint3 dtid : SV_DispatchThreadID)
{
    int64_t dn = In.Load<int64_t>(0);
    int64_t y = DynamicScaleReciprocalGpuImpl(dn);
    Out.Store<int64_t>(0, y);
}
