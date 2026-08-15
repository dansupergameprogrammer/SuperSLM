// T-1986 GPU-serial port, B1 (Sec7.1): RequantTokenCodeWideGpu.
// Bit-exact port of src/intmath.cpp's RequantTokenCodeWide (C22's formula):
// q_i = clamp(round_half_away_from_zero((x_i*127*R) / 2^(62-s)), -127, 127).
// The int8 result is widened to int64_t for a uniform comparison surface
// across every B1 primitive (gpu_port.h's own documented convention).
#include "wide128.hlsli"

ByteAddressBuffer   In  : register(t0);  // int64 x_i, int64 r, int64 s
RWByteAddressBuffer Out : register(u0);  // int64 (widened int8 code)

[numthreads(1, 1, 1)]
void main(uint3 dtid : SV_DispatchThreadID)
{
    int64_t x_i = In.Load<int64_t>(0);
    int64_t r = In.Load<int64_t>(8);
    int s = (int)In.Load<int64_t>(16);

    int exponent = 62 - s;  // in [32, 63] for this primitive's own contract
    uint64_t abs_x = (x_i < 0) ? (~(uint64_t)x_i + 1ULL) : (uint64_t)x_i;

    U128 prod = UMulWide(UMul(abs_x, (uint64_t)r), 127ULL);
    U128 numerator = UAdd64(UTwice(prod), (uint64_t(1) << exponent));
    uint64_t magnitude = UShrToU64(numerator, exponent + 1);

    if (magnitude > 127ULL) magnitude = 127ULL;
    int q = (int)magnitude;
    int code = (x_i < 0) ? -q : q;  // fits int8's range [-127,127] exactly; no HLSL int8 scalar type at cs_6_2
    Out.Store<int64_t>(0, (int64_t)code);
}
