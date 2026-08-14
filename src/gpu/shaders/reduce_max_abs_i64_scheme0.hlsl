// T-1986 GPU-serial port, B5 (Sec5.5, Sec11 B5): MaxAbsReduceWideGpuScheme0.
// SCHEME 0 (sequential) of the proven `reduce_max_abs_i64` family
// (Claude/Loki/t1993-probe/reduce_max_abs_i64.hlsl, T-1993's own remedy,
// D-SLM2924) -- ported here as this design's own build artifact (single-row
// call shape matching gpu_port.h's own `MaxAbsReduceWideGpuScheme0(x, n)`
// signature, not the probe's M-batched harness). Body unchanged from the
// proven probe: same absmag (unsigned two's-complement negate, defined for
// INT64_MIN), same accumulation order.
//
// Input layout: offset 0 = int64 n; offset 8 = row[0..n-1] (int64 each).
// Output: int64 (the CPU MaxAbsReduceWide's own saturating-narrowed result).
ByteAddressBuffer   In  : register(t0);
RWByteAddressBuffer Out : register(u0);

uint64_t absmag(int64_t v)
{
    return v < 0 ? (~(uint64_t)v + 1ULL) : (uint64_t)v;
}

[numthreads(256, 1, 1)]
void main(uint3 tid : SV_GroupThreadID)
{
    if (tid.x != 0) return;
    int64_t n = In.Load<int64_t>(0);
    uint64_t acc = 0;
    for (int64_t i = 0; i < n; ++i)
    {
        acc = max(acc, absmag(In.Load<int64_t>(8 + (uint)i * 8)));
    }
    if (acc < 1ULL) acc = 1ULL;  // all-zero-row guard (intmath.cpp:433)
    // Saturating narrow to int64_t range (intmath.cpp:434-441).
    int64_t result = (acc > 0x7FFFFFFFFFFFFFFFULL) ? 0x7FFFFFFFFFFFFFFFLL : (int64_t)acc;
    Out.Store<int64_t>(0, result);
}
