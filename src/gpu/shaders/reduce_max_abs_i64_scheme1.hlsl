// T-1986 GPU-serial port, B5 (Sec5.5, Sec11 B5): MaxAbsReduceWideGpuScheme1.
// SCHEME 1 (shared-memory binary tree) of the proven `reduce_max_abs_i64`
// family (Claude/Loki/t1993-probe/reduce_max_abs_i64.hlsl, T-1993's own
// remedy, D-SLM2924) -- ported here as this design's own build artifact
// (single-row call shape matching gpu_port.h's own
// `MaxAbsReduceWideGpuScheme1(x, n)` signature). Body unchanged from the
// proven probe: same absmag, same 256-thread strided partial reduction, same
// log2(256) tree combine.
//
// Input layout: offset 0 = int64 n; offset 8 = row[0..n-1] (int64 each).
// Output: int64 (the CPU MaxAbsReduceWide's own saturating-narrowed result).
ByteAddressBuffer   In  : register(t0);
RWByteAddressBuffer Out : register(u0);

groupshared uint64_t gmax[256];

uint64_t absmag(int64_t v)
{
    return v < 0 ? (~(uint64_t)v + 1ULL) : (uint64_t)v;
}

[numthreads(256, 1, 1)]
void main(uint3 tid : SV_GroupThreadID)
{
    int64_t n = In.Load<int64_t>(0);
    uint t = tid.x;

    uint64_t local = 0;
    for (int64_t i = (int64_t)t; i < n; i += 256)
    {
        local = max(local, absmag(In.Load<int64_t>(8 + (uint)i * 8)));
    }
    gmax[t] = local;
    GroupMemoryBarrierWithGroupSync();
    for (uint s = 128; s > 0; s >>= 1)
    {
        if (t < s) gmax[t] = max(gmax[t], gmax[t + s]);
        GroupMemoryBarrierWithGroupSync();
    }
    if (t == 0)
    {
        uint64_t acc = gmax[0];
        if (acc < 1ULL) acc = 1ULL;
        int64_t result = (acc > 0x7FFFFFFFFFFFFFFFULL) ? 0x7FFFFFFFFFFFFFFFLL : (int64_t)acc;
        Out.Store<int64_t>(0, result);
    }
}
