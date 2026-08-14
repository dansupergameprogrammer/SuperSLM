// T-1986 GPU-serial port, B3 (Sec5.1, Sec11 B3): the generic descriptor-table
// binding substrate, TestT2019_B3_DescriptorTableBinding_KnownPatternReadback's
// own N-arbitrary-arrays shape. One thread per array; each thread reads its
// own array's bytes back through the bound, unbounded-array-in-shader
// descriptor table (the SM6.2-compatible idiom Sec5.1 specifies), sign-
// extending each byte the same way Sec5.2's corrected unpack expression does
// (shift-left-then-arithmetic-right-shift, since HLSL has no int8 scalar type
// at this compile target), and widens every element to int32 on write --
// matching this design's own established "int, widened int8" storage
// convention (the proven qproj_site.hlsl spike shader's own Out.Store<int>).
ByteAddressBuffer Bufs[] : register(t0, space1);  // the N bound array buffers
ByteAddressBuffer Counts : register(t1);          // N x int64 element counts
ByteAddressBuffer Offsets : register(t2);         // N x int64 element-index output offsets
RWByteAddressBuffer Out : register(u0);           // total_elements x int32 (widened int8)

cbuffer C : register(b0) { uint NArrays; };

int LoadSignedByte(uint arrayIndex, uint byteIdx)
{
    // NonUniformResourceIndex: each of this dispatch's threads indexes a
    // DIFFERENT element of the bound Bufs[] table (one array per thread),
    // a genuinely divergent index across the wave -- the documented D3D12/
    // HLSL requirement for correctness of a per-lane-varying resource-array
    // index, not merely a style preference.
    uint w = Bufs[NonUniformResourceIndex(arrayIndex)].Load((byteIdx / 4u) * 4u);
    uint shift = (byteIdx % 4u) * 8u;
    uint b = (w >> shift) & 0xFFu;
    return int(b << 24) >> 24;  // sign-extend, Sec5.2's own corrected shift idiom
}

[numthreads(64, 1, 1)]
void main(uint3 dtid : SV_DispatchThreadID)
{
    uint i = dtid.x;
    if (i >= NArrays) return;

    int64_t count = Counts.Load<int64_t>(i * 8u);
    int64_t elem_offset = Offsets.Load<int64_t>(i * 8u);

    for (int64_t e = 0; e < count; ++e)
    {
        int v = LoadSignedByte(i, (uint)e);
        Out.Store((uint)((elem_offset + e) * 4), (uint)v);
    }
}
