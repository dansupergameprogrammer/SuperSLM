// T-2851 (design Sec4.6): the token finish's logits row on the device. For every vocabulary row j,
// out[j] = the exact int64 sum over k of x[k] * head[j * hidden + k] -- the value the host's
// GemmInt8AccumulateRow writes into its wide row. No narrowing happens here and no domain flag
// exists: the host reads the int64 row back and runs its unchanged NarrowRowChecked on it.
//
// The arithmetic is GemmCoalescedGpuAt's (site_common.hlsli) without the weight-scale fold, which
// LogitsSite does not apply: 32 lanes cooperate on one row, each keeping an exact int64 partial,
// and a fixed groupshared tree combines them. Integer addition is exact, so the total equals the
// host's for any lane assignment. The packed-dword path runs when the row is dword-aligned and
// hidden % 4 == 0; otherwise the byte path computes the same sum. This is a separate shader, not a
// refactor of GemmCoalescedGpuAt, so the shipped layer shaders' binaries are unchanged.
//
// Bindings (gpu_1p0.cpp, MakeLogitsSiteRootSignature): b0 = {hidden, vocab, groups_x, unused};
// t0 = the head table, row-major, padded to a multiple of 4 bytes; t1 = x as one int32 per code;
// u0 = the int64 output row. Dispatch: ceil(vocab / 8) groups of 256 threads, laid out as
// groups_x x groups_y so no dimension exceeds the 65,535-group limit.
#define LANES 32u
#define ROWS_PER_GROUP (256u / LANES)

cbuffer Constants : register(b0)
{
    uint g_hidden;
    uint g_vocab;
    uint g_groups_x;
    uint g_unused;
};
ByteAddressBuffer Head : register(t0);
ByteAddressBuffer X : register(t1);
RWByteAddressBuffer Out : register(u0);

groupshared int64_t gAcc[256];

int LoadSignedByte(ByteAddressBuffer buf, uint byte_offset)
{
    uint w = buf.Load((byte_offset / 4u) * 4u);
    uint shift = (byte_offset % 4u) * 8u;
    return int(((w >> shift) & 0xFFu) << 24) >> 24;
}

[numthreads(256, 1, 1)]
void main(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID)
{
    const uint t = tid.x;
    const uint lane = t % LANES;
    const uint j = (gid.y * g_groups_x + gid.x) * ROWS_PER_GROUP + t / LANES;
    int64_t acc = 0;
    if (j < g_vocab)
    {
        const uint row = j * g_hidden;
        if ((row & 3u) == 0u && (g_hidden & 3u) == 0u)
        {
            const uint quads = g_hidden >> 2;
            for (uint q = lane; q < quads; q += LANES)
            {
                uint wp = Head.Load(row + q * 4u);
                uint4 a4 = X.Load4(q * 16u);
                acc += (int64_t)(int)a4.x * (int64_t)(int(wp << 24) >> 24);
                acc += (int64_t)(int)a4.y * (int64_t)(int(wp << 16) >> 24);
                acc += (int64_t)(int)a4.z * (int64_t)(int(wp << 8) >> 24);
                acc += (int64_t)(int)a4.w * (int64_t)(int(wp) >> 24);
            }
        }
        else
        {
            for (uint i = lane; i < g_hidden; i += LANES)
            {
                int a = X.Load<int>(i * 4u);
                int w = LoadSignedByte(Head, row + i);
                acc += (int64_t)a * (int64_t)w;
            }
        }
    }
    gAcc[t] = acc;
    GroupMemoryBarrierWithGroupSync();
    for (uint s = LANES >> 1; s > 0u; s >>= 1)
    {
        if (lane < s) gAcc[t] += gAcc[t + s];
        GroupMemoryBarrierWithGroupSync();
    }
    if (lane == 0u && j < g_vocab)
    {
        Out.Store<int64_t>(j * 8u, gAcc[t]);
    }
}
