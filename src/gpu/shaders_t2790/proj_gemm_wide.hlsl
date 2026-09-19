// T-2790 spike: the row-widened projection GEMM, one source for all six GEMM sites.
// Compile with -D SSLM_WIDE_H=<rows> -D SSLM_GEMM_SITE=<1..6> -I ../shaders.
//   1 q_proj   (in: normed,    weights Layout[2..5],   out WIDE_A, in_channels hidden, out q_width)
//   2 kv_proj  (in: normed,    K Layout[9,11..13] -> WIDE_A, V Layout[10,14..16] -> WIDE_B)
//   3 o_proj   (in: ctx_codes, weights Layout[25..28], out WIDE_A, in_channels q_width, out hidden)
//   4 gate_proj(in: normed,    weights Layout[37..40], out WIDE_A, in hidden, out intermediate)
//   5 up_proj  (in: normed,    weights Layout[42..45], out WIDE_A, in hidden, out intermediate)
//   6 down_proj(in: act_codes, weights Layout[48..51], out WIDE_A, in intermediate, out hidden)
//
// The shipped per-token sites (../shaders/<site>_gemm_site.hlsl) call GemmCoalescedGpu[At]
// (site_common.hlsli) once per token: `lanes` threads cooperate on one output channel, each lane
// sums its strided slice of the in_channels-wide int64 dot product, a fixed groupshared tree adds
// the lanes, and lane 0 folds and stores. This variant keeps that partition, that channel mapping
// and that fold, and changes one thing: each packed weight dword a lane loads is multiplied into
// every row's accumulator before the next load, so one pass over the weight row serves all
// SSLM_WIDE_H rows. Per row, the accumulator sums exactly the same set of int64 products as the
// per-token site (same weights, same input codes, same in_channels range); integer addition is
// associative and commutative, so the per-row sum, and therefore the folded value, is identical.
//
// A row whose sticky word is not Ok is not stored, which is what the per-token site's early
// return leaves (nothing written). Padded rows are inert this way: the host sets their sticky
// word before the chunk runs.
#include "site_common.hlsli"

cbuffer RootConstants : register(b0)
{
    uint g_layer_index; uint g_hidden_size; uint g_head_dim; uint g_num_kv_heads;
    uint g_context_cap; uint g_position; uint g_num_attention_heads; uint g_width;
    uint g_intermediate_size; uint g_num_hidden_layers; uint g_gemm_lanes; uint g_q_width;
};

ByteAddressBuffer   LayerWeights  : register(t0);
ByteAddressBuffer   Layout        : register(t1);
ByteAddressBuffer   RopeInfo      : register(t2);
ByteAddressBuffer   ScratchLayout : register(t7);
RWByteAddressBuffer SeqStateRows[SSLM_WIDE_H]     : register(u0, space1);
RWByteAddressBuffer LayerScratchRows[SSLM_WIDE_H] : register(u0, space2);
RWByteAddressBuffer KvCache                       : register(u2);
RWByteAddressBuffer WorkScratchRows[SSLM_WIDE_H]  : register(u0, space3);

#if SSLM_WIDE_H >= 8
#define SSLM_WIDE_BATCH 8
#else
#define SSLM_WIDE_BATCH SSLM_WIDE_H
#endif

// Reduction staging: SSLM_WIDE_BATCH rows' lane partials at once (16 KiB at a batch of 8).
groupshared int64_t gWideAcc[SSLM_WIDE_BATCH * 256];

// Every thread of the group must call this, with the same lanes -- it contains group barriers.
void GemmCoalescedWideAt(uint t, int j, uint in_base, uint w_base, uint id_base, uint mult_base,
                         uint shift_base, int in_channels, int out_channels, uint wide_base,
                         uint lanes, uint sticky_off)
{
    const uint lane = t % lanes;
    const bool valid = (j >= 0) && (j < out_channels);
    int64_t acc[SSLM_WIDE_H];
    [unroll] for (uint r0 = 0; r0 < SSLM_WIDE_H; ++r0) acc[r0] = 0;
    if (valid)
    {
        const uint row = w_base + (uint)j * (uint)in_channels;
        if ((row & 3u) == 0u && (in_channels & 3) == 0)
        {
            const int quads = in_channels >> 2;
            for (int q = (int)lane; q < quads; q += (int)lanes)
            {
                const uint wp = LayerWeights.Load(row + (uint)q * 4u);
                const int64_t w0 = (int64_t)(int(wp << 24) >> 24);
                const int64_t w1 = (int64_t)(int(wp << 16) >> 24);
                const int64_t w2 = (int64_t)(int(wp << 8) >> 24);
                const int64_t w3 = (int64_t)(int(wp) >> 24);
                [unroll] for (uint r = 0; r < SSLM_WIDE_H; ++r)
                {
                    const uint4 a4 = LayerScratchRows[r].Load4(in_base + (uint)q * 16u);
                    acc[r] += (int64_t)(int)a4.x * w0;
                    acc[r] += (int64_t)(int)a4.y * w1;
                    acc[r] += (int64_t)(int)a4.z * w2;
                    acc[r] += (int64_t)(int)a4.w * w3;
                }
            }
        }
        else
        {
            for (int i = (int)lane; i < in_channels; i += (int)lanes)
            {
                const int64_t w = (int64_t)LoadSignedByteGpu(LayerWeights, row + (uint)i);
                [unroll] for (uint r = 0; r < SSLM_WIDE_H; ++r)
                    acc[r] += (int64_t)LayerScratchRows[r].Load<int>(in_base + (uint)i * 4u) * w;
            }
        }
    }

    int identity = 0, mult = 0, shift = 0;
    if (valid && lane == 0u)
    {
        identity = (int)LayerWeights.Load<int>(id_base + (uint)j * 4u);
        mult = (int)LayerWeights.Load<int>(mult_base + (uint)j * 4u);
        shift = (int)LayerWeights.Load<int>(shift_base + (uint)j * 4u);
    }

    [unroll] for (uint b0 = 0; b0 < SSLM_WIDE_H; b0 += SSLM_WIDE_BATCH)
    {
        [unroll] for (uint b = 0; b < SSLM_WIDE_BATCH; ++b) gWideAcc[b * 256u + t] = acc[b0 + b];
        GroupMemoryBarrierWithGroupSync();
        for (uint s = lanes >> 1; s > 0u; s >>= 1)
        {
            if (lane < s)
            {
                [unroll] for (uint b = 0; b < SSLM_WIDE_BATCH; ++b)
                    gWideAcc[b * 256u + t] += gWideAcc[b * 256u + t + s];
            }
            GroupMemoryBarrierWithGroupSync();
        }
        if (valid && lane == 0u)
        {
            [unroll] for (uint b = 0; b < SSLM_WIDE_BATCH; ++b)
            {
                if (SeqStateRows[b0 + b].Load<int64_t>(sticky_off) == kTagOk)
                {
                    const int64_t folded = ApplyWeightScaleFoldGpu(gWideAcc[b * 256u + t], identity, mult, shift);
                    WorkScratchRows[b0 + b].Store<int64_t>(wide_base + (uint)j * 8u, folded);
                }
            }
        }
        GroupMemoryBarrierWithGroupSync();
    }
}

[numthreads(256, 1, 1)]
void main(uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID)
{
    const uint t = gtid.x;
    const int hidden_size = (int)g_hidden_size;
    const uint sticky_off = SeqStickyOffGpu(hidden_size);
    const uint layer_base = g_layer_index * Layout.Load<uint>(56 * 4);
    const uint lanes = g_gemm_lanes;
    const uint channels_per_group = 256u / lanes;
    const int c = (int)(gid.x * channels_per_group + t / lanes);

#if SSLM_GEMM_SITE == 2
    const int kv_hidden_size = (int)g_num_kv_heads * (int)g_head_dim;
    const uint normed_off = ScratchLayout.Load<uint>(0 * 4);
    const uint wide_b_off = ScratchLayout.Load<uint>(23 * 4);
    GemmCoalescedWideAt(t, c, normed_off, layer_base + Layout.Load<uint>(9 * 4),
                        layer_base + Layout.Load<uint>(11 * 4), layer_base + Layout.Load<uint>(12 * 4),
                        layer_base + Layout.Load<uint>(13 * 4), hidden_size, kv_hidden_size, 0u, lanes,
                        sticky_off);
    GemmCoalescedWideAt(t, c - kv_hidden_size, normed_off, layer_base + Layout.Load<uint>(10 * 4),
                        layer_base + Layout.Load<uint>(14 * 4), layer_base + Layout.Load<uint>(15 * 4),
                        layer_base + Layout.Load<uint>(16 * 4), hidden_size, kv_hidden_size, wide_b_off,
                        lanes, sticky_off);
#else
#if SSLM_GEMM_SITE == 1
    const uint in_off = ScratchLayout.Load<uint>(0 * 4);
    const uint wl = 2; const int in_ch = hidden_size; const int out_ch = (int)g_q_width;
#elif SSLM_GEMM_SITE == 3
    const uint in_off = ScratchLayout.Load<uint>(5 * 4);
    const uint wl = 25; const int in_ch = (int)g_q_width; const int out_ch = hidden_size;
#elif SSLM_GEMM_SITE == 4
    const uint in_off = ScratchLayout.Load<uint>(0 * 4);
    const uint wl = 37; const int in_ch = hidden_size; const int out_ch = (int)g_intermediate_size;
#elif SSLM_GEMM_SITE == 5
    const uint in_off = ScratchLayout.Load<uint>(0 * 4);
    const uint wl = 42; const int in_ch = hidden_size; const int out_ch = (int)g_intermediate_size;
#elif SSLM_GEMM_SITE == 6
    const uint in_off = ScratchLayout.Load<uint>(15 * 4);
    const uint wl = 48; const int in_ch = (int)g_intermediate_size; const int out_ch = hidden_size;
#else
#error SSLM_GEMM_SITE must be 1..6
#endif
    GemmCoalescedWideAt(t, c, in_off, layer_base + Layout.Load<uint>(wl * 4),
                        layer_base + Layout.Load<uint>((wl + 1) * 4), layer_base + Layout.Load<uint>((wl + 2) * 4),
                        layer_base + Layout.Load<uint>((wl + 3) * 4), in_ch, out_ch, 0u, lanes, sticky_off);
#endif
}
