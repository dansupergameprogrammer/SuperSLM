// T-2032/T-2035: RoPE -- real GPU dispatch, site 4 of 16. T-2032 built the
// guard only; T-2035 completes the site: bit-exact port of forward_sites.cpp's
// RopeApplySite (forward_sites.cpp:639-737) in full (steps 1-5), plus the
// per-head Q/K composition RunLayerLoopImpl drives it with
// (forward_sites.cpp:1546-1585) -- Q rotated into LayerScratch, K read from
// the just-landed KV cache row, rotated, and written back to the SAME row
// (option_g_fused_k_landing is always false at this design's build target,
// D-SLM2994, so the legacy read-rotate-then-writeback path is the only one
// this design's build ever takes).
#include "site_common2.hlsli"

cbuffer RootConstants : register(b0)
{
    uint g_layer_index;
    uint g_hidden_size;
    uint g_head_dim;
    uint g_num_kv_heads;
    uint g_context_cap;
    uint g_position;
    uint g_num_attention_heads;
    uint g_width;
    uint g_intermediate_size;
};

ByteAddressBuffer   LayerWeights   : register(t0);
ByteAddressBuffer   Layout         : register(t1);
ByteAddressBuffer   RopeInfo       : register(t2);
ByteAddressBuffer   ModelConstants : register(t3);
ByteAddressBuffer   SiluLut        : register(t4);
ByteAddressBuffer   RopeCosTable   : register(t5);
ByteAddressBuffer   RopeSinTable   : register(t6);
RWByteAddressBuffer SeqState       : register(u0);
RWByteAddressBuffer LayerScratch   : register(u1);
RWByteAddressBuffer KvCache        : register(u2);

[numthreads(1, 1, 1)]
void main(uint3 dtid : SV_DispatchThreadID)
{
    int64_t sticky = SeqState.Load<int64_t>(72);
    if (sticky != kTagOk) return;

    if (!CheckPositionOverCapGpu((int64_t)g_position, (int64_t)g_context_cap))
    {
        SeqState.Store<int64_t>(72, kTagPositionOverCap);
        return;
    }

    uint cos_present = RopeInfo.Load(0);
    uint sin_present = RopeInfo.Load(4);
    if (cos_present == 0u || sin_present == 0u)
    {
        SeqState.Store<int64_t>(72, kTagRopeTableTensorMissing);
        return;
    }

    uint64_t cos_elem_count = RopeInfo.Load<uint64_t>(8);
    uint64_t sin_elem_count = RopeInfo.Load<uint64_t>(16);
    uint pairs = g_head_dim / 2u;
    if (pairs == 0u || (uint64_t)pairs > cos_elem_count || (uint64_t)pairs > sin_elem_count)
    {
        SeqState.Store<int64_t>(72, kTagRopeTableExtentExceeded);
        return;
    }
    uint64_t cos_rows = cos_elem_count / (uint64_t)pairs;
    uint64_t sin_rows = sin_elem_count / (uint64_t)pairs;
    if ((uint64_t)g_position >= cos_rows || (uint64_t)g_position >= sin_rows)
    {
        SeqState.Store<int64_t>(72, kTagRopeTableExtentExceeded);
        return;
    }

    uint row_offset = g_position * pairs;
    uint group = (g_num_kv_heads > 0u) ? (g_num_attention_heads / g_num_kv_heads) : 1u;
    uint kv_half_off = KvHalfOffsetGpu(g_layer_index, g_context_cap, g_num_kv_heads, g_head_dim);

    for (uint h = 0; h < g_num_attention_heads; ++h)
    {
        // Q: read this head's codes from scratch (offset 48), rotate, write
        // the rotated result to q_rot (offset 96) -- never committed to
        // persistent state (§4: Q's own output is never landed).
        for (uint p = 0; p < pairs; ++p)
        {
            int x = (int)LayerScratch.Load<int>(48 + (h * g_head_dim + 2u * p) * 4u);
            int y = (int)LayerScratch.Load<int>(48 + (h * g_head_dim + 2u * p + 1u) * 4u);
            int cos_q30 = (int)RopeCosTable.Load<int64_t>((row_offset + p) * 8u);
            int sin_q30 = (int)RopeSinTable.Load<int64_t>((row_offset + p) * 8u);
            int64_t rx, ry;
            RopeApplyPairGpu(x, y, cos_q30, sin_q30, rx, ry);
            LayerScratch.Store<int>(96 + (h * g_head_dim + 2u * p) * 4u, (int)ClampRopeCodeGpu(rx));
            LayerScratch.Store<int>(96 + (h * g_head_dim + 2u * p + 1u) * 4u, (int)ClampRopeCodeGpu(ry));
        }

        // K: read the just-landed row (kv_proj already committed it this
        // layer), rotate, write back to the SAME row -- the guard above is
        // uniform across every head (same table, same position), so there is
        // no per-head rejection possible once it has passed once; no staging
        // is needed beyond that (unlike the fused-K-landing case forward_
        // sites.cpp guards against, which this design's build target never
        // reaches, D-SLM2994).
        uint kv_head = h / max(group, 1u);
        uint row_off = KvRowOffsetWithinHalfGpu(g_context_cap, g_head_dim, kv_head, g_position);
        for (uint p2 = 0; p2 < pairs; ++p2)
        {
            int kx = LoadSignedByteGpu(KvCache, kv_half_off + row_off + 2u * p2);
            int ky = LoadSignedByteGpu(KvCache, kv_half_off + row_off + 2u * p2 + 1u);
            int cos_q30b = (int)RopeCosTable.Load<int64_t>((row_offset + p2) * 8u);
            int sin_q30b = (int)RopeSinTable.Load<int64_t>((row_offset + p2) * 8u);
            int64_t rkx, rky;
            RopeApplyPairGpu(kx, ky, cos_q30b, sin_q30b, rkx, rky);
            StoreSignedByteGpu(KvCache, kv_half_off + row_off + 2u * p2, (int)ClampRopeCodeGpu(rkx));
            StoreSignedByteGpu(KvCache, kv_half_off + row_off + 2u * p2 + 1u, (int)ClampRopeCodeGpu(rky));
        }
    }
}
