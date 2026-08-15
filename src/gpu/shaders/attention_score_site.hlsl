// T-2045 (C3, Claude/Poirot/82cfca7-gpu-serial-port-build-review.md): the
// ratified attention-score site (site 5 of 16). §5.4/§13 place cross-site
// shader fusion explicitly outside this design's build target ("named as a
// future direction, not designed here, since nothing in this design or its
// grounding proves a fused shape correct") -- superseding T-2039's own
// four-site fusion (attention_site.hlsl, now retired), which collapsed
// attention-score/softmax/context-accumulate/ctx_fold into one dispatch and
// left the recorder emitting 14 dispatches/layer against the ratified 17
// (§5.8, D-SLM3069) every dependent figure (§5.4, §11 B10, §12 R1: 476/408
// dispatches/token) already assumes.
//
// score[k] = q_rot[h] . K_row(kv_head, position=k), k in [0, width) -- a pure
// int8 dot product, scale-independent, so this site can never reject (CPU
// carries no guard on the score computation itself; the funnel/domain checks
// live downstream at softmax and ctx_fold). Written to LayerScratch's own
// persistent `scores` region (ComputeScratchLayout's own `scores` field,
// `num_attention_heads * context_cap` int64 slots) rather than a transient
// WorkScratch slice, since the NEXT dispatch (softmax_site.hlsl) must read
// exactly what this one wrote.
//
// numthreads(256,1,1): attention heads partitioned across the group via
// N-per-thread striding, matching every other site's own per-output-channel
// parallelization (Sec5.4/Sec5.5) -- heads are this design's own independent
// production unit here, so no cross-thread cooperation is needed within one
// head's own score row.
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
ByteAddressBuffer   ScratchLayout  : register(t7);
RWByteAddressBuffer SeqState       : register(u0);
RWByteAddressBuffer LayerScratch   : register(u1);
RWByteAddressBuffer KvCache        : register(u2);
RWByteAddressBuffer WorkScratch    : register(u3);

[numthreads(256, 1, 1)]
void main(uint3 gtid : SV_GroupThreadID)
{
    uint t = gtid.x;
    int hidden_size = (int)g_hidden_size;
    uint sticky_off = SeqStickyOffGpu(hidden_size);
    int64_t sticky = SeqState.Load<int64_t>(sticky_off);
    if (sticky != kTagOk) return;

    int head_dim = (int)g_head_dim;
    int width = (int)g_width;
    uint group = (g_num_kv_heads > 0u) ? (g_num_attention_heads / g_num_kv_heads) : 1u;

    uint q_rot_off = ScratchLayout.Load<uint>(4 * 4);
    uint scores_base_all = ScratchLayout.Load<uint>(25 * 4);
    uint kv_half_off = KvHalfOffsetGpu(g_layer_index, g_context_cap, g_num_kv_heads, (uint)g_head_dim);

    for (uint h = t; h < g_num_attention_heads; h += 256u)
    {
        uint kv_head = h / max(group, 1u);
        uint my_scores_base = scores_base_all + h * g_context_cap * 8u;
        for (int k = 0; k < width; ++k)
        {
            uint k_row_off = KvRowOffsetWithinHalfGpu(g_context_cap, (uint)g_head_dim, kv_head, (uint)k);
            int64_t dot = 0;
            for (int d = 0; d < head_dim; ++d)
            {
                int qv = (int)LayerScratch.Load<int>(q_rot_off + (h * (uint)g_head_dim + (uint)d) * 4u);
                int kv = LoadSignedByteGpu(KvCache, kv_half_off + k_row_off + (uint)d);
                dot += (int64_t)qv * (int64_t)kv;
            }
            LayerScratch.Store<int64_t>(my_scores_base + (uint)k * 8u, dot);
        }
    }
}
