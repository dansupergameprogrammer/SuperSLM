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
// T-2113 (B4, design Sec3/Sec6.1, re-derived from Claude/Laplace/t2105-gpu-speed-ceiling-
// 2026-08-14.md Sec2 change 6): FLATTENED and GRIDDED. This site's work used to be
// partitioned over attention heads alone (`for (h = t; h < g_num_attention_heads; h +=
// 256)`) in a single group -- at the real 1.5B tier num_attention_heads is 12, so twelve of
// the group's 256 threads did every iteration and the other 244 returned immediately, and
// the whole dispatch was one group: twelve threads of a many-thousand-core card carrying the
// site. The real independent work item is one (head, key position) pair --
// num_attention_heads * width of them, each an independent head_dim-long dot product -- so
// the dispatch is now flattened over that item space and gridded across
// ceil(num_attention_heads * width / 256) groups (superslm_gpu.cpp computes the SAME product
// from the SAME `g_num_attention_heads`/`g_width` this shader reads, so the two cannot
// disagree about how many items exist). Each item's own computation and its own internal
// reduction order are UNCHANGED -- only which physical thread executes which item changes,
// the identical correctness argument the GEMM sites' own multi-group split rests on. There
// is no cross-thread state in this shader and no barrier, so a thread holding no item simply
// returns.
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
void main(uint3 dtid : SV_DispatchThreadID)
{
    uint t = dtid.x;
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

    // One work item per (head, key position): num_attention_heads * width of them, each an
    // independent head_dim-long dot product. The host dispatches
    // ceil(num_attention_heads * width / 256) groups from the SAME `width` root constant
    // this shader reads, so the two cannot disagree about how many items exist.
    uint items = g_num_attention_heads * (uint)width;
    if (t >= items) return;
    {
        uint h = t / (uint)width;
        uint kv_head = h / max(group, 1u);
        uint my_scores_base = scores_base_all + h * g_context_cap * 8u;
        {
            int k = (int)(t % (uint)width);
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
